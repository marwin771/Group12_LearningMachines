import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import numpy as np
import os
from itertools import count
import math
import cv2
import datetime

from typing import Literal

from data_files import FIGRURES_DIR

from robobo_interface import (
        IRobobo,
        SimulationRobobo
)

# Helper fns
def irs_to_state(rob: IRobobo, clamp = 250) -> torch.Tensor:
    # Clamp the IR values to 150
    return torch.tensor([list(map(lambda ir: ir if ir < clamp else clamp, rob.read_irs()))], device=device, dtype=torch.float)

def get_device_type() -> Literal['cuda', 'mps', 'cpu']: 
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

device = torch.device(get_device_type())

Transition = namedtuple('Transition', ('state', 'speeds', 'next_state', 'reward'))

# Define the replay memory
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the neural network
# Takes in the state and outputs the speeds for the left and right wheels
# n_observations is the size of the IR sensors
class WheelDQN(nn.Module):
    def __init__(self, n_observations):
        super(WheelDQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 2)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class RobotNNController:
    def __init__(self, n_observations, batch_size = 128, gamma = 0.99, lr=1e-4, memory_capacity=10000):
        self.steps_done = 0
        self.state_size = n_observations
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr

        self.policy_net = WheelDQN(n_observations).to(device)
        self.target_net = WheelDQN(n_observations).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(memory_capacity)

        self.target_net.eval()
        
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 1000 


    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        self.epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > self.epsilon_threshold:
            with torch.no_grad():
                return self.policy_net(state)
        else:
            return torch.tensor([[random.uniform(-50, 100), random.uniform(-50, 100)]], device=device, dtype=torch.float64)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def push(self, state, speeds, next_state, reward):
        self.memory.push(state, speeds, next_state, reward)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))


        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                     batch.next_state)), device=device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state
        #                                             if s is not None])

        state_batch = torch.cat(batch.state)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        
        state_speed_values = self.policy_net(state_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # # state value or 0 in case the state was final.
        # next_state_values = torch.zeros(self.batch_size, device=device)
        # with torch.no_grad():
        #     next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_speed_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

def get_camera_image(rob: IRobobo) -> torch.Tensor:
    image = rob.get_image_front()
    
    res = 192 # I'm too lazy to write a good split that equally splits for non-divisor numbers, so for the time being I need this to be divisible by 6
    # res = 96

    image = cv2.resize(image, (res, res))
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36,25,25), (70,255,255))

    image[mask > 0] = [1,1,1] # or [1,1,1] but then you can't view the result, so we'll take this extra step here
    image[mask <= 0] = [0,0,0]

    green_binary = image[:, :, 0]
    green_binary = image[:, :, None]

    return torch.tensor(green_binary, device=device, dtype=torch.float)

def split_data(rows, cols, data): # please use something that is divisible by our dimensions (natural powers of two), otherwise it's gonna cut off the end
    data = np.array(data)  # Ensure data is a numpy array
    steps_row = data.shape[0] // rows
    steps_col = data.shape[1] // cols
    buffer = []
    for row in range(rows):
        buffer_row = []
        for col in range(cols):
            slice_ = data[row * steps_row: (row + 1) * steps_row, col * steps_col: (col + 1) * steps_col]
            buffer_row.append(slice_)
        buffer.append(np.array(buffer_row))

    '''
    1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1
    ------------------|------------------
    1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1
    ------------------|------------------
    1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1
    '''

    return np.array(buffer)

def get_reward(rob_after_movement, starting_pos, left_speed, right_speed, irs_before_movement, image_before_movement, move_time):

    global food_consumed

    # hypers
    collision_treshold = 150
    distance_between_wheels = 10 # cm

    # point -> reward
    # see green in front (middle col but not only lower) and goes front -> reward
    # sees green in lower front (middle col but only lower) and floors it -> reward
    # doesnt see green in front but sees green in either side and turns towards
    
    
    

    # math (these are calculated when we're not going backwards, it might work for backwards movement but I'm too tired to think about it, so only use it with forward movement)
    radius = 0 if left_speed == right_speed else ((left_speed + right_speed) * distance_between_wheels) / (2 * np.abs(left_speed - right_speed))
    angular_velocity = np.abs(left_speed - right_speed) / (distance_between_wheels / 2) 
    angle_rad = angular_velocity * move_time / 1000 # in seconds
    turn = "left" if left_speed < right_speed else ("right" if right_speed < left_speed else "forward")
    turn_angle = angle_rad / 2

    irs_after_movement = rob_after_movement.read_irs()
    
    current_pos = rob_after_movement.get_position()
    wheels = rob_after_movement.read_wheels()

    left = wheels.wheel_pos_l
    right = wheels.wheel_pos_r
    
    front = np.array([8,3,5,4,6])
    back = np.array([1,7,2])

    distance = np.linalg.norm(np.array([current_pos.x, current_pos.y]) - np.array([starting_pos.x, starting_pos.y]))

    one_by_two = split_data(1, 2, image_before_movement)
    three_by_three = split_data(3, 3, image_before_movement)
    res = 192
    all_pixels = res * res

    reward = 0

    if food_consumed < rob_after_movement.nr_food_collected():
        reward += 4500 * (rob_after_movement.nr_food_collected() - food_consumed)
        food_consumed = rob_after_movement.nr_food_collected()


    

    def how_forward_coefficient(left, right):
        return 1/(1/6 * np.abs(left - right) + 1) + 1


    # here comes the ugly if tree because I'm not gonna go over the logic to optimise the amount of lines written at 3:23 AM. For more see: https://en.wikipedia.org/wiki/Gordian_Knot
    if right_speed > 0 and left_speed > 0: # going forward in any way
        if np.sum(three_by_three[:, 1]) / (all_pixels / 3) > 0.05: # the middle column has considerable green
            reward += 500 * how_forward_coefficient(left_speed, right_speed)
        if np.sum(three_by_three[2, 1]) > 0 and left_speed > 50 and right_speed > 50: # the bottom middle has any green in it
            reward += 500
    elif right_speed * left_speed <= 0 and np.sum(three_by_three[:, 1]) / (all_pixels / 3) < 0.05: # turning when there is nothing in front of us
        reward += 200
        if np.sum(one_by_two[0, 0]) / (all_pixels / 2) < 0.01 and np.sum(one_by_two[0, 1]) / (all_pixels / 2) < 0.01: # there is less than considerable green to both the left and the right
            reward += 300
        elif np.sum(one_by_two[0, 0]) > np.sum(one_by_two[0, 1]): # we have considerable green somewhere and more on the left
            if left_speed < right_speed: # and we turn left
                reward += 400
            # else:
            #     reward -= 200
        else: # considerable green but more to the right
            if right_speed < left_speed: # and we turn right
                reward += 400
    elif right_speed < 0 and left_speed < 0: # going in reverse
        if np.sum(three_by_three[:,:]) <= 10 and sum(irs_before_movement[0, front - 1]) > 3 * collision_treshold: # nothing green in front and we most likely hit something
            reward += 500 * how_forward_coefficient(left_speed, right_speed)
        else:
            reward -= 200


        # if (there was a non-wall object in front of us and we didn't turn (/ we went towards it))
        # alternatively: there was a non-wall object in front of us AND [there still is a non-wall object OR we bumped into it to collect]
            # increase reward
            # increase reward further if we bumped into it
            # if we didn't, increase reward further if an object in front of us is now in close proximity (so not a wall)
        # elif (there was an object at the angle where we turned)
            # increase reward for good turn:  in respect to how good the turn was / how in-front-of-us the object is
            # maybe increase reward for how in 
        # if (there wasn't anything in our proximity)
            # increase reward by how far it moved (so how fast the wheels moved)
    
    return torch.tensor([reward], device=device)

def run_training(rob: SimulationRobobo, controller: RobotNNController, num_episodes = 30, load_previous=False, moves=20):
    highest_reward = -float('inf')
    model_path = FIGRURES_DIR 

    total_left, total_right = 0.0, 0.0

    global iterations_since_last_collision
    global sensor_readings
    sensor_readings = []
    global rewards
    rewards = []

    if load_previous and os.path.exists(model_path):
        controller.policy_net.load_state_dict(torch.load(model_path))
        controller.target_net.load_state_dict(controller.policy_net.state_dict())
        print("Loaded saved model.")

    for episode in range(num_episodes):
        # iterations_since_last_collision = 1

        print(f'Started Episode: {episode}')
        
        # Start the simulation
        rob.play_simulation()
        rob.sleep(0.5)

        state = irs_to_state(rob)
        camera_state = get_camera_image(rob)
        starting_pos = rob.get_position()
        total_reward = 0

        global food_consumed
        food_consumed = 0

        for t in count():
            # state here is what we see before moving
            sensor_readings.append(rob.read_irs())
            speeds = controller.select_action(state)
            left_speed, right_speed = speeds[0, 0].item(), speeds[0, 1].item() # choose a movement
            move_time = 100
            rob.reset_wheels()
            rob.move_blocking(int(left_speed), int(right_speed), move_time) # execute movement
            next_state = irs_to_state(rob) # what we see after moving
            next_camera_state = get_camera_image(rob)
            wheels = rob.read_wheels()

            total_left += wheels.wheel_pos_l
            total_right += wheels.wheel_pos_r
            
            # reward gets rob (after moving), left_speed and right_speed (of the last movement),
            reward = get_reward(rob, starting_pos, left_speed, right_speed, state, camera_state, move_time)
            total_reward += reward.item()

            controller.push(state, speeds, next_state, reward)
            state = next_state
            camera_state = next_camera_state

            controller.optimize_model()

            if t > moves:
                rob.stop_simulation()
                break
        rewards.append(total_reward)
        controller.update_target()
        print(f"Episode: {episode}, Total Reward: {total_reward}")
        if total_reward > highest_reward:
            highest_reward = total_reward
            torch.save(controller.policy_net.state_dict(), model_path / f"{datetime.datetime.now()}")
            print(f"Saved best model with highest reward: {highest_reward}")


def clamp(n, smallest, largest): 
    if n < 0:
        return max(n, smallest)
    return min(n, largest)

def run_model(rob: IRobobo, controller: RobotNNController):
    # load the model
    model_path = FIGRURES_DIR  / 'top_hardware.model'
    controller.policy_net.load_state_dict(torch.load(model_path))
    controller.target_net.load_state_dict(controller.policy_net.state_dict())
    controller.policy_net.eval()

    # Start the simulation
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    state = irs_to_state(rob)
    
    collisions = 0
    still_colliding = False

    while True:
        speeds = controller.select_action(state)
        left_speed, right_speed = speeds[0, 0].item(), speeds[0, 1].item()
        print(f"Speeds: {left_speed}, {right_speed}")
        if isinstance(rob, SimulationRobobo):
            move_time = 100
        else:
            move_time = 500
        rob.reset_wheels()
        
        rob.move_blocking(clamp(int(left_speed), -100, 100), clamp(int(right_speed), -100, 100), move_time)
        next_state = irs_to_state(rob)
        state = next_state

        if rob.read_irs()[0] > 250 and not still_colliding:
            collisions += 1
            still_colliding = True
            print(f"Collisions: {collisions}")
        elif rob.read_irs()[0] < 250:
            still_colliding = False

        # Exit on collision
        # if collisions > 9:
        #     break

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


# Initialize the agent and run the simulation
# n_observations = 8 IR sensors
controller = RobotNNController(n_observations=8, memory_capacity=10000, batch_size=64, gamma=0.99, lr=1e-3)

def generate_plots():
    global sensor_readings
    global rewards
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme()
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    sensor_readings = np.array(sensor_readings)
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(8):
        ax.plot(sensor_readings[:, i], label=f"IR {i+1}")

    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("IR Sensor Value")
    ax.set_title("IR Sensor Readings Over Time")
    
    # save the figure to the figures directory
    plt.savefig(FIGRURES_DIR / 'sensor_readings_training.png')

    # Plot the rewards
    rewards = np.array(rewards)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(rewards)
    ax.set_xlabel("Time")
    ax.set_ylabel("Reward")
    ax.set_title("Rewards Over Time")
    plt.savefig(FIGRURES_DIR / 'rewards_training.png')

def run_all_actions(rob):
    # rob.moveTiltTo(100, 100, True)
    # rob.sleep(5)
    # rob.startCamera()
    run_training(rob, controller, num_episodes=30, load_previous=False, moves=40)
    generate_plots()

def run_task1_actions(rob):
    run_model(rob, controller)

def run_task0_actions(rob):
    print('Task 0 actions')
