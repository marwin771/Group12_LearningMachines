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

from typing import Literal, Tuple

from data_files import FIGRURES_DIR

from robobo_interface import (
        IRobobo,
        SimulationRobobo,
        Position,
)

# Helper fns
def irs_to_state(rob: IRobobo, clamp = 250) -> torch.Tensor:
    # Clamp the IR values to 150
    return torch.tensor(irs_clamped(rob=rob, clamp=clamp), device=device, dtype=torch.float)

def irs_clamped(rob: IRobobo, clamp = 250):
    return [ir if ir < clamp else clamp for ir in rob.read_irs()]

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
        # print(f'Input dim: {x.shape}')
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return torch.sigmoid(x) * 100


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
        self.epsilon_end = 0.1
        self.epsilon_decay = 30000


    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        self.epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > self.epsilon_threshold:
            with torch.no_grad():
                return self.policy_net(state)
        else:
            return torch.tensor([random.uniform(0, 100), random.uniform(0, 100)], device=device, dtype=torch.float64)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def push(self, state, speeds, next_state, reward):
        # print(f'State: {state.shape}, Speeds: {speeds.shape}, Next State: {next_state.shape}, Reward: {reward.shape}')
        self.memory.push(state, speeds, next_state, reward)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert batch-array of Transitions to Transition of batch-arrays
        state_batch = torch.stack(batch.state)
        reward_batch = torch.stack(batch.reward)
        next_state_batch = torch.stack(batch.next_state)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_speed_values = self.policy_net(state_batch)
        # print(f'State speed values: {state_speed_values.shape}')

        # Compute V(s_{t+1}) for all next states.
        res: torch.Tensor =  self.target_net(next_state_batch)
        max_prediction = res.max(1, keepdim=True)
        next_state_values = max_prediction[0]
        # print(f'Max prediction: {max_prediction}')
        # print(f'Next state values: {next_state_values.shape}')

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        # print(f'Expected state action values: {expected_state_action_values.repeat(1, 2).shape}')
        # print(f'Expected state action values: {expected_state_action_values.unsqueeze(1).shape}')
        loss = F.smooth_l1_loss(state_speed_values, expected_state_action_values.repeat(1, 2))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

def get_camera_image(rob: IRobobo) -> Tuple[torch.Tensor, torch.Tensor]:
    image = rob.get_image_front()
    
    res = 192 # I'm too lazy to write a good split that equally splits for non-divisor numbers, so for the time being I need this to be divisible by 6
    # res = 96

    image = cv2.resize(image, (res, res))
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36,25,25), (70,255,255))
    mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))

    a = 1
    masked_green = image.copy()
    masked_green[mask_green > 0] = [a,a,a]
    masked_green[mask_green <= 0] = [0, 0, 0]
    
    masked_red1 = image.copy()
    masked_red1[mask_red1 > 0] = [a,a,a]
    masked_red1[mask_red1 <= 0] = [0, 0, 0]

    masked_red2 = image.copy()
    masked_red2[mask_red2 > 0] = [a,a,a]
    masked_red2[mask_red2 <= 0] = [0, 0, 0]

    masked_red = masked_red1 + masked_red2
    
    # cv2.imwrite(FIGRURES_DIR / "test1.png", masked_green)
    # cv2.imwrite(FIGRURES_DIR / "test2.png", masked_red)
    # cv2.imwrite(FIGRURES_DIR / "test0.png", image)


    # image[mask > 0] = [1,1,1] # or [1,1,1] but then you can't view the result, so we'll take this extra step here
    # image[mask <= 0] = [0,0,0]

    green_binary = masked_green[:, :, 0]
    red_binary = masked_red[:, :, 0]

    green_split = split_data(3, 3, green_binary)
    red_split = split_data(3, 3, red_binary)

    green_3x3 = np.zeros((3, 3), dtype=int)
    red_3x3 = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for k in range(3):
            green_3x3[i, k] = 1 if np.sum(green_split[i,k]) > 0 else 0
            red_3x3[i, k] = 1 if np.sum(red_split[i,k]) > 0 else 0
            
    return green_3x3, red_3x3

def get_camera_image_as_tensor(rob: IRobobo) -> Tuple[torch.Tensor, torch.Tensor]:
    green_binary, red_binary = get_camera_image(rob)
    return torch.tensor(green_binary, device=device, dtype=torch.float), torch.tensor(red_binary, device=device, dtype=torch.float) 

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


def calculate_percentage_of_white(img):
    return np.sum(img == 1) / np.prod(img.shape)

def is_stuck(starting_pos: Position, current_pos: Position, threshold = 3) -> bool:
    if abs(starting_pos.x - current_pos.x) < threshold and abs(starting_pos.y - current_pos.y) < threshold:
        return True
    return False

def get_distance(p1: Position, p2: Position) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def is_equal(p1: Position, p2: Position) -> bool:
    return p1.x == p2.x and p1.y == p2.y

def get_reward(rob: SimulationRobobo, starting_pos: Position, base_position: Position, previous_food_position: Position, move, moves_in_episode):
    global food_consumed

    # current_pos = rob.get_position()
    current_food_pos = rob.food_position()

    green, red = get_camera_image(rob)
    green_percentage = calculate_percentage_of_white(green)
    red_percentage = calculate_percentage_of_white(red)
    # print(f"Green: {green_percentage * 100}, Red: {red_percentage * 100}")
    reward = 0


    if rob.nr_food_collected() > 0 and not food_consumed:
        print('Food collected')
        food_consumed = True
        reward += 0.5
    elif rob.nr_food_collected() > 0 and not is_equal(current_food_pos, previous_food_position): # if we have food, we want to move towards the base
        reward += 1 * green_percentage
        reward += 0.5 * (get_distance(previous_food_position, base_position) - get_distance(current_food_pos, base_position))
    else: # if we don't have food, we want to move towards the food
        reward += 1 * red_percentage

    if rob.base_detects_food():
        reward += 100 * ((moves_in_episode - move) / moves_in_episode)

    return torch.tensor([reward], device=device)



def run_training(rob1: SimulationRobobo, rob2: SimulationRobobo, controller: RobotNNController, num_episodes = 200, load_previous=False, moves=50, swap_chance = 0.1):
    highest_reward = -float('inf')
    model_path = FIGRURES_DIR 

    total_left, total_right = 0.0, 0.0

    global iterations_since_last_collision
    global sensor_readings
    sensor_readings = []
    global rewards
    rewards = []
    global food_consumed

    rob = rob1

    if load_previous and os.path.exists(model_path):
        controller.policy_net.load_state_dict(torch.load(model_path))
        controller.target_net.load_state_dict(controller.policy_net.state_dict())
        print("Loaded saved model.")

    for episode in range(num_episodes):
        if random.random() <= swap_chance:
            print('Swapping robots')
            if rob._api_port == rob1._api_port:
                rob = rob2
            else:
                rob = rob1

        print(f'Started Episode: {episode}')
        
        # Start the simulation
        rob.play_simulation()
        rob.sleep(0.5)
        rob.set_phone_tilt_blocking(109, 1000)

        global starting_base_food_distance
        starting_base_food_distance = -1
        base_position = rob.base_position()

        state = irs_to_state(rob)

        green_camera_state, red_camera_state = get_camera_image_as_tensor(rob)
        starting_pos = rob.get_position()

        state = torch.cat((state.view(-1), green_camera_state.view(-1), red_camera_state.view(-1)))
        # print(f'Starting state dim: {state.shape}')

        total_reward = 0

        global food_consumed
        food_consumed = False
        moves_in_ep = moves + (episode // 5) * 5

        for t in count():
            # state here is what we see before moving
            sensor_readings.append(rob.read_irs())

            previous_position = rob.get_position()
            previous_food_position = rob.food_position()

            speeds = controller.select_action(state)
            left_speed, right_speed = speeds[0].item(), speeds[1].item() # choose a movement
            move_time = 100
            rob.reset_wheels()
            rob.move_blocking(int(left_speed), int(right_speed), move_time) # execute movement
            next_state = irs_to_state(rob) # what we see after moving
            next_green_camera_state, next_red_camera_state = get_camera_image_as_tensor(rob)
            next_state = torch.cat((next_state.view(-1), next_green_camera_state.view(-1), next_red_camera_state.view(-1)))
            # print(f'Next state dim: {next_state.shape}')
            wheels = rob.read_wheels()

            total_left += wheels.wheel_pos_l
            total_right += wheels.wheel_pos_r
            
            # reward gets rob (after moving), left_speed and right_speed (of the last movement),
            reward = get_reward(rob, previous_position, base_position, previous_food_position, t, moves_in_ep)
            total_reward += reward.item()

            controller.push(state, speeds, next_state, reward)
            state = next_state
            green_camera_state = next_green_camera_state
            red_camera_state = next_red_camera_state

            controller.optimize_model()

            if t > moves_in_ep or rob.base_detects_food():
                rob.stop_simulation()
                break
        rewards.append(total_reward)
        controller.update_target()
        print(f"Episode: {episode}, Total Reward: {total_reward}")
        if total_reward > highest_reward:
            highest_reward = total_reward
            torch.save(controller.policy_net.state_dict(), model_path / f"{datetime.datetime.now()}.model")
            print(f"Saved best model with highest reward: {highest_reward}")


def clamp(n, smallest, largest): 
    if n < 0:
        return max(n, smallest)
    return min(n, largest)

def run_model(rob: IRobobo, controller: RobotNNController, model_name: str = 'top_hardware.model'):
    # load the model
    model_path = FIGRURES_DIR  / model_name
    controller.policy_net.load_state_dict(torch.load(model_path))
    controller.target_net.load_state_dict(controller.policy_net.state_dict())
    controller.policy_net.eval()

    # Start the simulation
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    rob.set_phone_tilt_blocking(100, 100)
    rob.sleep(0.5)

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
controller = RobotNNController(n_observations=(8 + 2 * 3 * 3), memory_capacity=10000, batch_size=64, gamma=0.99, lr=1e-3)

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

def run_all_actions():
    # rob.moveTiltTo(100, 100, True)
    # rob.sleep(5)
    # rob.startCamera()
    rob1 = SimulationRobobo()
    # rob2 = SimulationRobobo(api_port=23001)
    rob2 = None
    # rob2 = None
    run_training(rob1, rob2, controller, num_episodes=150, load_previous=False, moves=150, swap_chance=-1)
    generate_plots()

def run_task1_actions(rob, model_name = None):
    if model_name is None:
        run_model(rob, controller)
    else:
        run_model(rob, controller, model_name)

def run_task0_actions(rob):
    print('Task 0 actions')
