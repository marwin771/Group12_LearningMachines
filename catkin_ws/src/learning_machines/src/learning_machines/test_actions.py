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
            return torch.tensor([[random.uniform(-100, 100), random.uniform(-100, 100)]], device=device, dtype=torch.float64)

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


def get_reward(rob, starting_pos, total_left, total_right, left_speed, right_speed, move_time):
    global iterations_since_last_collision
    reward = 0
    irs = rob.read_irs()

    wheels = rob.read_wheels()
    left = wheels.wheel_pos_l
    right = wheels.wheel_pos_r
    current_pos = rob.get_position()
    distance = np.linalg.norm(np.array([current_pos.x, current_pos.y]) - np.array([starting_pos.x, starting_pos.y]))

    def penalty(left, right, irs):
        global iterations_since_last_collision
        iterations_since_last_collision += 1

        collision_treshold = 180

        collision_penalty = 1000
        reverse_penalty = 150

        buffer = 0
        # ring = [1,2,3,5,7,5,3,2,1]
        if left * right > 0: # if not turning in place
            for i in ([8,3,5,4,6] if left > 0 else [1,7,2]): # if front then front otherwise back
                if irs[i - 1] > collision_treshold: # if collision
                    iterations_since_last_collision = 0
                    return collision_penalty
                buffer += irs[i - 1]
        else: # if turning in place
            for i in range(8):
                if irs[i] > collision_treshold:
                    iterations_since_last_collision = 0
                    return collision_penalty
                buffer += 2 * irs[i]
        
        buffer += reverse_penalty if left < 0 and right < 0 else 0
        return buffer

    proximity_penalty = penalty(left_speed, right_speed, irs)
    consecutive_mult = 2 * np.arctan(5*iterations_since_last_collision/2) / np.pi + 1
    reward = distance * 5000 * consecutive_mult - proximity_penalty
    
    return torch.tensor([reward], device=device)

def run_training(rob: SimulationRobobo, controller: RobotNNController, num_episodes = 30, load_previous=False, moves=20):
    highest_reward = -float('inf')
    model_path = FIGRURES_DIR  / 'top.model'

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
        iterations_since_last_collision = 1
        print(f'Started Episode: {episode}')
        
        # Start the simulation
        rob.play_simulation()

        state = irs_to_state(rob)
        starting_pos = rob.get_position()
        total_reward = 0

        for t in count():
            sensor_readings.append(rob.read_irs())
            speeds = controller.select_action(state)
            left_speed, right_speed = speeds[0, 0].item(), speeds[0, 1].item()
            move_time = 100
            rob.reset_wheels()
            rob.move_blocking(int(left_speed), int(right_speed), move_time)
            next_state = irs_to_state(rob)
            wheels = rob.read_wheels()

            total_left += wheels.wheel_pos_l
            total_right += wheels.wheel_pos_r

            reward = get_reward(rob, starting_pos, total_left, total_right, left_speed, right_speed, move_time)
            total_reward += reward.item()

            controller.push(state, speeds, next_state, reward)
            state = next_state

            controller.optimize_model()

            if t > moves:
                rob.stop_simulation()
                break
        rewards.append(total_reward)
        controller.update_target()
        print(f"Episode: {episode}, Total Reward: {total_reward}")
        if total_reward > highest_reward:
            highest_reward = total_reward
            torch.save(controller.policy_net.state_dict(), model_path)
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
    run_training(rob, controller, num_episodes=30, load_previous=False, moves=20)
    generate_plots()

def run_task1_actions(rob):
    run_model(rob, controller)

def run_task0_actions(rob):
    print('Task 0 actions')
