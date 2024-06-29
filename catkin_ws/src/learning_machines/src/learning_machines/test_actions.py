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
    SimulationRobobo
)


# Helper fns
def irs_to_state(rob: IRobobo, clamp=250) -> torch.Tensor:
    # Clamp the IR values to 150
    return torch.tensor([list(map(lambda ir: ir if ir < clamp else clamp, rob.read_irs()))], device=device,
                        dtype=torch.float)


def get_device_type() -> Literal['cuda', 'mps', 'cpu']:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


device = torch.device(get_device_type())

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

ACTIONS = {
    0: (50, 50),    # Forward
    1: (0, 25),     # Left
    2: (25, 0),     # Right
    3: (0, 50),   # Turn more left
    4: (50, 0)    # Turn more right
}


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
# Takes in the state and outputs the action index
# n_observations is the size of the IR sensors
class WheelDQN(nn.Module):
    def __init__(self, n_observations):
        super(WheelDQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 5)  # Output 5 actions

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class RobotNNController:
    def __init__(self, n_observations, batch_size=128, gamma=0.99, lr=1e-4, memory_capacity=10000):
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

    def select_action(self, state: torch.Tensor, green_image: torch.Tensor, red_image: torch.Tensor,
                      food_collected: bool) -> int:
        sample = random.random()
        self.epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if food_collected:
            if green_image.sum() > 0:  # If there is any green detected
                first_column = green_image[:, 0].sum().item()
                middle_column = green_image[:, green_image.shape[1] // 2].sum().item()
                last_column = green_image[:, -1].sum().item()

                if middle_column > first_column and middle_column > last_column:
                    return 0  # Move forward if green is in the middle
                elif first_column > last_column:
                    return 1  # Turn left
                elif last_column > first_column:
                    return 2  # Turn right
            else:
                # No green detected, turn more
                return random.choice([3, 4])
        else:
            if red_image.sum() > 0:  # If there is any red detected
                first_column = red_image[:, 0].sum().item()
                middle_column = red_image[:, red_image.shape[1] // 2].sum().item()
                last_column = red_image[:, -1].sum().item()

                if middle_column > first_column and middle_column > last_column:
                    return 0  # Move forward if red is in the middle
                elif first_column > last_column:
                    return 1  # Turn left
                elif last_column > first_column:
                    return 2  # Turn right
            else:
                # No red detected, turn more
                return random.choice([3, 4])

        if sample > self.epsilon_threshold:
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).item()
        else:
            return random.randint(0, 4)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def push(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, device=device)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def get_camera_image(rob: IRobobo) -> Tuple[torch.Tensor, torch.Tensor]:
    image = rob.get_image_front()

    res = 192  # I'm too lazy to write a good split that equally splits for non-divisor numbers, so for the time being I need this to be divisible by 6

    image = cv2.resize(image, (res, res))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
    mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))

    a = 255
    masked_green = image.copy()
    masked_green[mask_green > 0] = [a, a, a]
    masked_green[mask_green <= 0] = [0, 0, 0]

    masked_red1 = image.copy()
    masked_red1[mask_red1 > 0] = [a, a, a]
    masked_red1[mask_red1 <= 0] = [0, 0, 0]

    masked_red2 = image.copy()
    masked_red2[mask_red2 > 0] = [a, a, a]
    masked_red2[mask_red2 <= 0] = [0, 0, 0]

    masked_red = masked_red1 + masked_red2

    green_binary = masked_green[:, :, 0]
    green_binary = masked_green[:, :, None]

    red_binary = masked_red[:, :, 0]
    red_binary = masked_red[:, :, None]

    return torch.tensor(green_binary, device=device, dtype=torch.float), torch.tensor(red_binary, device=device,
                                                                                      dtype=torch.float)


def split_data(rows, cols, data):
    data = np.array(data)  # Ensure data is a numpy array
    steps_row = data.shape[0] // rows
    steps_col = data.shape[1] // cols
    buffer = []
    for row in range(rows):
        buffer_row = []
        for col in range(cols):
            slice_ = data[row * steps_row: (row + 1) * steps_row, col * steps_col: (col + 1) * steps_col]
            buffer_row.append(torch.tensor(slice_, device=device, dtype=torch.float))
        buffer.append(torch.stack(buffer_row))
    return torch.stack(buffer)


def get_reward(rob_after_movement, starting_pos, left_speed, right_speed, irs_before_movement,
               green_image_before_movement, red_image_before_movement, move_time, base_position):
    global food_consumed
    global starting_base_food_distance
    # hypers
    collision_treshold = 150
    distance_between_wheels = 10  # cm

    # math (these are calculated when we're not going backwards, it might work for backwards movement but I'm too tired to think about it, so only use it with forward movement)
    radius = 0 if left_speed == right_speed else ((left_speed + right_speed) * distance_between_wheels) / (
                2 * np.abs(left_speed - right_speed))
    angular_velocity = np.abs(left_speed - right_speed) / (distance_between_wheels / 2)
    angle_rad = angular_velocity * move_time / 1000  # in seconds
    turn = "left" if left_speed < right_speed else ("right" if right_speed < left_speed else "forward")
    turn_angle = angle_rad / 2

    irs_after_movement = irs_to_state(rob_after_movement)
    current_pos = rob_after_movement.get_position()
    wheels = rob_after_movement.read_wheels()

    left = wheels.wheel_pos_l
    right = wheels.wheel_pos_r

    front = np.array([8, 3, 5, 4, 6])
    back = np.array([1, 7, 2])

    distance = np.linalg.norm(np.array([current_pos.x, current_pos.y]) - np.array([starting_pos.x, starting_pos.y]))

    one_by_two = split_data(1, 2, green_image_before_movement)
    three_by_three = split_data(3, 3, green_image_before_movement)
    two_by_three = split_data(2, 3, red_image_before_movement)  # CAREFUL.. THIS ONE IS RED!!
    res = 192
    all_pixels = res * res

    reward = 0

    food_pickup_reward = 0
    three_by_three_reward = 0
    center_column_red_reward = 0
    base_food_progression_reward = 0
    ir_sensor_reward = 0
    base_detection_reward = 0

    if food_consumed == 0 and irs_after_movement[0, 4] > 100 and torch.sum(irs_after_movement[0, torch.tensor(
            [7, 5])]) < 20:  # when we are literally just getting food rn (runs into this only once... hopefully)
        food_consumed = 1
        starting_base_food_distance = np.linalg.norm(
            np.array([current_pos.x, current_pos.y]) - np.array([base_position.x, base_position.y]))
        food_pickup_reward = 100
        reward += food_pickup_reward  # yippie
        print("PICKED UP")

    if not starting_base_food_distance == -1:  # if we have a starting distance we can get down to defining things, supposedly if this happens food_consumed will always be > 0
        base_food_distance = np.linalg.norm(
            np.array([current_pos.x, current_pos.y]) - np.array([base_position.x, base_position.y]))
        base_food_progression = 1 - base_food_distance / starting_base_food_distance
    else:
        base_food_progression = 0

    if food_consumed > 0:  # supposedly we have the stuff in the thing
        three_by_three_reward = np.sqrt(5 * torch.sum(three_by_three) / all_pixels).item()  # 0 to 1 (sort of)
        base_food_progression_reward = base_food_progression  # 1 if standing on it, 0 if the food is the same distance away as we started and negative if it's further away
        ir_sensor_reward = 1 if irs_after_movement[0, 4] > 20 and torch.sum(
            irs_after_movement[0, torch.tensor([7, 2, 3, 5])]) < 30 else 0
        base_detection_reward = 2000 if rob_after_movement.base_detects_food() else 0  # woohooo
        reward += three_by_three_reward
        reward += base_food_progression_reward
        reward += ir_sensor_reward
        reward += base_detection_reward
    else:
        three_by_three_reward = np.sqrt(20 * torch.sum(two_by_three) / all_pixels).item()
        center_column_red_reward = np.sqrt(8 * 3 * torch.sum(
            two_by_three[:, 1]) / all_pixels).item()  # np.sqrt(3 * 6 * torch.sum(two_by_three[1,1]) / all_pixels)
        ir_sensor_reward = 1 if irs_after_movement[0, 4] > 20 and torch.sum(
            irs_after_movement[0, torch.tensor([7, 2, 3, 5])]) < 30 else 0
        base_detection_reward = 0
        reward += three_by_three_reward
        reward += center_column_red_reward
        reward += ir_sensor_reward

    # Print each reward component
    print(f"Food Pickup Reward: {food_pickup_reward}")
    print(f"3x3 detection reward: {three_by_three_reward}")
    print(f"Center red reward: {center_column_red_reward}")
    print(f"Base food progression Reward: {base_food_progression_reward}")
    print(f"IR sensor reward: {ir_sensor_reward}")
    print(f"Base detection reward: {base_detection_reward}")
    print(f"TOTAL: {reward}")

    return torch.tensor([reward], device=device)


def run_training(rob: SimulationRobobo, controller: RobotNNController, num_episodes=30, load_previous=False, moves=20):
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
        print(f'Started Episode: {episode}')

        # Start the simulation
        rob.play_simulation()
        rob.sleep(0.5)
        rob.set_phone_tilt_blocking(150, 100)

        global starting_base_food_distance
        starting_base_food_distance = -1
        base_position = rob.base_position()

        state = irs_to_state(rob)
        green_camera_state, red_camera_state = get_camera_image(rob)
        starting_pos = rob.get_position()
        total_reward = 0

        global food_consumed
        food_consumed = 0

        for t in count():
            # state here is what we see before moving
            sensor_readings.append(rob.read_irs())
            action = controller.select_action(state, green_camera_state, red_camera_state, food_consumed > 0)
            left_speed, right_speed = ACTIONS[action]
            move_time = 100
            rob.reset_wheels()
            rob.move_blocking(int(left_speed), int(right_speed), move_time)  # execute movement

            print("IRS data: ", rob.read_irs())


            next_state = irs_to_state(rob)  # what we see after moving
            next_green_camera_state, next_red_camera_state = get_camera_image(rob)
            wheels = rob.read_wheels()

            total_left += wheels.wheel_pos_l
            total_right += wheels.wheel_pos_r

            reward = get_reward(rob, starting_pos, left_speed, right_speed, state, green_camera_state, red_camera_state,
                                move_time, base_position)
            total_reward += reward.item()

            controller.push(state, action, next_state, reward)
            state = next_state
            green_camera_state = next_green_camera_state
            red_camera_state = next_red_camera_state

            controller.optimize_model()

            if t > moves + (episode // 5) * 5:
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
    model_path = FIGRURES_DIR / 'best.model'

    controller.policy_net.load_state_dict(torch.load(model_path))
    controller.target_net.load_state_dict(controller.policy_net.state_dict())
    controller.policy_net.eval()

    # Start the simulation
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        rob.set_phone_tilt_blocking(150, 100)

    state = irs_to_state(rob)

    collisions = 0
    still_colliding = False
    food_collected = False  # Initialize food_collected state

    while True:
        # Capture the camera images
        green_image, red_image = get_camera_image(rob)

        # Determine if food is collected
        #food_collected = rob.base_detects_food()

        action_idx = controller.select_action(state, green_image, red_image, food_collected)
        left_speed, right_speed = ACTIONS[action_idx]
        print(f"Action: {action_idx}")

        if isinstance(rob, SimulationRobobo):
            move_time = 100
        else:
            move_time = 500

        rob.reset_wheels()
        rob.move_blocking(clamp(int(left_speed), -100, 100), clamp(int(right_speed), -100, 100), move_time)
        next_state = irs_to_state(rob)
        state = next_state

        # Print IR sensor data
        print("IRS data: ", rob.read_irs())

        if rob.read_irs()[0] > 250 and not still_colliding:
            collisions += 1
            still_colliding = True
            print(f"Collisions: {collisions}")
        elif rob.read_irs()[0] < 250:
            still_colliding = False

        # Exit on collision
        if collisions > 9:
            break

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
        ax.plot(sensor_readings[:, i], label=f"IR {i + 1}")

    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("IR Sensor Value")
    ax.set_title("IR Sensor Readings Over Time")

    plt.savefig(FIGRURES_DIR / 'sensor_readings_training.png')

    rewards = np.array(rewards)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(rewards)
    ax.set_xlabel("Time")
    ax.set_ylabel("Reward")
    ax.set_title("Rewards Over Time")
    plt.savefig(FIGRURES_DIR / 'rewards_training.png')


def run_all_actions(rob):
    #run_training(rob, controller, num_episodes=15, load_previous=False, moves=10000)
    #generate_plots()

    run_model(rob, controller)


def run_task1_actions(rob):
    run_model(rob, controller)


def run_task0_actions(rob):
    print('Task 0 actions')
