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

from typing import Literal

def image_test(num = '', type = 'png'):
    image = cv2.imread(f"inputs/input{num}.{type}")
    # res = 96
    res = 192 # I'm too lazy to write a good split that equally splits for non-divisor numbers, so for the time being I need this to be divisible by 6
    image = cv2.resize(image, (res, res))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36,25,25), (70,255,255))
    a = 255
    image[mask > 0] = [a,a,a]
    image[mask <= 0] = [0,0,0]
    
    image[64, :] = [0,0,255]
    image[128, :] = [0,0,255]
    image[:, 64] = [0,0,255]
    image[:, 128] = [0,0,255]

    cv2.imwrite(f"playground_results/result{num}.png", image)


import numpy as np

def split_data(rows, cols, data):
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
    return np.array(buffer)

    
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


# for i in range(1,7):
#     image_test(i)

data =  np.array([
    np.array([1,2,3,4,5,6,7,8,9]),
    np.array([1,2,3,4,5,6,7,8,9]),
    np.array([1,2,3,4,5,6,7,8,9]),
    np.array([1,2,3,4,5,6,7,8,9]),
    np.array([1,2,3,1,1,1,7,8,9]),
    np.array([1,2,3,4,5,6,7,8,9])
])

print(split_data(3,2,data))
hey = split_data(2,2,data)
three_by_three = split_data(3, 3, data)

print(sum(sum(sum(three_by_three[2:, 1]))))

for row in hey:
    for col in row:
        print(sum(sum(col))) 

print(sum(sum(sum(three_by_three))))