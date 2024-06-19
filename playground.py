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

def image_test():
    image = cv2.imread("input.jpg")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36,25,25), (70,255,255))
    a = 255
    image[mask > 0] = [a,a,a]
    image[mask <= 0] = [0,0,0]
    cv2.imwrite("result.png", image)


def split_data(rows, cols, data): # please use something that is divisible by our dimensions, otherwise it's gonna have an uneven end to its right and lower part
    
    steps_row = len(data) // rows
    steps_col = len(data[0]) // cols
    buffer = []
    for row in range(rows):
        buffer_row = []
        for col in range(cols):
            buffer_row.append(data[row * steps_row : (row + 1) * steps_row, col * steps_col : (col + 1) * steps_col])
        buffer.append(buffer_row)
    
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

    return buffer

data =  np.array([
    np.array([1,2,3,4,5,6,7,8]),
    np.array([1,2,3,4,5,6,7,8]),
    np.array([1,2,3,4,5,6,7,8]),
    np.array([1,2,3,4,5,6,7,8]),
    np.array([1,2,3,4,5,6,7,8]),
    np.array([1,2,3,4,5,6,7,8])
])

# print(split_data(2,3,data))
hey = split_data(1,2,data)
for row in hey:
    for col in row:
        print(sum(sum(col))) 