#coding:utf-8

import numpy as np
import math
import gym
from gym import spaces

ACTION_NUM = 28

class RobotPose(object):
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw


class RobotVelocity(object):
    def __init__(self, linear_v, angular_v):
        self.linear_v = linear_v
        self.angular_v = angular_v


class RobotState(object):
    def __init__(self, x, y, yaw, linear_v, angular_v):
        self.robot_position = RobotPose(x, y, yaw)
        self.robot_velocity = RobotVelocity(linear_v, angular_v)


class RobotAction(object):
    def __init__(self):
        self.action_space = spaces.Discrete(ACTION_NUM)
        self.cmd = [None] * ACTION_NUM
        self.cmd[0] = RobotVelocity(0.0, -0.6)
        self.cmd[1] = RobotVelocity(0.0, -0.4)
        self.cmd[2] = RobotVelocity(0.0, -0.2)
        self.cmd[3] = RobotVelocity(0.0, 0.0)
        self.cmd[4] = RobotVelocity(0.0, 0.2)
        self.cmd[5] = RobotVelocity(0.0, 0.4)
        self.cmd[6] = RobotVelocity(0.0, 0.6)
        self.cmd[7] = RobotVelocity(0.2, -0.6)
        self.cmd[8] = RobotVelocity(0.2, -0.4)
        self.cmd[9] = RobotVelocity(0.2, -0.2)
        self.cmd[10] = RobotVelocity(0.2, 0.0)
        self.cmd[11] = RobotVelocity(0.2, 0.2)
        self.cmd[12] = RobotVelocity(0.2, 0.4)
        self.cmd[13] = RobotVelocity(0.2, 0.6)
        self.cmd[14] = RobotVelocity(0.4, -0.6)
        self.cmd[15] = RobotVelocity(0.4, -0.4)
        self.cmd[16] = RobotVelocity(0.4, -0.2)
        self.cmd[17] = RobotVelocity(0.4, 0.0)
        self.cmd[18] = RobotVelocity(0.4, 0.2)
        self.cmd[19] = RobotVelocity(0.4, 0.4)
        self.cmd[20] = RobotVelocity(0.4, 0.6)
        self.cmd[21] = RobotVelocity(0.6, -0.6)
        self.cmd[22] = RobotVelocity(0.6, -0.4)
        self.cmd[23] = RobotVelocity(0.6, -0.2)
        self.cmd[24] = RobotVelocity(0.6, 0.0)
        self.cmd[25] = RobotVelocity(0.6, 0.2)
        self.cmd[26] = RobotVelocity(0.6, 0.4)
        self.cmd[27] = RobotVelocity(0.6, 0.6)

#        self.cmd[28] = RobotVelocity(-0.2, 0.0)
#        self.cmd[29] = RobotVelocity(-0.2, 0.2)
#        self.cmd[30] = RobotVelocity(-0.2, -0.2)
#        self.cmd[31] = RobotVelocity(-0.4, 0.0)
    
    def commander(self, i):
        return self.cmd[i]

