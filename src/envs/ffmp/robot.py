#coding:utf-8

import numpy as np
import math

class RobotPose(object):
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw


class RobotVelocity(object):
    def __init__(self, linear_v, angular_v):
        self.linear = linear_v
        self.angular = angular_v


class RobotState(RobotPosition, RobotVelocity):
    def __init__(self, x, y, yaw, linear_v, angular_v):
        self.robot_position = RobotPosition(x, y, yaw)
        self.robot_velocity = RobotVelocity(linear_v, angular_v)


class RobotAction(object):
    def __init__(self):
        self.cmd = [RobotVelocity()] * 28
        self.cmd[0].linear = 0.0
        self.cmd[0].angular = -0.9
        self.cmd[1].linear = 0.0
        self.cmd[1].angular = -0.6
        self.cmd[2].linear = 0.0
        self.cmd[2].angular = -0.3
        self.cmd[3].linear = 0.0
        self.cmd[3].angular = 0.0
        self.cmd[4].linear = 0.0
        self.cmd[4].angular = 0.3
        self.cmd[5].linear = 0.0
        self.cmd[5].angular = 0.6
        self.cmd[6].linear = 0.0
        self.cmd[6].angular = 0.9
        self.cmd[7].linear = 0.2
        self.cmd[7].angular = -0.9
        self.cmd[8].linear = 0.2
        self.cmd[8].angular = -0.6
        self.cmd[9].linear = 0.2
        self.cmd[9].angular = -0.3
        self.cmd[10].linear = 0.2
        self.cmd[10].angular = 0.0
        self.cmd[11].linear = 0.2
        self.cmd[11].angular = 0.3
        self.cmd[12].linear = 0.2
        self.cmd[12].angular = 0.6
        self.cmd[13].linear = 0.2
        self.cmd[13].angular = 0.9
        self.cmd[14].linear = 0.4
        self.cmd[14].angular = -0.9
        self.cmd[15].linear = 0.4
        self.cmd[15].angular = -0.6
        self.cmd[16].linear = 0.4
        self.cmd[16].angular = -0.3
        self.cmd[17].linear = 0.4
        self.cmd[17].angular = 0.0
        self.cmd[18].linear = 0.4
        self.cmd[18].angular = 0.3
        self.cmd[19].linear = 0.4
        self.cmd[19].angular = 0.6
        self.cmd[20].linear = 0.4
        self.cmd[20].angular = 0.9
        self.cmd[21].linear = 0.6
        self.cmd[21].angular = -0.9
        self.cmd[22].linear = 0.6
        self.cmd[22].angular = -0.6
        self.cmd[23].linear = 0.6
        self.cmd[23].angular = -0.3
        self.cmd[24].linear = 0.6
        self.cmd[24].angular = 0.0
        self.cmd[25].linear = 0.6
        self.cmd[25].angular = 0.3
        self.cmd[26].linear = 0.6
        self.cmd[26].angular = 0.6
        self.cmd[27].linear = 0.6
        self.cmd[27].angular = 0.9

