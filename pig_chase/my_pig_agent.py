# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

# ===================================================================================================================
# COMP90055 2018 SM1 Computing Project
# Team
# Xiaoyu Wang
# ===================================================================================================================

#TODO: self.action
from __future__ import division

import sys
import logging
import numpy as np
from common import ENV_TARGET_NAMES, ENV_ACTIONS
from six.moves import range
import random
from malmopy.agent import AStarAgent
from malmopy.agent import BaseAgent
import ast

VALID_POSITIONS = [(2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
                   (2, 3), (4, 3), (6, 3),
                   (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
                   (2, 5), (4, 5), (6, 5),
                   (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (1, 4), (7, 4)]

EASY_HUNTING_POINT = [(2, 2), (4, 2), (6, 2), (2, 4), (4, 4),
                      (6, 4), (2, 6), (4, 6), (6, 6)]

UNCATCHABLE_POSITIONS = [(4, 2), (4, 4), (4, 6)]
SINGLE_CATCHABLE = [(1, 4), (2, 4), (6, 4), (7, 4)]

P_FOCUSE = 0.75
REWARD_EXIT = 5
REWARD_CATCH_PIG = 25

FILE_NAME = "q_table.txt"

class PigAgent(BaseAgent):

    def __init__(self, name, visualizer=None):
        super(PigAgent, self).__init__(name, len(ENV_ACTIONS), visualizer=visualizer)
        self.agent = MyPigAgent(name, ENV_TARGET_NAMES[0], visualizer=visualizer)

    def act(self, new_state, reward, done, is_training=False):
        return self.agent.act(new_state, reward, done, is_training)

    def save(self, out_dir):
        self.agent.save(out_dir)

    def load(self, out_dir):
        self.agent(out_dir)

    def inject_summaries(self, idx):
        self.agent.inject_summaries(idx)

class MyPigAgent(AStarAgent):
    """myPigAgent"""
    def __init__(self, name, target, visualizer=None):
        super(MyPigAgent, self).__init__(name, len(ENV_ACTIONS), visualizer=visualizer)
        self.actions = ["move 1", "turn -1", "turn 1"]
        self.target = str(target)

        self.logger = logging.getLogger(__name__)
        if False:  # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.episode = 1
        self.reward_this_episode = 0
        self.reward_total_episode = 0

        self.q_table = {}
        self.load(FILE_NAME)
        self.epsilon = 0.01
        self.alpha = 0.1
        self.gamma = 1
        self.prev_a = None
        self.prev_s = None

    def act(self, state, reward, done, is_training=False):
        if done:
            self.episode += 1
            self.reward_this_episode += reward
            self.reward_total_episode += self.reward_this_episode
            print("reward for this episode:", self.reward_this_episode)
            self.save_reward_per_episode()
            self.reward_this_episode = 0
            self.prev_a = None
            self.prev_s = None
            if self.prev_a is not None and self.prev_s is not None:
                self.updateQTableFromTerminatingState(reward)
            self.save(FILE_NAME)
        else:
            self.reward_this_episode += reward
        # No observations, return random move
        # if state is None or type(state) is np.ndarray:
        if state is None:
            self.logger.info("None State")
            a = np.random.randint(0, len(self.actions))
            self.logger.info("Taking a random action: %s" % self.actions[a])
            return a

        entities = state[1]
        state = state[0]

        me = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self.name in k][0]
        target = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self.target in k][0]
        you = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if 'Agent_1' in k][0]
        if len(me) == 0 or len(target) == 0 or len(you) == 0:
            self.logger.info("empty observation")
            return np.random.randint(0, len(self.actions))

        # 0 for uncatchable, 1 for single-catchable, 2 for multi-catchable
        if target in UNCATCHABLE_POSITIONS:
            pig_state = 0
        elif target in SINGLE_CATCHABLE:
            pig_state = 1
        else:
            pig_state = 2

        me_details = [e for e in entities if e['name'] == self.name][0]
        me_yaw = int(me_details['yaw'])
        # convert Minecraft yaw to 0=north, 1=east, 2=south, 3=west
        me_direction = ((((me_yaw - 45) % 360) // 90) - 1) % 4

        you_details = [e for e in entities if e['name'] == 'Agent_1'][0]
        you_yaw = int(you_details['yaw'])
        you_direction = ((((you_yaw - 45) % 360) // 90) - 1) % 4
        #TODO: Random Belief
        me_to_target = 9999
        neighbors_of_target = self.neighbors(target)
        for n in neighbors_of_target:
            heuristic = self.get_heuristic(me, n, me_direction)
            if heuristic < me_to_target:
                me_to_target = heuristic

        you_to_target = 9999
        for n in neighbors_of_target:
            heuristic = self.get_heuristic(you, n, you_direction)
            if heuristic < you_to_target:
                you_to_target = heuristic

        me_to_exit_a = self.get_heuristic(me, (1,4), me_direction)
        me_to_exit_b = self.get_heuristic(me, (7,4), me_direction)
        if me_to_exit_a <= me_to_exit_b:
            me_to_exit = me_to_exit_a
        else:
            me_to_exit = me_to_exit_b

        current_s = (pig_state, me_to_target, you_to_target, me_to_exit)

        # if not self.q_table.has_key(current_s):
        if current_s not in self.q_table:
            self.q_table[current_s] = ([0] * len(self.actions))

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTable(reward, current_s)

        # select the next action
        rnd = random.random()
        if rnd < self.epsilon:
            a = random.randint(0, len(self.actions) - 1)
            self.logger.info("Random action: %s" % self.actions[a])
        else:
            m = max(self.q_table[current_s])
            self.logger.debug("Current values: %s" % ",".join(str(x) for x in self.q_table[current_s]))
            l = list()
            for x in range(0, len(self.actions)):
                if self.q_table[current_s][x] == m:
                    l.append(x)
            y = random.randint(0, len(l) - 1)
            a = l[y]
            self.logger.info("Taking q action: %s" % self.actions[a])


        # try to send the selected action, only update prev_s if this succeeds
        try:
            self.prev_s = current_s
            self.prev_a = a
            self.prev_my_s = me
            return a

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

        a = np.random.randint(0, len(self.actions))
        self.logger.info("Taking a random action: %s" % self.actions[a])
        return a

    def updateQTable(self, reward, current_state):

        old_q = self.q_table[self.prev_s][self.prev_a]
        new_q = old_q + self.alpha * (reward + self.gamma * max(self.q_table[current_state]) - old_q)

        self.q_table[self.prev_s][self.prev_a] = new_q
        # self.save(FILE_NAME)

    def updateQTableFromTerminatingState(self, reward):

        old_q = self.q_table[self.prev_s][self.prev_a]
        new_q = old_q + self.alpha * (reward + self.gamma * old_q)

        self.q_table[self.prev_s][self.prev_a] = new_q
        # self.save(FILE_NAME)

    def is_valid_position(self, pos):
            if pos not in VALID_POSITIONS:
                return False
            return True

    def neighbors(self, pos, **kwargs):
        neighbors = []
        x = pos[0]
        y = pos[1]
        possible_neighbors = [(x+1, y),(x-1, y),(x, y+1), (x, y-1)]
        for neighbor in possible_neighbors:
            if self.is_valid_position(neighbor):
                neighbors.append(neighbor)
        return neighbors

    # return the real heuristic value of a and b
    def get_heuristic(self, a, b, a_direction):
        if self.matches(a, b):
            return 0
        heuristic = self.heuristic(a, b)
        # if two agents are in the same column or row
        # same column
        if a[0] == b[0]:
            # in the line without pillars
            if a[0] % 2 == 0:
                if a_direction % 2 == 1:
                    return heuristic + 1
                elif (a[1] > b[1] and a_direction == 0) or (a[1] < b[1] and a_direction == 2):
                    return heuristic
                else:
                    return heuristic + 2
            # in the line with pillars
            else:
                if a_direction % 2 == 0:
                    return heuristic + 5
                else:
                    return heuristic + 4

        # same row
        elif a[1] == b[1]:
            # in the line without pillars
            if a[1] % 2 == 0:
                if a_direction % 2 == 0:
                    return heuristic + 1
                elif (a[0] > b[0] and a_direction == 3) or (a[0] < b[0] and a_direction == 1):
                    return heuristic
                else:
                    return heuristic + 2

            # in the line with pillars
            else:
                if a_direction % 2 == 0:
                    return heuristic + 4
                else:
                    return heuristic + 5

        # both a and b are at nine easy hunting points
        elif a in EASY_HUNTING_POINT and b in EASY_HUNTING_POINT:
            if a[0] > b[0] and a[1] > b[1]:
                if a_direction == 0 or a_direction == 3:
                    return heuristic + 1
                else:
                    return heuristic + 2

            if a[0] < b[0] and a[1] > b[1]:
                if a_direction == 0 or a_direction == 1:
                    return heuristic + 1
                else:
                    return heuristic + 2

            if a[0] > b[0] and a[1] < b[1]:
                if a_direction == 0 or a_direction == 1:
                    return heuristic + 2
                else:
                    return heuristic + 1

            else:
                if a_direction == 0 or a_direction == 3:
                    return heuristic + 2
                else:
                    return heuristic + 1

        # a or b is not at nine easy hunting points
        elif (a in EASY_HUNTING_POINT and b not in EASY_HUNTING_POINT and b[0] % 2 == 1) or (
                    a not in EASY_HUNTING_POINT and b in EASY_HUNTING_POINT and a[0] % 2 == 0) or (
                        a not in EASY_HUNTING_POINT and b in EASY_HUNTING_POINT and a[0] % 2 == 0 and b[0] % 2 == 1):
            if a_direction % 2 == 1:
                return heuristic + 2
            elif (a[1] > b[1] and a_direction == 0) or (a[1] < b[1] and a_direction == 2):
                return heuristic + 1
            else:
                return heuristic + 3

        elif (a in EASY_HUNTING_POINT and b not in EASY_HUNTING_POINT and b[0] % 2 == 0) or (
                    a not in EASY_HUNTING_POINT and b in EASY_HUNTING_POINT and a[0] % 2 == 1) or (
                        a not in EASY_HUNTING_POINT and b in EASY_HUNTING_POINT and a[0] % 2 == 1 and b[0] % 2 == 0):
            if a_direction % 2 == 0:
                return heuristic + 2
            elif (a[0] > b[0] and a_direction == 3) or (a[0] < b[0] and a_direction == 1):
                return heuristic + 1
            else:
                return heuristic + 3

        # a and b are not at nine easy hunting points
        elif a not in EASY_HUNTING_POINT and b not in EASY_HUNTING_POINT:
            if a[0] % 2 == 0:
                if a_direction % 2 == 1:
                    return heuristic + 2
                elif (a[1] > b[1] and a_direction == 0) or (a[1] < b[1] and a_direction == 2):
                    return heuristic + 1
                else:
                    return heuristic + 3
            else:
                if a_direction % 2 == 0:
                    return heuristic + 2
                elif (a[0] > b[0] and a_direction == 3) or (a[0] < b[0] and a_direction == 1):
                    return heuristic + 1
                else:
                    return heuristic + 3

    def heuristic(self, a, b, **kwargs):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def matches(self, a, b):
        # only compare position, ignore the direction
        return a == b

    def load(self, file_name):
        with open(file_name) as f:
            self.q_table = ast.literal_eval(f.read())

    def save(self, file_name):
        with open(file_name, "wb") as f:
            f.write(str(self.q_table))

    def save_reward_per_episode(self):
        with open("result.txt",'a') as f:
            f.write(str(self.reward_this_episode))
            f.write('\n')
