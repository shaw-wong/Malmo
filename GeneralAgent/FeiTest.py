# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #6: Discrete movement, rewards, and learning

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998
import torch
import DQN
import csv
import MalmoPython
import json
import logging
import os
import random
import sys
import Two_layer_agent_DQN
import time
import Tkinter as tk
from PIL import Image
import torchvision.transforms as T
import torch
from torch.autograd import Variable
import numpy as np
import dqn_pix
import math
from collections import namedtuple
dqn = Two_layer_agent_DQN.DQN()


class TabQAgent:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if False:  # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["move forwards", "move backwards", "strafe right", "strafe left", "turn right", "turn left", "attack", 'move to sheep']
        self.q_table = {} #FfFFFFffffffffffffffff

    def Translate_State(self,obs):
        s = []

        s.append(obs[u'XPos'])
        s.append(obs[u'ZPos'])
        if obs[u'IsAlive'] == True:
            s.append(0.0)
        else: s.append(1.0)
        s.append(1.0)

        return s

    def Calculate_Reward_FromState(self, state):
        special_reward = 0
        if state == self.state_Record:
            special_reward = -10


        return special_reward

    def Pix2State(self,world_state):
        resize = T.Compose([T.ToPILImage(),
                            T.Resize(40, interpolation=Image.CUBIC),
                            T.ToTensor()])
        frame = world_state.video_frames[-1]
        image = np.array(Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels)))
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image, dtype=np.float32) / 255
        image = torch.from_numpy(image)
        # Resize, and add a batch dimension (BCHW)
        image = resize(image).unsqueeze(0).type(torch.FloatTensor)
        return image

    def act(self, world_state, agent_host, current_r, dqn):
        """take 1 action in response to the current world state"""

        # print(len(world_state.video_frames))
        # screen = self.Pix2State(world_state)
        s = self.Pix2State(world_state)
        a = dqn.choose_action(s)
        # action = a[0][0]
        mission_learner = a[0][0]

        action, difference, reward, speed = self.choosesmaraction(mission_learner,world_state, agent_host, current_r)
        # current_r = current_r + reward


        # self.logger.info("Taking q action: %s" % self.actions[action])
        # if len(self.state_Record) > 0:
        #     current_r = current_r + reward
        #     Last_State = self.state_Record[-1]
        #     Last_Action = self.action_Record[-1]
        #     dqn.record_transition(Last_State,Last_Action,torch.FloatTensor([current_r]),s)
        #
        #
        # if dqn.memoryCounter > dqn.memory_size:
        #     dqn.learn()
        # try to send the selected action, only update prev_s if this succeeds
        try:
            # agent_host.sendCommand(self.actions[action])
            self.sendCommand(action, difference, speed)


            # self.state_Record.append(s)
            # self.action_Record.append(a)

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

    def sendCommand(self, action, difference, speed):
        if self.actions[action] == 'move forwards':
            agent_host.sendCommand("move 0.5")

        if self.actions[action] == 'move backwards':
            agent_host.sendCommand("move -0.5")

        if self.actions[action] == 'strafe right':
            agent_host.sendCommand("strafe 0.5")

        if self.actions[action] == 'strafe left':
            agent_host.sendCommand("strafe -0.5")

        if self.actions[action] == 'turn right':
            agent_host.sendCommand("turn 1")
            time.sleep(0.5)
            agent_host.sendCommand("turn 0")

        if self.actions[action] == 'turn left':
            agent_host.sendCommand("turn -1")
            time.sleep(0.5)
            agent_host.sendCommand("turn 0")

        if self.actions[action] == 'move to sheep':
            agent_host.sendCommand("turn " + str(difference))
            agent_host.sendCommand("move " + str(speed))

        if self.actions[action] == 'attack':
            agent_host.sendCommand("attack 1")
            agent_host.sendCommand("attack 0")

    def choosesmaraction(self,mission_learner,world_state, agent_host, current_r):
        action = reward = difference = speed = 0

        if mission_learner == 0:
            action, reward,difference, speed = self.hit_sheep_act(world_state)
        elif mission_learner == 1:
            action = self.mazeRun(world_state, current_r)

        return action, reward, difference, speed

    def mazeRun(self, world_state, current_r):
        self.epsilon = 0.01

        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)
        if obs.has_key(u'XPos'):
            you_yaw = obs[u'Yaw']
            you_direction = ((((you_yaw - 45) % 360) // 90) - 1) % 4
            current_s = "%d:%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']), you_direction)
        else: current_s = (0,0,0)

        if not self.q_table.has_key(current_s):
            self.q_table[current_s] = ([0] * len(self.actions))

        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTable(current_r, current_s)

        rnd = random.random()
        if rnd < self.epsilon:
            a = random.randint(0, len(self.actions) - 1) #ffffffffffffff
        else:
            m = max(self.q_table[current_s])
            actionList = list()
            for x in range(0, len(self.actions)): #fffffffffffff
                if self.q_table[current_s][x] == m:
                    actionList.append(x)
            y = random.randint(0, len(actionList) - 1) 
            a = actionList[y]

        self.prev_s = current_s
        self.prev_a = a

        return a

    def updateQTable(self, reward, current_state):

        learningRate = 0.5
        discountFactor = 1
        old_q = self.q_table[self.prev_s][self.prev_a]
        m = max(self.q_table[current_state])
        new_q = (1 - learningRate) * old_q + learningRate * (reward + discountFactor * m)
        self.q_table[self.prev_s][self.prev_a] = new_q

    def updateQTableFromTerminatingState(self, reward):

        learningRate = 0.5
        old_q = self.q_table[self.prev_s][self.prev_a]
        new_q = (1 - learningRate) * old_q + learningRate * reward
        self.q_table[self.prev_s][self.prev_a] = new_q

    def hit_sheep_act(self,world_state):
        EntityInfo = namedtuple('EntityInfo', 'x, y, z, yaw, pitch, name, colour, variation, quantity, life')
        EntityInfo.__new__.__defaults__ = (0, 0, 0, 0, 0, "", "", "", 1, "")
        current_r = 0
        action = random.randint(0,7)
        difference = 0
        speed = 0
        current_yaw = 0
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            ob = json.loads(msg)
            self_x = 0
            self_z = 0
            # Use the line-of-sight observation to determine when to hit and when not to hit:
            if u'LineOfSight' in ob:
                current_r = current_r + 6
                los = ob[u'LineOfSight']
                type = los["type"]
                if type == "Sheep":
                    # agent_host.sendCommand("attack 1")
                    # agent_host.sendCommand("attack 0")
                    action = 6
                    current_r += 10
                    return action, difference, current_r, speed
            # Get our position/orientation:
            if u'Yaw' in ob:
                current_yaw = ob[u'Yaw']
            if u'XPos' in ob:
                self_x = ob[u'XPos']
            if u'ZPos' in ob:
                self_z = ob[u'ZPos']
            # Use the nearby-entities observation to decide which way to move, and to keep track
            # of population sizes - allows us some measure of "progress".
            if u'entities' in ob:
                entities = [EntityInfo(**k) for k in ob["entities"]]
                x_pull = 0
                z_pull = 0
                for e in entities:
                    if e.name == "Sheep":
                        # Each sheep contributes to the direction we should head in...
                        dist = max(0.0001, (e.x - self_x) * (e.x - self_x) + (e.z - self_z) * (e.z - self_z))
                        # Prioritise going after wounded sheep. Max sheep health is 8, according to Minecraft wiki...
                        weight = 9.0 - e.life
                        x_pull += weight * (e.x - self_x) / dist
                        z_pull += weight * (e.z - self_z) / dist
                # Determine the direction we need to turn in order to head towards the "sheepiest" point:
                yaw = -180 * math.atan2(x_pull, z_pull) / math.pi
                difference = yaw - current_yaw;
                if difference != 0:
                    while difference < -180:
                        difference += 360;
                    while difference > 180:
                        difference -= 360;
                    difference /= 180.0;
                    # agent_host.sendCommand("turn " + str(difference))
                    action = 7


                speed = 1.0 if abs(difference) < 0.5 else 0  # move slower when turning faster - helps with "orbiting" problem
                # agent_host.sendCommand("move " + str(move_speed))

        if world_state.number_of_rewards_since_last_state > 0:
                    # Keep track of our total reward:
            current_r += world_state.rewards[-1].getValue()
        return action, difference, current_r, speed

    def run(self, agent_host, dqn):
        """run the agent on the world"""


        total_reward = 0

        self.prev_s = None
        self.prev_a = None


        self.state_Record = []
        self.action_Record = []

        is_first_action = True

        # main loop:
        world_state = agent_host.peekWorldState()




        while world_state.is_mission_running:
            current_r = 0

            if is_first_action:
                # wait until have received a valid observation
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()

                    for reward in world_state.rewards:
                        current_r += reward.getValue()

                    if world_state.is_mission_running and len(world_state.observations) > 0 and len(world_state.video_frames) > 0 and not \
                            world_state.observations[-1].text == "{}":
                        total_reward += current_r

                        self.act(world_state, agent_host, current_r, dqn)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False

            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()

                    for reward in world_state.rewards:
                        current_r += reward.getValue()

                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()


                    for reward in world_state.rewards:

                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations) > 0 and len(world_state.video_frames) > 0 and not \
                            world_state.observations[-1].text == "{}":
                        total_reward += current_r

                        self.act(world_state, agent_host, current_r, dqn)
                        break
                    if not world_state.is_mission_running:
                        break

        if len(self.state_Record) != 0:
            # print(current_r)
            Dead_State = self.state_Record[-1]
            dqn.record_transition(self.state_Record[-1],self.action_Record[-1],torch.FloatTensor([current_r]),Dead_State)


        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState(current_r)

        return total_reward



sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

agent = TabQAgent()
agent_host = MalmoPython.AgentHost()
#




# -- set up the mission -- #

all_file = ['./chasePig.xml', './FeiEight.xml']




max_retries = 3

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 100
cumulative_rewards = []


for i in range(num_repeats):

    mission_file = random.choice(all_file)
    with open(mission_file, 'r') as f:
        print "Loading mission from %s" % mission_file
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission.requestVideo(600, 400)
    #

    print 'Repeat %d of %d' % (i + 1, num_repeats)

    my_mission_record = MalmoPython.MissionRecordSpec()
    agent_host.startMission(my_mission, my_mission_record)


    print "Waiting for the mission to start",
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print "Error:", error.text

    # -- run the agent in the world -- #
    cumulative_reward = agent.run(agent_host,dqn)
    print 'Cumulative reward: %d' % cumulative_reward

    # torch.save(dqn.evalueNet, 'Netmodel.pkl')
    # torch.save(dqn.targetNet, 'Target.pkl')
    # csvfile = file('analysis.csv', 'a')
    # writer = csv.writer(csvfile)
    # writer.writerow([cumulative_reward])
    cumulative_rewards += [cumulative_reward]


    # -- clean up -- #
    time.sleep(0.5)  # (let the Mod reset)

print "Done."

print "Cumulative rewards for all %d runs:" % num_repeats



miaowacao = 1
point = 0
for result in cumulative_rewards:
    if result >-30:point = point +1
print(float(point)/float(100))






