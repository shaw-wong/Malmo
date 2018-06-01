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
import time
import Tkinter as tk
from PIL import Image
import torchvision.transforms as T
import torch
from torch.autograd import Variable
import numpy as np
import dqn_pix

dqn = dqn_pix.DQN()

class TabQAgent:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if False:  # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["move 1", "turn 1", "attack 1", "move 0"]

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
        action = a[0][0]


        if len(self.state_Record) > 0:
            Last_State = self.state_Record[-1]
            Last_Action = self.action_Record[-1]
            dqn.record_transition(Last_State,Last_Action,torch.FloatTensor([current_r]),s)

        if dqn.memoryCounter > dqn.memory_size:
            dqn.learn()
        # try to send the selected action, only update prev_s if this succeeds
        try:

            agent_host.sendCommand(self.actions[action])

            self.state_Record.append(s)
            self.action_Record.append(a)


        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

    def run(self, agent_host, dqn):
        """run the agent on the world"""

        total_reward = 0

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



        return total_reward



sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

agent = TabQAgent()
agent_host = MalmoPython.AgentHost()
#




# -- set up the mission -- #
# all_file = ['./simpleMap.xml','./simpleMap_2.xml','./simpleMap_3.xml']
all_file = ['./chasePig.xml']





max_retries = 3

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 3000
cumulative_rewards = []


for i in range(num_repeats):
    xiaoyushishabi = random.randint(0,1)

    if xiaoyushishabi == 0:
        pass


    mission_file = random.choice(all_file)
    with open(mission_file, 'r') as f:
        print "Loading mission from %s" % mission_file
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission.requestVideo(600, 400)


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
    if cumulative_reward<0:resultreward = -100
    else:resultreward = 100
    cumulative_rewards += [cumulative_reward]


    # -- clean up -- #
    time.sleep(0.5)  # (let the Mod reset)

print "Done."

print "Cumulative rewards for all %d runs:" % num_repeats

torch.save(dqn.evalueNet,'Netmodel.pkl')
torch.save(dqn.targetNet,'Target.pkl')

miaowacao = 1
point = 0
for result in cumulative_rewards:
    csvfile = file('analysis.csv', 'a')
    writer = csv.writer(csvfile)
    writer.writerow([result])


# for result in cumulative_rewards:
#     if miaowacao % 100 == 0:
#         point += result
#         point = (point/100)
#         csvfile = file('analysis.csv', 'a')
#         writer = csv.writer(csvfile)
#         writer.writerow([point])
#         point = 0
#     else:
#         point += result
#     miaowacao = miaowacao + 1



