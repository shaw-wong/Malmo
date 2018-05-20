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
import csv
import DQN
import MalmoPython
import json
import logging
import os
import random
import sys
import time
import Tkinter as tk


dqn = DQN.DQN()


class TabQAgent:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if False:  # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]

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


    def act(self, world_state, agent_host, current_r, dqn):
        """take 1 action in response to the current world state"""

        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)  # most recent observation
        s = self.Translate_State(obs)
        a = dqn.choose_action(s)
        a = int(a)
        special_reward = self.Calculate_Reward_FromState(s)
        Q_learning_Reward = special_reward + current_r
        self.logger.info("Taking q action: %s" % self.actions[a])
        if len(self.state_Record) > 0:
            Last_State = self.state_Record[-1]
            Last_Action = self.action_Record[-1]
            dqn.record_transition(Last_State,Last_Action,Q_learning_Reward,s)

        if dqn.memoryCounter > 200:
            dqn.learn()
        # try to send the selected action, only update prev_s if this succeeds
        try:
            agent_host.sendCommand(self.actions[a])
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
        world_state = agent_host.getWorldState()



        while world_state.is_mission_running:


            current_r = 0

            if is_first_action:
                # wait until have received a valid observation
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()

                    for reward in world_state.rewards:
                        current_r += reward.getValue()

                    if world_state.is_mission_running and len(world_state.observations) > 0 and not \
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
                    if world_state.is_mission_running and len(world_state.observations) > 0 and not \
                            world_state.observations[-1].text == "{}":
                        total_reward += current_r
                        self.act(world_state, agent_host, current_r, dqn)
                        break
                    if not world_state.is_mission_running:
                        break

        if len(self.state_Record) != 0:
            Dead_State = self.state_Record[-1]
            Dead_State[-2] = Dead_State[-2] - 1
            dqn.record_transition(self.state_Record[-1],self.action_Record[-1],current_r,Dead_State)


        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r



        return total_reward



sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

agent = TabQAgent()
agent_host = MalmoPython.AgentHost()
#

# -- set up the mission -- #
mission_file = './simpleMap.xml'
with open(mission_file, 'r') as f:
    print "Loading mission from %s" % mission_file
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)


max_retries = 3

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 1000
cumulative_rewards = []


for i in range(num_repeats):

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
    cumulative_rewards += [cumulative_reward]


    # -- clean up -- #
    time.sleep(0.5)  # (let the Mod reset)

print "Done."

print "Cumulative rewards for all %d runs:" % num_repeats
print cumulative_rewards
add = 0
for result in cumulative_rewards:
    csvfile = file('analysis.csv', 'a')
    write = csv.write(csvfile)
    write.writerow([result])
