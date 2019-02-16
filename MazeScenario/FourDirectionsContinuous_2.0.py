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
#
# Project Name: MazeScenario_2.2 -- Four directions in continuous movement model 
# with state (X, Z, Yaw) and six actions
# Name: Fei TENG
# Algorithm: Q-learning

import MalmoPython
import json
import logging
import os
import random
import sys
import time



class TabQAgent:


  def __init__(self):
    self.epsilon = 0.01
    self.logger = logging.getLogger(__name__)
    if False:
      self.logger.setLevel(logging.DEBUG)
    else:
      self.logger.setLevel(logging.INFO)
    self.logger.handlers = []
    self.logger.addHandler(logging.StreamHandler(sys.stdout))
    self.actions = ["move forwards", "move backwards", "strafe right", "strafe left", "turn right", "turn left"]
    self.q_table = {}
    self.canvas = None
    self.root = None
    self.Zpos = 0


  def updateQTable(self, reward, current_state):

    learningRate = 0.7
    discountFactor = 1
    old_q = self.q_table[self.prev_s][self.prev_a]
    m = max(self.q_table[current_state])
    new_q = (1 - learningRate) * old_q + learningRate * (reward + discountFactor * m)
    self.q_table[self.prev_s][self.prev_a] = new_q

  def updateQTableFromTerminatingState(self, reward):

    learningRate = 0.7
    old_q = self.q_table[self.prev_s][self.prev_a]
    new_q = (1 - learningRate) * old_q + learningRate * reward
    self.q_table[self.prev_s][self.prev_a] = new_q


  def act(self, world_state, agent_host, current_r):

    obs_text = world_state.observations[-1].text
    obs = json.loads(obs_text)
    self.logger.debug(obs)

    if not u'XPos' in obs or not u'ZPos' in obs:
      self.logger.error("Incomplete observation received: %s" % obs_text)
      return 0
  # set state in Q-table with coordinate and the direction that agents face (X, Z, Yaw)
    you_yaw = obs[u'Yaw']
    you_direction = ((((you_yaw - 45) % 360) // 90) - 1) % 4

    current_s = "%d:%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']), you_direction)

    self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
    if not self.q_table.has_key(current_s):
      self.q_table[current_s] = ([0] * len(self.actions))

    if self.prev_s is not None and self.prev_a is not None:
      self.updateQTable(current_r, current_s)

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

    try:

      if self.actions[a] == 'move forwards':
        agent_host.sendCommand("move 1.2")

      if self.actions[a] == 'move backwards':
        agent_host.sendCommand("move -1.2")

      if self.actions[a] == 'strafe right':
        agent_host.sendCommand("strafe 1,2")

      if self.actions[a] == 'strafe left':
          agent_host.sendCommand("strafe -1.2")

      if self.actions[a] == 'turn right':
        agent_host.sendCommand("turn 1")
        time.sleep(0.5)
        agent_host.sendCommand("turn 0")

      if self.actions[a] == 'turn left':
        agent_host.sendCommand("turn -1")
        time.sleep(0.5)
        agent_host.sendCommand("turn 0")

      self.prev_s = current_s
      self.prev_a = a

    except RuntimeError as e:
      self.logger.error("Failed to send command: %s" % e)
    return current_r


  def run(self, agent_host):

    total_reward = 0
    self.prev_s = None
    self.prev_a = None

    is_first_action = True

    world_state = agent_host.getWorldState()

    while world_state.is_mission_running:

      current_r = 0

      if is_first_action:
        # wait until have received a valid observation
        while True:
          time.sleep(0.1)
          world_state = agent_host.getWorldState()

          for error in world_state.errors:
            self.logger.error("Error: %s" % error.text)
          for reward in world_state.rewards:
            current_r += reward.getValue()
          if world_state.is_mission_running and len(world_state.observations) > 0 and not world_state.observations[
            -1].text == "{}":
            total_reward += self.act(world_state, agent_host, current_r)
            break
          if not world_state.is_mission_running:
            break
        is_first_action = False
      else:
        while world_state.is_mission_running and current_r == 0:
          time.sleep(0.1)
          world_state = agent_host.getWorldState()
          for error in world_state.errors:
            self.logger.error("Error: %s" % error.text)
          for reward in world_state.rewards:
            current_r += reward.getValue()

        while True:
          time.sleep(0.1)
          world_state = agent_host.getWorldState()

          for error in world_state.errors:
            self.logger.error("Error: %s" % error.text)
          for reward in world_state.rewards:
            current_r += reward.getValue()
          if world_state.is_mission_running and len(world_state.observations) > 0 and not world_state.observations[
            -1].text == "{}":
            total_reward += self.act(world_state, agent_host, current_r)
            break
          if not world_state.is_mission_running:
            break


    self.logger.debug("Final reward: %d" % current_r)
    total_reward += current_r

    # update Q values
    if self.prev_s is not None and self.prev_a is not None:
      self.updateQTableFromTerminatingState(current_r)


    return total_reward

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

agent = TabQAgent()
agent_host = MalmoPython.AgentHost()
try:
  agent_host.parse(sys.argv)
except RuntimeError as e:
  print 'ERROR:', e
  print agent_host.getUsage()
  exit(1)
if agent_host.receivedArgument("help"):
  print agent_host.getUsage()
  exit(0)
agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)

# -- set up the mission -- #
mission_file = './FeiEight.xml'
with open(mission_file, 'r') as f:
  print "Loading mission from %s" % mission_file
  mission_xml = f.read()
  my_mission = MalmoPython.MissionSpec(mission_xml, True)

max_retries = 3

if agent_host.receivedArgument("test"):
  num_repeats = 1
else:
  num_repeats = 500

cumulative_rewards = []
for i in range(num_repeats):

  print
  print 'Repeat %d of %d' % (i + 1, num_repeats)

  my_mission_record = MalmoPython.MissionRecordSpec()

  for retry in range(max_retries):
    try:
      agent_host.startMission(my_mission, my_mission_record)
      break
    except RuntimeError as e:
      if retry == max_retries - 1:
        print "Error starting mission:", e
        exit(1)
      else:
        time.sleep(2.5)

  print "Waiting for the mission to start",
  world_state = agent_host.getWorldState()
  while not world_state.has_mission_begun:
    sys.stdout.write(".")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
      print "Error:", error.text
  print

  # -- run the agent in the world -- #
  cumulative_reward = agent.run(agent_host)
  print 'Cumulative reward: %d' % cumulative_reward
  cumulative_rewards += [cumulative_reward]

  # -- clean up -- #
  time.sleep(0.5)

print "Done."

print
print "Cumulative rewards for all %d runs:" % num_repeats
print cumulative_rewards



