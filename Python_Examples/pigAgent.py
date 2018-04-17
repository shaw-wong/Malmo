
import MalmoPython
import json
import logging
import os
import random
import sys
import time
import Tkinter as tk

# class AgentFactory:
#     def __init__(self):
#         self.numberOfAgent = 0
#         self.agent1 = None
#         self.agent2 = None
#
#     def chooseAgent(self):
#         if self.numberOfAgent == 0:
#             self.agent1 = AStarAgent()
#             self.numberOfAgent += 1
#
#         if self.numberOfAgent == 1:
#             self.agent2 = AStarAgent
#             self.numberOfAgent += 1
#
#     def creatAgents(self):
#         while self.numberOfAgent < 2:
#             self.chooseAgent()


class AStarAgent:
    def __init__(self):

        self.target = None
        self.logger = logging.getLogger(__name__)
        if False:  # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
        self.previous_s = None
        self.previous_a = None

    def act(self, world_state, agent_host, current_r):

        msg = world_state.observations[-1].text
        observations = json.loads(msg)
        current_s = (observations[u'XPos'], observations[u'ZPos'])

        entities = observations.get(u'entities', 0)
        target = [(item[u'x'], item[u'z']) for item in entities if item[u'name'] == u'Pig']
        self.target = target[0]

        heuristic = 9999
        nextMove = "movenorth 1"
        for n in self.neighbors(world_state):
            nextPosition = (n[0], n[1])
            cost = self.heuristic(nextPosition, self.target)
            if not (current_s == self.previous_s and n[2] == self.previous_a):
                if heuristic >= cost:
                       heuristic = cost
                       nextMove = n[2]

        try:
            self.previous_s = current_s
            self.previous_a = nextMove
            agent_host.sendCommand(nextMove)
            self.logger.info("Taking q action: %s" % nextMove)

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)
        return current_r

    def neighbors(self, world_state):

        neighbors = []
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)

        curr_x = int(obs[u'XPos'])
        curr_z = int(obs[u'ZPos'])

        for action in self.actions:
            if action == "movenorth 1":
                neighbors.append((curr_x, curr_z-1, action))
            elif action == "movewest 1":
                neighbors.append((curr_x-1, curr_z, action))
            elif action == "movesouth 1":
                neighbors.append((curr_x, curr_z+1, action))
            elif action == "moveeast 1":
                neighbors.append((curr_x+1, curr_z, action))
        # TODO
        # filter the invalid neighbors
        return neighbors

    def heuristic(self, a, b, state=None):
        (x1, y1) = (a[0], a[1])
        (x2, y2) = (b[0], b[1])
        return abs(x1 - x2) + abs(y1 - y2)

    def run(self, agent_host):
        """run the agent on the world"""

        total_reward = 0

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
                    if world_state.is_mission_running and len(world_state.observations) > 0 and not \
                            world_state.observations[-1].text == "{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False

            else:
                # wait for non-zero reward
                # while world_state.is_mission_running and current_r == 0:
                #     time.sleep(0.1)
                #     world_state = agent_host.getWorldState()
                #     for error in world_state.errors:
                #         self.logger.error("Error: %s" % error.text)
                #     for reward in world_state.rewards:
                #         current_r += reward.getValue()
                #         print(current_r)
                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations) > 0 and not \
                            world_state.observations[-1].text == "{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break

        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        return total_reward

class TabQAgent():
    def __init__(self):
        self.epsilon = 0.01  # chance of taking a random action instead of the best
        self.alpha = 0.1
        self.gamma = 1

        self.logger = logging.getLogger(__name__)
        if False:  # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
        self.q_table = {}
        self.canvas = None
        self.root = None

    def updateQTable(self, reward, current_state):
        """Change q_table to reflect what we have learnt."""

        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]

        new_q = old_q + self.alpha * (reward + self.gamma * max(self.q_table[current_state]) - old_q)

        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q

    def updateQTableFromTerminatingState(self, reward):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""

        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]

        # TODO: what should the new action value be?
        new_q = old_q + self.alpha * (reward + self.gamma * old_q)

        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q

    def act(self, world_state, agent_host, current_r):
        """take 1 action in response to the current world state"""

        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)  # most recent observation
        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
        if not self.q_table.has_key(current_s):
            self.q_table[current_s] = ([0] * len(self.actions))

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTable(current_r, current_s)

        self.drawQ(curr_x=int(obs[u'XPos']), curr_y=int(obs[u'ZPos']))

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
            agent_host.sendCommand(self.actions[a])
            self.prev_s = current_s
            self.prev_a = a

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

        return current_r

    def run(self, agent_host):
        """run the agent on the world"""

        total_reward = 0

        self.prev_s = None
        self.prev_a = None

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
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations) > 0 and not \
                            world_state.observations[-1].text == "{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations) > 0 and not \
                            world_state.observations[-1].text == "{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break

        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState(current_r)

        self.drawQ()

        return total_reward

    def drawQ(self, curr_x=None, curr_y=None):
        scale = 40
        world_x = 9
        world_y = 7
        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            self.canvas = tk.Canvas(self.root, width=world_x * scale, height=world_y * scale, borderwidth=0,
                                    highlightthickness=0, bg="black")
            self.canvas.grid()
            self.root.update()
        self.canvas.delete("all")
        action_inset = 0.1
        action_radius = 0.1
        curr_radius = 0.2
        action_positions = [(0.5, action_inset), (0.5, 1 - action_inset), (action_inset, 0.5), (1 - action_inset, 0.5)]
        # (NSWE to match action order)
        min_value = -20
        max_value = 20
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d" % (x, y)
                self.canvas.create_rectangle(x * scale, y * scale, (x + 1) * scale, (y + 1) * scale, outline="#fff",
                                             fill="#000")
                for action in range(4):
                    if not s in self.q_table:
                        continue
                    value = self.q_table[s][action]
                    color = 255 * (value - min_value) / (max_value - min_value)  # map value to 0-255
                    color = max(min(color, 255), 0)  # ensure within [0,255]
                    color_string = '#%02x%02x%02x' % (255 - color, color, 0)
                    self.canvas.create_oval((x + action_positions[action][0] - action_radius) * scale,
                                            (y + action_positions[action][1] - action_radius) * scale,
                                            (x + action_positions[action][0] + action_radius) * scale,
                                            (y + action_positions[action][1] + action_radius) * scale,
                                            outline=color_string, fill=color_string)
        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval((curr_x + 0.5 - curr_radius) * scale,
                                    (curr_y + 0.5 - curr_radius) * scale,
                                    (curr_x + 0.5 + curr_radius) * scale,
                                    (curr_y + 0.5 + curr_radius) * scale,
                                    outline="#fff", fill="#fff")
        self.root.update()



# agent = AStarAgent()
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

# -- set up the mission -- #
mission_file = './chasePig.xml'
with open(mission_file, 'r') as f:
    print "Loading mission from %s" % mission_file
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)

max_retries = 3

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 150

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
    time.sleep(0.5)  # (let the Mod reset)

print "Done."

print
print "Cumulative rewards for all %d runs:" % num_repeats
print cumulative_rewards
