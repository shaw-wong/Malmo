import MalmoPython
import json
import logging
import sys
import time


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

    def act(self, world_state, agent_host, current_r):

        msg = world_state.observations[-1].text
        observations = json.loads(msg)
        entities = observations.get(u'entities', 0)
        target = [(item[u'x'], item[u'z']) for item in entities if item[u'name'] == u'Pig']
        self.target = target[0]

        heuristic = 9999
        nextMove = None
        for n in self.neighbors(world_state):
            nextPosition = (n[0], n[1])
            cost = self.heuristic(nextPosition, self.target)
            if heuristic >= cost:
               heuristic = cost
               nextMove = n[2]

        try:
            agent_host.sendCommand(nextMove)
            print(nextMove)
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
            elif action == "moveeast 1":
                neighbors.append((curr_x-1, curr_z, action))
            elif action == "movesouth 1":
                neighbors.append((curr_x, curr_z+1, action))
            elif action == "movewest 1":
                neighbors.append((curr_x-1, curr_z, action))

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

agent = AStarAgent()
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
