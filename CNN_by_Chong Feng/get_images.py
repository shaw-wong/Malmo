from __future__ import print_function

# ----------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import MalmoPython
import json
import logging
import os
import random
import sys
import time
from PIL import Image
from builtins import object
from future import standard_library

standard_library.install_aliases()
IMAGE_NUMBER = 100
STEP = 5
MISSION = "maze"
PATH = "data/" + MISSION + "/"


class ExploreAgent(object):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if False:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["move 1", "move -1", "turn 1", "strafe 1", "strafe -1"]

    def act(self, world_state, agent_host):
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)
        self.logger.debug(obs)
        if u"XPos" not in obs or u"ZPos" not in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_s = "%d:%d" % (int(obs[u"XPos"]), int(obs[u"ZPos"]))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (
                          current_s, float(obs[u"XPos"]), float(obs[u"ZPos"])))

        a = random.randint(0, len(self.actions) - 1)
        self.logger.info("Random action: %s" % self.actions[a])

        try:
            agent_host.sendCommand(self.actions[a])
        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

    def run(self, agent_host):
        is_first_action = True
        step = 0
        image_number = 0
        world_state = agent_host.getWorldState()

        while image_number < IMAGE_NUMBER and world_state.is_mission_running:
            if is_first_action:
                while True:
                    time.sleep(1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    if world_state.is_mission_running and \
                       len(world_state.observations) > 0 and \
                       not world_state.observations[-1].text == "{}":

                            frame = world_state.video_frames[-1]
                            if image_number < IMAGE_NUMBER and \
                               step % STEP == 0:
                                    image = Image.frombytes(
                                        "RGB", (frame.width, frame.height),
                                        bytes(frame.pixels))
                                    image.save(PATH + str(image_number) +
                                               ".png")
                                    image_number += 1
                                    self.logger.info("Image number is %d" %
                                                     image_number)

                            self.act(world_state, agent_host)
                            step += 1
                            break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False

            else:
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    if world_state.is_mission_running and \
                       len(world_state.observations) > 0 and \
                       not world_state.observations[-1].text == "{}":
                            self.act(world_state, agent_host)
                            step += 1

                            frame = world_state.video_frames[-1]
                            if image_number < IMAGE_NUMBER and \
                               step % STEP == 0:
                                    image = Image.frombytes(
                                        "RGB", (frame.width, frame.height),
                                        bytes(frame.pixels))
                                    image.save(PATH + str(image_number) +
                                               ".png")
                                    image_number += 1
                                    self.logger.info("Image number is %d" %
                                                     image_number)

                    break
                    if not world_state.is_mission_running:
                        break


if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 0)
else:
    import functools
    print = functools.partial(print, flush=True)
agent = ExploreAgent()
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print("ERROR:", e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)
mission_file = MISSION + ".xml"
with open(mission_file, "r") as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
print()
my_mission_record = MalmoPython.MissionRecordSpec()
try:
    agent_host.startMission(my_mission, my_mission_record)
except RuntimeError as e:
    print("Error starting mission:", e)
    exit(1)
print("Waiting for the mission to start", end=" ")
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:", error.text)
print()
agent.run(agent_host)
print("Done.")
print()
