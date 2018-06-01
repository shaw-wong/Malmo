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
import glob
import numpy as np
import tensorflow as tf

standard_library.install_aliases()
IMAGE_NUMBER = 11
STEP = 1
MISSION = "maze"
PATH = "images/"


class ExploreAgent(object):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if False:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["turn 1"]

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


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias(length):
    return tf.Variable(tf.constant(0.1, shape=[length]))


def layer(input, num_input_channels, filter_size, num_filters, training,
          use_bn=True, use_relu=True, use_pool=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = weight(shape)
    biases = bias(num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1],
                         padding="SAME")
    layer += biases

    if use_bn:
        layer = tf.layers.batch_normalization(layer, training=training)

    if use_relu:
        layer = tf.nn.relu(layer)

    if use_pool:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding="SAME")

    return layer


def flat_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = weight([num_inputs, num_outputs])
    biases = bias(num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


filter_size1 = 3
num_filters1 = 32
filter_size2 = 3
num_filters2 = 64
filter_size3 = 3
num_filters3 = 128
filter_size4 = 3
num_filters4 = 256
fc_size = 128
num_channels = 3
img_size = 128
img_size_flat = img_size * img_size * num_channels
img_shape = (img_size, img_size)
classes = ['maze', 'pig']
num_classes = len(classes)

test_x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='test_x')
test_x_image = tf.reshape(test_x, [-1, img_size, img_size, num_channels])
test_y_true = tf.placeholder(tf.float32, shape=[None, num_classes],
                             name='test_y_true')
test_y_true_cls = tf.argmax(test_y_true, dimension=1)

test_layer1 = layer(input=test_x_image, num_input_channels=num_channels,
                    filter_size=filter_size1, num_filters=num_filters1,
                    training=False)
test_layer2 = layer(input=test_layer1, num_input_channels=num_filters1,
                    filter_size=filter_size2, num_filters=num_filters2,
                    training=False)
test_layer3 = layer(input=test_layer2, num_input_channels=num_filters2,
                    filter_size=filter_size3, num_filters=num_filters3,
                    training=False)
# test_layer4 = layer(input=test_layer3, num_input_channels=num_filters3,
#                     filter_size=filter_size4, num_filters=num_filters4,
#                     training=False)
test_layer_flat, test_num_features = flat_layer(test_layer3)
test_layer_fc1 = fc_layer(input=test_layer_flat, num_inputs=test_num_features,
                          num_outputs=fc_size, use_relu=True)
test_layer_fc2 = fc_layer(input=test_layer_fc1, num_inputs=fc_size,
                          num_outputs=num_classes, use_relu=False)

test_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=test_layer_fc2, labels=test_y_true)
test_y_pred_cls = tf.argmax(test_layer_fc2, dimension=1)

test_path = "images/*.png"

test_set = []
test_files = glob.glob(test_path)
for file in test_files:
    img = Image.open(file)
    test_set.append((np.array(img)))
    img.close()

sess = tf.Session()
saver = tf.train.Saver()
if tf.train.latest_checkpoint('ckpts') is not None:
    saver.restore(sess, tf.train.latest_checkpoint('ckpts'))
else:
    assert 'can not find checkpoint folder path!'

test = np.array(test_set).reshape(len(test_set), img_size_flat)

results = []
for i in range(len(test)):
    results.append(test_y_pred_cls.eval(
        session=sess, feed_dict={test_x: test[i].reshape(1, img_size_flat)}))
sess.close()

maze = 0
pig = 0
for result in results:
    if result == 0:
        maze += 1
    else:
        pig += 1
print("%s votes for maze!" % str(maze))
print("%s votes for pig!" % str(pig))
if maze > pig:
    print("So, this is a maze game!")
else:
    print("So, this is a pig game!")
