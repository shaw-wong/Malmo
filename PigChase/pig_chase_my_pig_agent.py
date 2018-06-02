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

import numpy as np
import os
import sys

from argparse import ArgumentParser
from datetime import datetime

import six
from os import path
from threading import Thread, active_count
from time import sleep

from malmopy.agent import RandomAgent
try:
    from malmopy.visualization.tensorboard import TensorboardVisualizer
    from malmopy.visualization.tensorboard.cntk import CntkConverter
except ImportError:
    print('Cannot import tensorboard, using ConsoleVisualizer.')
    from malmopy.visualization import ConsoleVisualizer

from common import parse_clients_args, visualize_training, ENV_AGENT_NAMES, ENV_TARGET_NAMES
from agent import PigChaseChallengeAgent, FocusedAgent, TabularQLearnerAgent, get_agent_type
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder
from my_pig_agent import MyPigAgent

# Enforce path
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))

BASELINES_FOLDER = 'results/myPig/pig_chase/%s'
EPOCH_SIZE = 100


def agent_factory(name, role, clients, max_epochs,
                  logdir, visualizer):

    assert len(clients) >= 2, 'Not enough clients (need at least 2)'
    clients = parse_clients_args(clients)

    builder = PigChaseSymbolicStateBuilder()
    env = PigChaseEnvironment(clients, builder, role=role,
                              randomize_positions=True)

    if role == 0:
        agent = PigChaseChallengeAgent(name)
        obs = env.reset(get_agent_type(agent))

        reward = 0
        agent_done = False

        while True:
            if env.done:
                while True:
                    obs = env.reset(get_agent_type(agent))
                    if obs:
                        break

            # select an action
            action = agent.act(obs, reward, agent_done, is_training=True)
            # reset if needed
            if env.done:
                obs = env.reset(get_agent_type(agent))

            # take a step
            obs, reward, agent_done = env.do(action)

    else:

        agent = MyPigAgent(name, ENV_AGENT_NAMES[0], visualizer)
        obs = env.reset()
        reward = 0
        agent_done = False
        viz_rewards = []

        max_training_steps = EPOCH_SIZE * max_epochs
        for step in six.moves.range(1, max_training_steps+1):

            # check if env needs reset
            if env.done:
                while True:
                    if len(viz_rewards) == 0:
                        viz_rewards.append(0)
                    visualize_training(visualizer, step, viz_rewards)
                    tag = "Episode End Conditions"
                    visualizer.add_entry(step, '%s/timeouts per episode' % tag, env.end_result == "command_quota_reached")
                    visualizer.add_entry(step, '%s/agent_1 defaults per episode' % tag, env.end_result == "Agent_1_defaulted")
                    visualizer.add_entry(step, '%s/agent_2 defaults per episode' % tag, env.end_result == "Agent_2_defaulted")
                    visualizer.add_entry(step, '%s/pig caught per episode' % tag, env.end_result == "caught_the_pig")
                    agent.inject_summaries(step)
                    viz_rewards = []
                    obs = env.reset()
                    if obs:
                        break

            # select an action
            action = agent.act(obs, reward, agent_done, is_training=True)
            # take a step
            obs, reward, agent_done = env.do(action)
            viz_rewards.append(reward)

            #agent.inject_summaries(step)


def run_experiment(agents_def):
    assert len(agents_def) == 2, 'Not enough agents (required: 2, got: %d)'\
                % len(agents_def)

    processes = []
    for agent in agents_def:
        p = Thread(target=agent_factory, kwargs=agent)
        p.daemon = True
        p.start()

        # Give the server time to start
        if agent['role'] == 0:
            sleep(1)

        processes.append(p)

    try:
        # wait until only the challenge agent is left
        while active_count() > 2:
            sleep(0.1)
    except KeyboardInterrupt:
        print('Caught control-c - shutting down.')


if __name__ == '__main__':
    arg_parser = ArgumentParser('Pig Chase My Agent experiment')
    arg_parser.add_argument('-e', '--epochs', type=int, default=5,
                            help='Number of epochs to run.')
    arg_parser.add_argument('clients', nargs='*',
                            default=['0.0.0.0:10000', '0.0.0.0:10001'],
                            help='Minecraft clients endpoints (ip(:port)?)+')
    args = arg_parser.parse_args()

    logdir = BASELINES_FOLDER % (datetime.utcnow().isoformat())
    if 'malmopy.visualization.tensorboard' in sys.modules:
        visualizer = TensorboardVisualizer()
        visualizer.initialize(logdir, None)
    else:
        visualizer = ConsoleVisualizer()

    agents = [{'name': agent, 'role': role,
               'clients': args.clients, 'max_epochs': args.epochs,
               'logdir': logdir, 'visualizer': visualizer}
              for role, agent in enumerate(ENV_AGENT_NAMES)]

    run_experiment(agents)

