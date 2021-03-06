# COMP90055 Computing Project

This repository contains the raw code for the Computing project.


## Overview of the Project

Nowadays, most of the AI agents are designed to deal with one specific task who may not perform well in other tasks. For example, an agent is designed to play chess cannot be used to drive a car. This project focuses on building a general agent that can deal with more than one task.

A 2-layer Deep Q-Network (DQN) structure is utilized in this project. 

We made an assumption that the 2-layer structure is more efficient than the 1-layer structure. 

For the 2-layer structure, the lower layer contains two specific agents, pathfinding agent and pig chase agent, who are carefully designed to accomplish specific tasks with high performance. The upper layer is the DQN algorithm with the pixel as the input which records the states of the game and infers which game it is dealing with and then calls the corresponding agent in the lower layer. 

The 1-layer structure only has the DQN algorithm. 

## Structure of The Repository

To make it clear, we put our code in four seperate folders.

* GeneralAgent: it contains the Two-Layer DQN structure and One-Layer DQN structure as the baseline.

* MazeScenario: it contains the Pathfinding Agent and its baseline agent

* PigChase: it contains the the Pig Chase agent and its baseline agent.

* CNN_by_ChongFeng: it contains the CNN agent

All other files come from [Malmo](https://github.com/Microsoft/malmo)

See more details on the README in the specific folder

## How to use

The two agents defined in the lower layer can be independently executed as well.

### Run the 2-layer DQN

* `cd Malmo/GeneralAgent`
* `python TestAgent.py`

### Run the Pathfinding

* `cd Malmo/MazeScenario`
* `python EightDirectionsContinuous_2.0.py`

### Run the Pig Chase

* Start two instances of the Malmo Client on ports `10000` and `10001`
* `cd Malmo/PigChase`
* `python pig_chase_eval_myAgent.py`

## Author

Haoran Sun 839693
haorans@student.unimelb.edu

Fei Teng 809370
fteng1@student.unimelb.edu.au

Xiaoyu Wang 799778
xiaoyuw6@student.unimelb.edu.au

Chong Feng 852833
cfeng2@student.unimelb.edu.au



