# Pig Chase Mission

This mission is based on the malmo collaborative AI Challenge - [Pig Chase](https://github.com/Microsoft/malmo-challenge/tree/master/ai_challenge/pig_chase)

## Structure

There are several major files:

* my_pig_agent.py: this is the Q_Learning agent utilized in the project

* pig_chase.xml: the map of the game

* pig_chase_eval_myAgent.py: this is the file to run my pig agent against the challenge agent

* pig_chase_eval_randomAgent.py: this is the baseline 1 to run the random agent against the challenge agent

* pig_chase_eval_focusedAgent.py: this is the baseline 2 to run the focused agent aginst the challenge agent

* q_table.txt: the stored Q table derived from the 20,000-eposide training.

* result.txt: the reward of each eposide

* ave_result.csv: the average reward


## How to use

### To Run the Pig Chase

* Start two instances of the Malmo Client on ports `10000` and `10001`

* Open a Terminal in the corresponding path

* `python pig_chase_eval_myAgent.py`


## Author

Xiaoyu Wang 799778
xiaoyuw6@student.unimelb.edu.au
