# Pathfinding Mission
The aim of this mission is to verify whether agents have a better learning performance in Continuous Movement Model than it in Discrete Movement Model.
## Structure
### Map
* FeiEight.xml: This is a map for Continuous Movement Model
* FeiEight2.xml: This is a map for Discrete Movement Model
### Files
All of these files utilize Q-learning algorithm but give different expressions of the state in Q-table and run in different Movement Models:
* **FourDirectionsDiscrete.py:** This is the file agents run in the Discrete Movement Model. It is the baseline. 
* **FourDirectionsContinuous_1.0.py:** This is the file agents run in the Continuous Movement Model in which the Q-table state is expressed by the coordinate of agent (X, Z) and the number of alternative actions is four. 
* **FourDirectionsContinuous_2.0.py:** This is the file agents run in the Continuous Movement Model in which the Q-table state is expressed by the coordinate of agent as well as the orientation of agent, namely (X, Z, Yaw). And the number of alternative actions is six.
* **EightDirectionsContinuous_1.0.py:** This is the file agents run in the Continuous Movement Model in which the Q-table state is expressed by the coordinate of agent (X, Z) and the number of alternative actions is eight. That means the available directions which agent move to is eight. 
* **EightDirectionsContinuous_2.0.py:** This is the file agents run in the Continuous Movement Model in which the Q-table state is expressed by the coordinate of agent as well as the orientation of agent, namely (X, Z, Yaw). And the number of alternative actions is six.
## How to use
* Open a Terminal in the corresponding path
* ` python FourDirectionsContinuous2.0.py`