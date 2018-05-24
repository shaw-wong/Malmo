import xml.dom.minidom as MD

import csv
import pandas


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np



Batch_Size = 32
LR = 0.01
GAMMA = 0.9
EPSILON = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_SIZE = 700


STATES_DIMENTION = 4
ACTIONS_DIMENTION = 4







class Net(torch.nn.Module):


    def __init__(self):
        super(Net, self).__init__()
        self.input = torch.nn.Linear(STATES_DIMENTION, 50)
        self.input.weight.data.normal_(0,0.2)

        self.output = torch.nn.Linear(50 ,ACTIONS_DIMENTION)
        self.output.weight.data.normal_(0,0.2)

    def forward(self,x):
        x = self.input(x)
        x = F.relu(x)
        actions_value = self.output(x)
        return actions_value


class DQN(object):


    def __init__(self):
        self.evalueNet = Net()
        self.targetNet = Net()
        self.log = None

        self.learnCounter = 0
        self.memoryCounter = 0
        self.memory = np.zeros((MEMORY_SIZE, STATES_DIMENTION * 2 +2))
        self.optimizer = torch.optim.Adam(self.evalueNet.parameters(), lr = LR)
        self.lossFunction = nn.MSELoss()




    def choose_action(self,x):

        x = Variable(torch.unsqueeze(torch.FloatTensor(x),0))
        if np.random.uniform() < EPSILON :
            actionsValue = self.evalueNet.forward(x)
            # print(actionsValue)
            action = torch.max(actionsValue,1)[1].data.numpy()
            # self.log.debug(action)
        else:
            action = np.random.randint(0,ACTIONS_DIMENTION)

        return action




    def record_transition(self,s,a,r,next_s):


        transition = np.hstack((s,[a,r],next_s))

        i = self.memoryCounter % MEMORY_SIZE
        self.memory[i, :] = transition
        self.memoryCounter += 1

    def learn(self):
        if self.learnCounter % TARGET_REPLACE_ITER == 0:
            self.targetNet.load_state_dict(self.evalueNet.state_dict())
        self.learnCounter +=1

        sampleIndex = np.random.choice(MEMORY_SIZE,Batch_Size)
        sampleMemory = self.memory[sampleIndex, :]

        sample_s = Variable(torch.FloatTensor(sampleMemory[:,:STATES_DIMENTION]))
        sample_a = Variable(torch.LongTensor(sampleMemory[:, STATES_DIMENTION:STATES_DIMENTION+1].astype(int)))
        sample_r = Variable(torch.FloatTensor(sampleMemory[:,STATES_DIMENTION+1:STATES_DIMENTION+2]))
        sample_next_s = Variable(torch.FloatTensor(sampleMemory[:, -STATES_DIMENTION:]))

        q_value = self.evalueNet(sample_s).gather(1,sample_a)

        q_next = self.targetNet(sample_next_s).detach()
        q_target = sample_r + GAMMA * q_next.max(1)[0].view(Batch_Size,1)
        loss = self.lossFunction(q_value, q_target)
        # print(loss)


        # loss_value = float(loss)
        # if loss_value!= None:
        #     with open('/Users/sunhaoran/Desktop/analysis.csv', 'a') as csvfile:
        #         writer = csv.writer(csvfile)
        #         writer.writerow([loss_value])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()





