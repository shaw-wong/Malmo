import xml.dom.minidom as MD
import math
import csv
import pandas
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from PIL import Image
from collections import namedtuple



Batch_Size = 128
LR = 0.01
GAMMA = 0.9
Frst_EPSILON = 0.5
Final_EPISILON = 0.01
EPSILON_DECAY = 20000


TARGET_REPLACE_ITER = 100

ACTIONS_DIMENTION = 2

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward' ,'next_state'))





class Net(torch.nn.Module):


    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.output = torch.nn.Linear(256 ,ACTIONS_DIMENTION)
        self.output.weight.data.normal_(0,0.2)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        actions_value = self.output(x.view(x.size(0), -1))
        return actions_value


class DQN(object):


    def __init__(self):
        self.evalueNet = Net()
        self.targetNet = Net()
        self.log = None

        self.learnCounter = 0
        self.memoryCounter = 0
        self.memory_size = 2000
        self.memory = []
        self.optimizer = torch.optim.Adam(self.evalueNet.parameters(), lr = LR)
        self.lossFunction = nn.MSELoss()
        self.epsilon = Frst_EPSILON




    def choose_action(self,x):

        x = Variable(x, volatile=True).type(torch.FloatTensor)
        self.epsilon = Final_EPISILON + (Frst_EPSILON - Final_EPISILON) * math.exp(-1. * self.epsilon / EPSILON_DECAY)





        if np.random.uniform() > self.epsilon :
            actionsValue = self.evalueNet.forward(x)
            action = actionsValue.data.max(1)[1].view(1, 1)

        else:
            # print('RANDOM')
            action = torch.LongTensor([[random.randrange(ACTIONS_DIMENTION)]])

        return action




    def record_transition(self,s,a,r,next_s):


        transition = Transition(s,a,r,next_s)

        i = self.memoryCounter % self.memory_size
        if self.memoryCounter < self.memory_size:
            self.memory.append(transition)
        else:
            self.memory[i] = transition

        self.memoryCounter += 1

    def learn(self):


        if self.learnCounter % TARGET_REPLACE_ITER == 0:
            self.targetNet.load_state_dict(self.evalueNet.state_dict())
        self.learnCounter +=1

        samples = random.sample(self.memory,Batch_Size)

        batch = Transition(*zip(*samples))

        # print('I am here')



        sample_s = Variable(torch.cat(batch.state))
        sample_a = Variable(torch.cat(batch.action))
        sample_r = Variable(torch.cat(batch.reward))
        sample_next_s = Variable(torch.cat(batch.next_state))
        # print(len(sample_s),len(sample_a))
        # exit()

        q_value = self.evalueNet(sample_s).gather(1,sample_a)

        q_next = self.targetNet(sample_next_s).max(1)[0].detach()
        q_target = sample_r + GAMMA * q_next
        q_target = Variable(q_target.data)
        loss = self.lossFunction(q_value, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
















