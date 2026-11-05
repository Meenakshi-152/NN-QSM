#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 12:25:58 2022

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:12:19 2021

@author: venkatesh
"""
import torch.nn.functional as F
import os
import copy
import torch
import codecs
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets


#%%


class Dw(nn.Module):
    def __init__(self):
        super(Dw,self).__init__()

        self.lambda_val = 0.05

        self.p=torch.nn.Parameter(torch.Tensor([1.4]), requires_grad = True)

        self.conv1=nn.Sequential(nn.Conv3d(2,64,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(64),
                            nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(64),
                            nn.ReLU())
        self.conv3=nn.Sequential(nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(64),
                            nn.ReLU())
        self.conv4=nn.Sequential(nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(64),
                            nn.ReLU())
        self.conv5=nn.Sequential(nn.Conv3d(64,2,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(2))

        

    def forward(self,x):
        x1=self.conv1(x)
        # print(x1.shape)
        x2=self.conv2(x1)
        # print(x2.shape)

        x3=self.conv3(x2)
        # print(x3.shape)

        x4=self.conv4(x3)
        # print(x4.shape)

        x5=self.conv5(x4)    
        # print(x5.shape)

        out = x + x5
        return out
    
#%%


# # creating a model
# model=Dw()
# #%%
# # creating a input of size 2x64x64x64
# # at 0 th axis 1 st slice real part
# # at 0 th axis 2 nd slice imaginary part 

# input = torch.randn(10,2,64,64, 64)
# output=model(input)

# print('input',input.shape)
# print('output.shape',output.shape)
