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


class ResBlock(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(ResBlock,self).__init__()
        self.conv_11=nn.Sequential(nn.Conv3d(input_channels,output_channels,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(output_channels),
                            nn.ReLU())
        self.conv_21=nn.Sequential(nn.Conv3d(output_channels,output_channels,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(output_channels))
        self.relu_22=nn.ReLU();

    def forward(self,x):
        x_11=self.conv_11(x);

        x_21=self.conv_21(x_11);

        x_22=self.relu_22(x+x_21)
        return x_22


#%%



# creating a model
model=ResBlock(input_channels=2, output_channels=64)
model=model.float().cuda()
#%%
# creating a input of size 2x64x64x64
# at 0 th axis 1 st slice real part
# at 0 th axis 2 nd slice imaginary part 

input = torch.randn(10,2,64,64,64).float().cuda()
output=model(input)

print('input',input.shape)
print('output.shape',output.shape)

#%%
# sigmoid_function=torch.nn.Sigmoid()
# p=1+sigmoid_function(model.p)
# lambda_val=sigmoid_function(model.lambda_val)
# print(p.device,lambda_val.device)
# print(p,lambda_val)




#%%
class Dw(nn.Module):
    def __init__(self):
        super(Dw,self).__init__()

        self.lambda_val = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad = True)

        self.p=torch.nn.Parameter(torch.Tensor([2]), requires_grad = True)

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
# model=Dw().cuda()
# #%%
# # creating a input of size 2x64x64x64
# # at 0 th axis 1 st slice real part
# # at 0 th axis 2 nd slice imaginary part 

# input = torch.randn(10,2,64,64, 64).cuda()
# output=model(input)

# print('input',input.shape)
# print('output.shape',output.shape)

#%%
# sigmoid_function=torch.nn.Sigmoid()
# p=1+sigmoid_function(model.p)
# lambda_val=sigmoid_function(model.lambda_val)
# print(p.device,lambda_val.device)
# print(p,lambda_val)
