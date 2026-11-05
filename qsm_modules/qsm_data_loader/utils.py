#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 15:01:32 2020

@author: cds


"""


import torch
#from QSMnet import QSMnet
import numpy as np

def padding_data(input_field):
    N = np.shape(input_field)
    N_16 = np.ceil(np.divide(N,16.))*16
    N_dif = np.int16((N_16 - N) / 2)
    npad = ((N_dif[0],N_dif[0]),(N_dif[1],N_dif[1]),(N_dif[2],N_dif[2]))
    pad_field = np.pad(input_field, pad_width = npad, mode = 'constant', constant_values = 0)
    pad_field = np.expand_dims(pad_field, axis=0)
    pad_field = np.expand_dims(pad_field, axis=0)
    return pad_field, N_dif, N_16

def crop_data(result_pad, N_dif):
    result_pad = result_pad.squeeze()
    N_p = np.shape(result_pad)
    result_final  = result_pad[N_dif[0]:N_p[0]-N_dif[0],N_dif[1]:N_p[1]-N_dif[1],N_dif[2]:N_p[2]-N_dif[2]]
    return result_final

#%%
# if __name__=="__main__":
#     net = QSMnet().cuda(0)
#     inp = np.random.rand(170,170,160)
#     inp, N_dif, N_16 = padding_data(inp)
#     print('input shape:',inp.shape)
#     inp = torch.tensor(inp).float()
#     print(inp.shape)
#     inp = inp.cuda(0)
#     out = net(inp).squeeze().detach().cpu().numpy()
#     print(out.shape)
#     out = crop_data(out, N_dif)
#     print(out.shape)
    
#%%
