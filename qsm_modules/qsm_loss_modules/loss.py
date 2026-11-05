#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 10:30:13 2021

@author: venkatesh
"""

import math
import torch
import torch.nn.functional as F

import numpy as np
import torch

def dipole_kernel(matrix_size, voxel_size, B0_dir=[0,0,1], device=None):
    """
    Creates dipole kernel for QSM reconstruction.
    
    Args:
        matrix_size: Tuple of (H, W, L)
        voxel_size: Tuple of voxel dimensions
        B0_dir: Magnetic field direction
        device: torch.device or string ('cuda', 'cpu'). If None, uses 'cuda' if available
    
    Returns:
        torch.Tensor: Dipole kernel on specified device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    [Y,X,Z] = np.meshgrid(np.linspace(-int(matrix_size[1]/2),int(matrix_size[1]/2)-1, matrix_size[1]),
                       np.linspace(-int(matrix_size[0]/2),int(matrix_size[0]/2)-1, matrix_size[0]),
                       np.linspace(-int(matrix_size[2]/2),int(matrix_size[2]/2)-1, matrix_size[2]))
    X = X/(matrix_size[0])*voxel_size[0]
    Y = Y/(matrix_size[1])*voxel_size[1]
    Z = Z/(matrix_size[2])*voxel_size[2]
    D = 1/3 - np.divide(np.square(X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2]), 
                        np.square(X)+np.square(Y)+np.square(Z) + np.finfo(float).eps)
    D = np.where(np.isnan(D),0,D)

    D = np.roll(D,int(np.floor(matrix_size[0]/2)),axis=0)
    D = np.roll(D,int(np.floor(matrix_size[1]/2)),axis=1)
    D = np.roll(D,int(np.floor(matrix_size[2]/2)),axis=2)
    D = np.float32(D)
    
    # Create tensor directly on the specified device
    D = torch.tensor(D, device=device).unsqueeze(dim=0)
    
    return D


def sobel_kernel(device=None):
    """
    Creates 3D Sobel kernel for gradient computation.
    
    Args:
        device: torch.device or string ('cuda', 'cpu'). If None, uses 'cuda' if available
    
    Returns:
        torch.Tensor: Sobel kernels on specified device with shape (3, 1, 3, 3, 3)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    s = [
        [
            [1,   2,   1],
            [2,   4,   2],
            [1,   2,   1]
        ],
        [
            [0,   0,   0],
            [0,   0,   0],
            [0,   0,   0]
        ],
        [
            [-1, -2, -1],
            [-2, -4, -2],
            [-1, -2, -1]
        ]
    ]
    
    # Create tensor directly on the specified device
    s = torch.FloatTensor(s).to(device)
    sx = s
    sy = s.permute(1, 2, 0)
    sz = s.permute(2, 0, 1)
    ss = torch.stack([sx, sy, sz]).unsqueeze(1)

    return ss


def total_loss_l1(chi, b, d, m, sobel,y=None,get_field=False):    
    
    # chi = predicetd susc
    # y   = cosmos susc
    # b   = phs
    # d   = dipole kernel
    # m   = mask
    # y_mean = label mean
    # y_std  = label std
    # b_mean = input_mean
    # b_std  = input_std
    
    def _l1error(x1, x2):
        return torch.mean(torch.abs(x1 - x2))


    def _chi_to_b(chi, b, d, m):
        
        # chi = predicetd susc
        # b   = phs
        # d   = dipole kernel
        # m   = mask
        
        chi_fourier = torch.fft.fftn(chi,dim=[2,3,4])
        b_hat_fourier = (chi_fourier * d)
        b_hat = torch.real(torch.fft.ifftn(b_hat_fourier,dim=[2,3,4]))
    
        # Multiply masks
        b = b * m
        b_hat = b_hat * m    
        return b, b_hat
    
    def loss_l1(chi, y):
        return _l1error(chi, y)


    def loss_model(b, b_hat):    
        return _l1error(b, b_hat)
    
    def loss_gradient(chi, y, sobel):
        temp=y - chi
        difference   = F.conv3d(temp.float()  ,   sobel.float(), padding=1)
        return torch.mean(torch.abs(difference))
    
    w1 = 0.5
    w2 = 1
    w3 = 0.5
     
    b, b_hat = _chi_to_b(chi, b, d, m)
    
    if get_field:
        return b_hat
    
    # loss_model = w1 * loss_model(b, b_hat)
    
    # loss_l1    = w2 * loss_l1(chi, y)
    
    # loss_grad  = w3 * loss_gradient(chi, y, sobel)
    
    # loss = loss_l1 +  loss_grad +loss_model
    
    # #loss = loss_model 
    # #loss = loss_l1 
    # #loss = loss_grad    
    # return loss


    # Calculate raw losses
    loss_model_raw = loss_model(b, b_hat)
    loss_l1_raw = loss_l1(chi, y)
    loss_grad_raw = loss_gradient(chi, y, sobel)

    # Apply weights
    loss_model_weighted = w1 * loss_model_raw
    loss_l1_weighted = w2 * loss_l1_raw
    loss_grad_weighted = w3 * loss_grad_raw

    # Total loss
    loss = (
        loss_l1_weighted
        + loss_grad_weighted
        + loss_model_weighted
    )

    return (
        loss,
        loss_l1_weighted,
        loss_grad_weighted,
        loss_model_weighted
    )

#%%
def unsupervised_model_loss(chi, y, b, d, m):
    def _chi_to_b(chi, b, d, m):
        
        # chi = predicetd susc
        # b   = phs
        # d   = dipole kernel
        # m   = mask
        
        chi_fourier = torch.fft.fftn(chi,dim=[2,3,4])
        b_hat_fourier = (chi_fourier * d)
        b_hat = torch.real(torch.fft.ifftn(b_hat_fourier,dim=[2,3,4]))
    
        # Multiply masks
        b = b * m
        b_hat = b_hat * m    
        return b, b_hat
    
    def _l2error(x1,x2):
        return torch.mean(torch.square(x1-x2))
    
    def loss_model(b, b_hat):    
        return _l2error(b, b_hat)
#%%    

def total_loss_l2(chi, y, b, d, m, sobel):    
    
    # chi = predicetd susc
    # y   = cosmos susc
    
    # b   = phs
    # d   = dipole kernel
    # m   = mask
    # y_mean = label mean
    # y_std  = label std
    # b_mean = input_mean
    # b_std  = input_std
    def _l2error(x1,x2):
        return torch.mean(torch.square(x1-x2))

    def _chi_to_b(chi, b, d, m):
        
        # chi = predicetd susc
        # b   = phs
        # d   = dipole kernel
        # m   = mask
        
        chi_fourier = torch.fft.fftn(chi,dim=[2,3,4])
        b_hat_fourier = (chi_fourier * d)
        b_hat = torch.real(torch.fft.ifftn(b_hat_fourier,dim=[2,3,4]))
    
        # Multiply masks
        b = b * m
        b_hat = b_hat * m    
        return b, b_hat
    
    def loss_l2(chi, y):
        return _l2error(chi, y)


    def loss_model(b, b_hat):    
        return _l2error(b, b_hat)
    
    def loss_gradient(chi, y, sobel):
        difference   = F.conv3d(y - chi  ,   sobel.double(), padding=1)
        return torch.mean(torch.square(difference))
    
    w1 = 0.5
    w2 = 1
    w3 = 0.5
     
    b, b_hat = _chi_to_b(chi, b, d, m)
    
    # loss_model = w1 * loss_model(b, b_hat)
    
    # loss_l1    = w2 * loss_l2(chi, y)
    
    # loss_grad  = w3 * loss_gradient(chi, y, sobel)
    
    # loss = loss_l1 + loss_grad
    # #print('loss_model',loss_model)
    # #print(loss_model.item(),loss_grad.item(),loss_l1.item(),loss.item())
    # #loss = loss_model 
    # #loss = loss_l1 
    # #loss = loss_grad    
    
    
    # return loss

    loss_model_raw = loss_model(b, b_hat)
    loss_l1_raw    = loss_l2(chi, y)
    loss_grad_raw  = loss_gradient(chi, y, sobel)

    # Apply weights
    loss_model_weighted = w1 * loss_model_raw
    loss_l1_weighted    = w2 * loss_l1_raw
    loss_grad_weighted  = w3 * loss_grad_raw

    # Total loss
    loss = (
        loss_l1_weighted
        + loss_grad_weighted
        + loss_model_weighted
    )

    return (
        loss,
        loss_l1_weighted,
        loss_grad_weighted,
        loss_model_weighted
    )














