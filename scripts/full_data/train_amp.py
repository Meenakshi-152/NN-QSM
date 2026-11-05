#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:35:07 2022

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:28:22 2022

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:30:38 2022

@author: venkatesh
"""

#%%

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import lr_scheduler
from torch.amp import autocast, GradScaler

import numpy as np
import torch.optim as optim
import time
import tqdm
import scipy.io
import matplotlib.pyplot as plt
import os
from datetime import datetime
from utils import *

#%%

from qsm_modules.qsm_loss_modules.loss import *
from qsm_modules.qsm_data_loader.QSM_Dataset_updated import mydataloader
from qsm_modules.qsm_dw_models.model_for_dw_QSMnet_lambda_p_trainable import QSMnet
from qsm_modules.qsm_dw_models.WideResnet import WideResNet
from qsm_modules.qsm_dw_models.model_for_dw_normal_cnn_lambda_p_trainable import Dw
from qsm_modules.qsm_data_loader.utils import *
from qsm_modules.qsm_dw_models.model_for_dw_deepqsm_lambda_p_trainable import DeepQSM
from qsm_modules.qsm_dw_models.lmunet_deep_supervision import LightMUNet
from qsm_modules.qsm_dw_models.ViT3D import ViT3DProximal

import pandas as pd
#%%

from skimage.metrics import structural_similarity as ssim



def z_real_to_z_complex(z_real):
  z_complex_recon=torch.complex(z_real[:,0,:,:,:].unsqueeze(1),z_real[:,1,:,:,:].unsqueeze(1))
  # print('\n at z_real_to_z_complex sum',torch.sum(z_real[:,1,:,:,:].unsqueeze(1)))
  
  return z_complex_recon

def z_complex_to_z_real(z_complex):
  z_real=z_complex.real
  z_imag=z_complex.imag
  # print('\n at z_complex_to_z_real sum',torch.sum(z_complex.imag))
  z_real_recon=torch.cat([z_real,z_imag],axis=1)
  return z_real_recon


#%%

# paramaters for dipole kernel
matrix_size = [64, 64, 64]
voxel_size = [1,  1,  1]

device_id=0 # GPU Number

#for restoring weights
restore=False

K_unrolling=1
batch_size=2
No_samples=16800
epoch=0
epsilon=1e-5
learning_rate=225e-6
break_amount=None
is_data_normalized=True
model='WideResNet20'
#model='LightMUNet'
#model = 'ViT3D'
MM_steps=1

#%%
"""data_source='given_single_patient_data'
Training_patient_no=1
data_source_no=2
if(data_source=='generated_data'):

    raw_data_path='../QSM_data/data_for_experiments/generated_data/raw_data/'
    data_path='../QSM_data/data_for_experiments/generated_data/data_source_1/'
    #data_path='../QSM_data/data_for_experiments/generated_data/single_patient_patches/patient_'
    #patients_list =[7,32,9,10]

elif(data_source=='given_data'):

    raw_data_path='../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
    
    if(data_source_no==1):
        patients_list =[7,8,9,10,11,12]
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_1//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==2):
        patients_list =[10,11,12,1,2,3]
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_2//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==3):
        patients_list =[1,2,3,4,5,6]
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_3//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==4):
        patients_list =[4,5,6,7,8,9]
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_4//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    
elif(data_source=='generated_noisy_data'):
    raw_data_path='../QSM_data/data_for_experiments/generated_data/raw_data_noisy_sigma_0.1/'
    data_path='../QSM_data/data_for_experiments/generated_data/data_source_1_sigma_0.1/'

elif(data_source=='given_single_patient_data'):
    raw_data_path='../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
    if(Training_patient_no==1):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_1/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(Training_patient_no==2):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_2/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(Training_patient_no==3):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_3/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    if(Training_patient_no==4):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_4/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'



import ostrain_1_patient.csv
#print(os.listdir(raw_data_path))
#print(os.listdir(data_path))
print('csv_path:',csv_path)

print("data_path:",data_path)
"""
#%%
# making directory for saving models
print ('*******************************************************')
#data_source = "Single_patient"
data_source = "Full_patient"
start_time=time.time()
experiments_folder=f"savedModelsfull_wide20/Neumann_QSM_MODELS_dw_QSMnet/experiments_on_given_data/dw_{model}20/data_training_without_sampling/"

experiment_name=datetime.now().strftime("%b_%d_%I_%M_%P_")+"model_K_"+ str(K_unrolling)+"_"+data_source+'_dw_'+model;
cwd=os.getcwd()

directory=experiments_folder+"/"+experiment_name+"/"
print(directory)
print('Model will be saved to  :', directory)
#print(os.listdir('/home/guest/Documents/Meenakshi/Image_Quality_Assessment/dataset/train_tkd_2'))
#%%


#%%

#train_csv_path = "train_1_patient.csv"------ fold 1 path
#train_csv_path = "train_1_patient_updated.csv"
train_csv_path = "/home/guest/Documents/Siraj TM/Major Project/DeepQSM/Data Fold 1/Training Data/train1.csv"
val_csv_path = "/home/guest/Documents/Siraj TM/Major Project/DeepQSM/Data Fold 1/Validation Data/val1.csv"
train_data_path = "/home/guest/Documents/Siraj TM/Major Project/DeepQSM/Data Fold 1/Training Data/"
#print(os.listdir('/home/guest/Documents/Meenakshi/Image_Quality_Assessment/dataset/train_tkd_2'))
val_data_path = "/home/guest/Documents/Siraj TM/Major Project/DeepQSM/Data Fold 1/Validation Data/"
loader = mydataloader(train_csv_path, train_data_path)
trainloader = DataLoader(loader, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)

valdata    = mydataloader(val_csv_path, val_data_path)
valloader  = DataLoader(valdata, batch_size = batch_size, shuffle=True, num_workers=1,drop_last=True)


#%%

if(not is_data_normalized):
    sus_mean=0
    sus_std=1
    print('\n\n data is not normalized..................\n\n ')
#/home/guest/Documents/Meenakshi/Image_Quality_Assessment/dataset/train_tkd
else:
    stats = scipy.io.loadmat('/home/guest/Documents/Siraj TM/Major Project/DeepQSM/Data Fold 1/Training Data/'+'tr-stats.mat')
    sus_mean= torch.tensor(stats['out_mean']).cuda(device_id)
    sus_std = torch.tensor(stats['out_std' ]).cuda(device_id)
    print(sus_mean,sus_std)



#%%
try:
    os.makedirs(directory)
except:
    print("Exception...")

# making a log file.......

#%%
dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
dk=dk.float()
dk = torch.unsqueeze(dk, dim=0)

dk=dk.cuda(device_id)
Dk_square=dk * dk

print('dk.shape',dk.shape)
print('Dk_square.shape',Dk_square.shape)

#%%


if(model=='deepqsm'):
    dw = DeepQSM().cuda(device_id)
elif(model=='QSMnet'):
    dw=QSMnet().cuda(device_id)
elif(model=='WideResNet20'):
    dw=WideResNet().cuda(device_id)
elif(model=='simple_cnn'):
    dw=Dw().cuda(device_id)
elif(model=='LightMUNet'):
    dw = LightMUNet(
        spatial_dims=3,
        init_filters=8,
        in_channels=2,      # same as WideResNet: 2ch complex input
        out_channels=2,     # same as WideResNet: 2ch complex output
        blocks_down=(1,2,2,4),
        blocks_up=(1,1,1),
    ).cuda(device_id)
    dw.Eta_val = torch.nn.Parameter(torch.Tensor([0.5])).cuda(device_id)
    dw.p       = torch.nn.Parameter(torch.Tensor([2.0]), requires_grad=False).cuda(device_id)
# elif(model=='VIT'):
#     dw = ViT3DReconstruct(
#         in_channels=2,
#         out_channels=2,
#         img_size= (64,64,64)
#     ).cuda(device_id)
#     dw.Eta_val = torch.nn.Parameter(torch.Tensor([0.5])).cuda(device_id)
#     dw.p       = torch.nn.Parameter(torch.Tensor([2.0]), requires_grad=False).cuda(device_id)
    
elif(model=='ViT3D'):
    dw = ViT3DProximal(
        in_channels  = 2,
        out_channels = 2,
        img_size     = (64, 64, 64),
        patch_size   = 4,
        embed_dim    = 96,
        depth        = 6,
        num_heads    = 4,
        window_size  = 4,
    ).cuda(device_id)
    
dw1 = torch.nn.DataParallel(dw, device_ids=[device_id])  
print("Before checkpoint:", dw.Eta_val.item())
#dw.load_state_dict(torch.load('best_models/best_k1_60.pth'))
#dw.load_state_dict(torch.load('best_models/best_k1_60_2.pth'))
#dw.load_state_dict(torch.load('best_models/DeepQSM_k1_.pth'))
#dw.load_state_dict(torch.load('best_models/WideResNet16_k1.pth'))
#dw.load_state_dict(torch.load('best_models/WideResNet14_k1.pth'))
#dw.load_state_dict(torch.load('best_models/WideResNet20_lim_k1_70.pth'))
#dw.load_state_dict(torch.load('best_models/WideResNet20_lim_k4_ini_k1_eta005.pth'))
#dw.load_state_dict(torch.load('best_models/WideResNet20_im_k1_2_without_sch.pth'))
#dw.load_state_dict(torch.load('best_models/WideResNet20_lim_best_k4_without_sch.pth'))
#dw.load_state_dict(torch.load('best_models_lim/WideResNet20_model_best_k4_ini_k1_eta005_2.pth'))
#dw.load_state_dict(torch.load('best_models2/LightMUNet20_model_best_k1_full_eta05_lr1e-4.pth'))
#dw.load_state_dict(torch.load('best_models_final/WideResNet20_model_best_k4_full_eta055_lr1e-4_batch2.pth'))
#dw.load_state_dict(torch.load('best_models_lim/WideResNet20_model_best_k4_full_eta0038.pth'))
#dw.load_state_dict(torch.load('best_models/WideResNet18_k1.pth'))
#dw.load_state_dict(torch.load('best_models2/LightMUNet20_model_best_1.pth'))
#dw.load_state_dict(torch.load('best_models_final/WideResNet22_model_best_k1_full_eta1_lr1e-4_batch2.pth'))
#dw.load_state_dict(torch.load('best_models/WideResNet24_model_best_k1_lim_eta05_lr1e-4_batch2.pth'))
#dw.load_state_dict(torch.load('fold3/WideResNet20_model_best_k1_full_batch2_e-4.pth'))
#dw.load_state_dict(torch.load('fold4/WideResNet20_model_best_k1_full_batch2_e-4.pth'))
#dw.load_state_dict(torch.load('fold3/WideResNet20_model_best_k4_full_batch2_2_5e-4_drp_0.2.pth'))
#dw.load_state_dict(torch.load('fold4/WideResNet20_model_best_k4_full_batch2_2e-4.pth'))
#dw.load_state_dict(torch.load('fold4/WideResNet20_model_best_k4_full_batch2_2_5e-4_drp_0.2.pth'))
#dw.load_state_dict(torch.load('fold_2/WideResNet20_model_best_k1_full_batch2_2_5e-4_drp_0.225.pth'))
#dw.load_state_dict(torch.load('fold1/WideResNet20_model_best_k1_epoch19_full_batch2_25e-5_drp_0.225.pth'))
#dw.load_state_dict(torch.load('best_models_final/WideResNet20_model_best_k4_full_eta055_lr1e-4_batch2.pth'))
#dw.load_state_dict(torch.load('fold1/WideResNet20_model_best_k3_epoch20_full_batch2_25e-5_drp_0.225_best.pth'))
#dw.load_state_dict(torch.load('fold4/WideResNet20_model_best_k3_epoch43_full_batch2_275e-6_drp_0.225.pth'))
#dw.load_state_dict(torch.load('fold4/WideResNet20_model_best_k3_epoch10_full_batch2_275e-6_drp_0.2.pth'))
#dw.load_state_dict(torch.load('fold4/WideResNet20_model_best_k3_epoch7_full_batch2_275e-6_drp_0.2.pth'))
#dw.load_state_dict(torch.load('fold4/WideResNet20_model_best_k4_full_batch2_2_5e-4_drp_0.2.pth'))
#dw.load_state_dict(torch.load('fold3/WideResNet20_model_best_k4_full_batch2_2e-4_new_2.pth'))
#dw.load_state_dict(torch.load('fold2/WideResNet20_model_best_k4_full_batch2_2755e-6_drp_0.235_best.pth'))
#dw.load_state_dict(torch.load('fold2/WideResNet20_model_best_k3_epoch4_full_batch2_275e-6_drp_0.2_best2.pth'))
#dw.load_state_dict(torch.load('fold1/WideResNet20_model_best_k3_epoch47_full_batch2_275e-6_drp_0.225_best_final.pth'))
#dw.load_state_dict(torch.load('fold3/WideResNet20_model_best_k3_epoch5_full_batch2_275e-6_drp_0.2.pth'))


# with torch.no_grad():
#      dw.Eta_val.fill_(0.6)
# print("after checkpoint:", dw.Eta_val.item())
print("p",dw.p)
print("lambda_val",dw.Eta_val)
#%%
#%%
if restore:
    restore_weights_path=''
    dw.load_state_dict(torch.load(restore_weights_path))
    print("we have restored weights of the model:",restore_weights_path)
    #logging.warning('restore_weights_path:'+(restore_weights_path))
    print("p",dw.p)
    if(model=='LightMUNet'):
        # FIX
        print("lambda_val", dw.Eta_val)
    else:
        print("lambda_val",dw.lambda_val)
        


#%%

# Experiment details....
print('----------------------------------------------------------------')
print('\n Experiment_name:')
print(directory)
print('----------------------------------------------------------')

print('K_unrolling:',K_unrolling)
print('batch_size:',batch_size)
print('No_samples:',No_samples)
print('break_amount:',break_amount)
print('learning_rate:',learning_rate)
print('device_id:',device_id)
print('----------------------------------------------------------')

print('no of training batches:',len(trainloader))
print('----------------------------------------------------------')
print('no of validation batches:',len(valloader))



#logging.warning('no of training batches:'+str(len(trainloader)))
print('----------------------------------------------------------')
#logging.warning('no of validation batches:'+str(len(valloader)))


print('restored model',restore)
#logging.warning('restored model:'+str(restore))

if(restore):
        print("we have restored weights of the model:",restore_weights_path)
#        logging.warning("we have restored weights of the model:"+restore_weights_path)


#%%
ss = sobel_kernel()
ss=ss.float()
ss = ss.cuda(device_id)
# print(ss.shape)

# Count parameters (do this once)
trainable_params = sum(p.numel() for p in dw.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

#%%
optimizer = optim.Adam(dw.parameters(), lr=learning_rate)
#                       ,weight_decay = 1e-5)
#%%
loss_Train=[]
loss_Val=[]
lambda_list=[]
p_list=[]

ssim_Val=[]
hfen_Val=[]
nrmse_Val=[]

#%%


def b_gpu(y,lambda_val, z_k):
    
    # print('\t \t  calling b_gpu:')
    # print('y.shape:',y.shape)
    # print('z_k.shape:',z_k.shape)

    # print(y)
    # print(lambda_val)
    # print(z_k)

    output1 = torch.fft.fftn(y)
    output2 = dk * output1
    output3 = torch.fft.ifftn(output2)
    
    # print('at b_gpu output3:',output3.dtype,output3.shape)
    #print('output3.get_device:', output3.get_device())
    #print('lambda_val.get_device:', lambda_val.get_device())
    #print('z_k.get_device:',z_k.get_device())

    # code added for spinet
    w_square_z_k=w_square*z_k
    # print('w_square_z_k.shape:',w_square_z_k.shape)

    # code added for spinet
    output4 = output3+lambda_val*w_square_z_k
    
    # print('output4.shape',output4.shape,output4.dtype)


    return output4

# x sshould be in gpu....
    
def A_gpu(x,lambda_val,p):
    # print('\t \t calling A')
    # print('---------------------')
    output1 = Dk_square*torch.fft.fftn(x)
    output2 = torch.fft.ifftn(output1)
    
    # print('at A_gpu',output2.dtype,output2.shape)
    
    
    # print('output2.shape:', output2.shape)
    # print('w_square.shape:',w_square.shape)
    # print('x.shape:',x.shape)
    
    # code added for spinet
    
    # print('output2',output2)

    w_square_x=w_square*x
    
    # print('w_square',w_square)
    # print('w_square_x.shape:',w_square_x.shape)
    
    # code added for spinet
    output3 = output2+lambda_val * w_square_x
    # print('at A_gpu',output3.dtype,output3.shape)

    
    
    return output3

def CG_GPU(local_field_gpu, z_k_gpu):

    # print('CG GPU Calling............')
    #print('--------------------------')
    
    x_0 = torch.zeros(size=(1, 1, 64, 64, 64),dtype=torch.float64).cuda(device_id)
    
    temp=b_gpu(local_field_gpu, dw.lambda_val,z_k_gpu)
    
    # print(temp.shape,temp.dtype)

    r_0 = b_gpu(local_field_gpu, dw.lambda_val,z_k_gpu)-A_gpu(x_0,dw.lambda_val,dw.p)
    
    # print('\t r_0.shape',r_0.shape)
    p_0 = r_0

    # print('\t r_0.shape', r_0.shape)
    # print('\t P_0 shape', p_0.shape)

    r_old = r_0
    p_old = p_0
    x_old = x_0

    r_stat = []
    
    r_stat.append(torch.sum(r_old.conj()*r_old).real.item())
    # print('\t r_stat',r_stat)

    for i in range(30):

        # alpha calculation
        r_old_T_r_old = torch.sum(r_old.conj()*r_old)
        # print('\t r_old_T_r_old',r_old_T_r_old,r_old_T_r_old.shape)


        if(r_old_T_r_old.real.item()<1e-10):
            # print('r_stat',r_stat,r_old_T_r_old.item(),'iteration:',len(r_stat))
            
            # logging.warning('r_stat')
            # logging.warning(r_stat)
            # logging.warning(r_old_T_r_old.item())
            # logging.warning('iteration:'+str(len(r_stat)))
            
            return x_old
        
        
        if(r_old_T_r_old.real.item()>r_stat[-1] and r_stat[-1] < 1e-06):
            # print("Convergence issue:",r_old_T_r_old.item(),r_stat[-1])
            
            # logging.warning("Convergence issue:")
            # logging.warning(r_old_T_r_old.item())
            # logging.warning(r_stat[-1])
            return x_old


        r_stat.append( torch.sum(r_old.conj()* r_old).real.item())

        # print('dw.lambda_val,dw.p',dw.lambda_val.item(),dw.p.item())
        # print('dw.lambda_val',dw.lambda_val)
        # print('dw.p',dw.p)
        Ap_old = A_gpu(p_old,dw.lambda_val,dw.p)
        # print('\t Ap_old.shape',Ap_old.shape)
        # print('\t Ap_old',Ap_old)
        
        p_old_T_A_p_old = torch.sum(p_old.conj() * Ap_old)
        # print('\t p_old_T_A_p_old',p_old_T_A_p_old.item())
        
        alpha = r_old_T_r_old/p_old_T_A_p_old
        # print('\t alpha',alpha.item())

        # updating the x
        x_new = x_old+alpha*p_old
        # print('\t x_new',x_new)
        # print('\t x_new.shape',x_new.shape,x_new.dtype)
        

        # updating the remainder
        r_new = r_old-alpha*Ap_old
        # print('\t r_new.shape',r_new.shape)
        
        # beta calculation
        r_new_T_r_new = torch.sum(r_new.conj() * r_new)

        #r_stat.append(r_new_T_r_new.real.item())
        
        beta = r_new_T_r_new/r_old_T_r_old
        
        # print('\t beta',beta,beta.dtype)

        # new direction p calculationubu 
        p_new = r_new+beta*p_old
        
        
        # print('p_new',p_new)
        # print('\t p_new.shape',p_new.shape,p_new.dtype)

        # preparing for the new iteration...

        r_old = r_new
        p_old = p_new
        x_old = x_new

    # print('\t x_new',x_new.shape,x_new.dtype)
    # print(r_stat)
    
    return x_new

#%%


break_amount = (len(trainloader) * 1) // 1

print(f'Taking {break_amount} batches out of {len(trainloader)} batches')

scaler = GradScaler()
#scaler = GradScaler()
epoch=0
least_val_loss = np.inf
#scheduler = warmup_lr_scheduler(optimizer, warmup_epochs=5, total_epochs = 31)#, base_lr=1e-3, max_lr=1e-2)
#%%
#%%
for epoch in range(60):

    runningLoss = 0
    
    # Store metric values for all validation volumes/batches
    ssim_epoch_values = []
    hfen_epoch_values = []
    nrmse_epoch_values = []

    print('\n-------------------------------------------------------------------------------------------------\n')
    print('epoch---', epoch)

    # ============================================================
    # TRAINING
    # ============================================================

    dw.train()

    for i, data in enumerate(tqdm.tqdm(trainloader)):

        phs, msk, sus, file_name = data

        # dk_repeat=torch.repeat_interleave(dk, repeats=batch_size,dim=(0))
        dk_repeat = dk.repeat(batch_size, 1, 1, 1, 1)

        phs = phs.cuda(device_id)
        msk = msk.cuda(device_id)
        sus = sus.cuda(device_id)

        if break_amount:
            if i == break_amount:
                break

        # --------------------------------------------------------
        # B0 = Eta * Phi^H * y
        # --------------------------------------------------------

        with autocast(device_type='cuda'):

            B_0_complex = torch.fft.ifftn(
                dk_repeat * torch.fft.fftn(
                    phs,
                    dim=[2, 3, 4]
                ),
                dim=[2, 3, 4]
            )

            B_0_complex = dw.Eta_val * B_0_complex
            B_0_complex = B_0_complex.cuda(device_id)

            # initialize with mask
            B_0_complex = B_0_complex * msk

            # initialize B_k
            B_k_complex_sum = torch.zeros_like(B_0_complex)
            B_k_complex_sum = B_0_complex.clone()

            B_k_complex = B_0_complex
            B_k_complex = B_k_complex * msk

            B_k_set = [B_0_complex.clone()]

            # ----------------------------------------------------
            # Neumann unrolling
            # ----------------------------------------------------

            for k in range(K_unrolling):

                # term-1 calculation
                # X_T_X = torch.fft.ifftn(
                #     dk_repeat * dk_repeat *
                #     torch.fft.ifftn(
                #         B_k_complex,
                #         dim=[2, 3, 4]
                #     ),
                #     dim=[2, 3, 4]
                # )
                
                X_T_X = torch.fft.ifftn(
                    dk_repeat * dk_repeat *
                    torch.fft.fftn(
                        B_k_complex,
                        dim=[2, 3, 4]
                    ),
                    dim=[2, 3, 4]
                )

                term_1_complex = (
                    B_k_complex -
                    dw.Eta_val * X_T_X
                )

                # ------------------------------------------------
                # term-2: regularization
                # ------------------------------------------------

                B_k_real = z_complex_to_z_real(
                    B_k_complex
                )

                B_k_real = (
                    B_k_real - sus_mean
                ) / sus_std

                if model == 'LightMUNet':

                    R_X_real_term = dw(B_k_real)

                else:

                    R_X_real_term = dw(B_k_real)

                R_X_real_term = (
                    R_X_real_term * sus_std +
                    sus_mean
                )

                R_X_complex_term = (
                    z_real_to_z_complex(
                        R_X_real_term
                    )
                )

                # ------------------------------------------------
                # Update B_k
                # ------------------------------------------------

                B_k_complex = (
                    term_1_complex -
                    dw.Eta_val * R_X_complex_term
                )

                B_k_complex = B_k_complex * msk

                B_k_set.append(
                    B_k_complex.clone()
                )

                B_k_complex_sum = (
                    B_k_complex_sum +
                    B_k_complex
                )

            # final reconstruction
            x_k_complex = B_k_complex_sum * msk

        # ========================================================
        # TRAINING LOSS
        # ========================================================
        
        # loss = total_loss_l1(
        #     chi=x_k_complex.real,
        #     y=sus,
        #     b=phs,
        #     d=dk,
        #     m=msk,
        #     sobel=ss
        # )

        loss, l1_loss, grad_loss, physics_loss = total_loss_l1(chi=x_k_complex.real,
            y=sus,
            b=phs,
            d=dk,
            m=msk,
            sobel=ss
        )

        if i == 0:
            print(
                f"L1: {l1_loss.item():.6f}, "
                f"Gradient: {grad_loss.item():.6f}, "
                f"Physics: {physics_loss.item():.6f}, "
                f"Total: {loss.item():.6f}"
            )
        scaler.scale(loss).backward()

        # nn.utils.clip_grad_value_(
        #     dw.parameters(),
        #     clip_value=1.0
        # )

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        runningLoss += loss.item()


    # ============================================================
    # END OF TRAINING LOOP
    # ============================================================

    mean_training_loss = (
        runningLoss / len(trainloader)
    )

    # Save ONE training loss value for this epoch
    loss_Train.append(mean_training_loss)

    print('Training_loss:', mean_training_loss)
    print(
        'Learning Rate:',
        optimizer.param_groups[0]['lr']
    )

    print('---------------------------------------------------------')
    print('Eta:', dw.Eta_val.item())
    print('p:', dw.p.item())
    print('---------------------------------------------------------')


    # ============================================================
    # VALIDATION
    # ============================================================

    dw.eval()

    running_val_Loss = 0

    with torch.no_grad():

        for i, data in enumerate(
            tqdm.tqdm(valloader)
        ):

            try:

                phs, msk, sus, file_name = data

                dk_repeat = dk.repeat(
                    batch_size,
                    1,
                    1,
                    1,
                    1
                )

                phs = phs.cuda(device_id)
                msk = msk.cuda(device_id)
                sus = sus.cuda(device_id)

                # ------------------------------------------------
                # Initialization
                # ------------------------------------------------

                with autocast(device_type='cuda'):

                    B_0_complex = torch.fft.ifftn(
                        dk_repeat *
                        torch.fft.fftn(
                            phs,
                            dim=[2, 3, 4]
                        ),
                        dim=[2, 3, 4]
                    )

                    B_0_complex = (
                        dw.Eta_val *
                        B_0_complex
                    )

                    B_0_complex = (
                        B_0_complex.cuda(device_id)
                    )

                    B_0_complex = (
                        B_0_complex * msk
                    )

                    # initialize
                    B_k_complex_sum = (
                        torch.zeros_like(
                            B_0_complex
                        )
                    )

                    B_k_complex_sum = (
                        B_0_complex.clone()
                    )

                    B_k_complex = B_0_complex
                    B_k_complex = (
                        B_k_complex * msk
                    )

                    B_k_set = [
                        B_0_complex.clone()
                    ]

                    # ------------------------------------------------
                    # Neumann unrolling
                    # ------------------------------------------------

                    for k in range(K_unrolling):

                        # term-1 calculation
                        # X_T_X = torch.fft.ifftn(
                        #     dk_repeat *
                        #     dk_repeat *
                        #     torch.fft.ifftn(
                        #         B_k_complex,
                        #         dim=[2, 3, 4]
                        #     ),
                        #     dim=[2, 3, 4]
                        # )
                        
                        X_T_X = torch.fft.ifftn(
                            dk_repeat * dk_repeat *
                            torch.fft.fftn(
                                B_k_complex,
                                dim=[2, 3, 4]
                            ),
                            dim=[2, 3, 4]
                        )

                        term_1_complex = (
                            B_k_complex -
                            dw.Eta_val *
                            X_T_X
                        )

                        # ------------------------------------------------
                        # Regularization
                        # ------------------------------------------------

                        B_k_real = (
                            z_complex_to_z_real(
                                B_k_complex
                            )
                        )

                        B_k_real = (
                            B_k_real - sus_mean
                        ) / sus_std

                        if model == 'LightMUNet':

                            R_X_real_term = dw(
                                B_k_real
                            )

                        else:

                            R_X_real_term = dw(
                                B_k_real
                            )

                        R_X_real_term = (
                            R_X_real_term *
                            sus_std +
                            sus_mean
                        )

                        R_X_complex_term = (
                            z_real_to_z_complex(
                                R_X_real_term
                            )
                        )

                        # ------------------------------------------------
                        # Update B_k
                        # ------------------------------------------------

                        B_k_complex = (
                            term_1_complex -
                            dw.Eta_val *
                            R_X_complex_term
                        )

                        B_k_complex = (
                            B_k_complex * msk
                        )

                        B_k_set.append(
                            B_k_complex.clone()
                        )

                        B_k_complex_sum = (
                            B_k_complex_sum +
                            B_k_complex
                        )

                    # final reconstruction
                    x_k_complex = (
                        B_k_complex_sum * msk
                    )
                    
                    # ====================================================
                    # VALIDATION METRICS
                    # ====================================================

                    pred_np = x_k_complex.real.detach().cpu().numpy()
                    gt_np   = sus.detach().cpu().numpy()

                    for b in range(pred_np.shape[0]):

                        pred_volume = pred_np[b, 0]
                        gt_volume   = gt_np[b, 0]

                        # -----------------------------
                        # NRMSE
                        # -----------------------------
                        numerator = np.linalg.norm(pred_volume - gt_volume)
                        denominator = np.linalg.norm(gt_volume)

                        if denominator > 0:
                            nrmse_value = 100 * numerator / denominator
                            nrmse_epoch_values.append(nrmse_value)
                        # -----------------------------
                        # SSIM
                        # -----------------------------
                        data_range = gt_volume.max() - gt_volume.min()

                        if data_range > 0:
                            ssim_value = ssim(
                                gt_volume,
                                pred_volume,
                                data_range=data_range,
                                win_size=7
                            )

                            ssim_epoch_values.append(ssim_value)

                        # -----------------------------
                        # HFEN
                        # -----------------------------
                        # Use your existing HFEN function here.
                        hfen_value = (
                            100
                            * compute_hfen(
                                torch.from_numpy(pred_volume).float(),
                                torch.from_numpy(gt_volume).float()
                            ).item()
                        )

                        hfen_epoch_values.append(hfen_value)
                
                # ====================================================
                # VALIDATION LOSS
                # ====================================================

                # loss = total_loss_l1(
                #     chi=x_k_complex.real,
                #     y=sus,
                #     b=phs,
                #     d=dk,
                #     m=msk,
                #     sobel=ss
                # )
                loss, l1_loss, grad_loss, physics_loss = total_loss_l1(
                    chi=x_k_complex.real,
                    y=sus,
                    b=phs,
                    d=dk,
                    m=msk,
                    sobel=ss
                )

                running_val_Loss += loss.item()

            except Exception as e:

                print(e)
                print('error at', i)


    # ============================================================
    # END OF VALIDATION LOOP
    # ============================================================

    mean_validation_loss = (
        running_val_Loss / len(valloader)
    )

    # Save ONE validation loss value for this epoch
    loss_Val.append(
        mean_validation_loss
    )
    
    # ============================================================
    # MEAN VALIDATION METRICS FOR THIS EPOCH
    # ============================================================

    mean_validation_ssim = np.mean(ssim_epoch_values)
    mean_validation_hfen = np.mean(hfen_epoch_values)
    mean_validation_nrmse = np.mean(nrmse_epoch_values)

    ssim_Val.append(mean_validation_ssim)
    hfen_Val.append(mean_validation_hfen)
    nrmse_Val.append(mean_validation_nrmse)

    print('Validation_SSIM :', mean_validation_ssim)
    print('Validation_HFEN :', mean_validation_hfen)
    print('Validation_NRMSE:', mean_validation_nrmse)



    # ============================================================
    # SAVE BEST MODEL
    # ============================================================

    if mean_validation_loss < least_val_loss:

        least_val_loss = mean_validation_loss

        torch.save(
            dw.state_dict(),
            directory +
            f'{model}_model_best_k{K_unrolling}_epoch{epoch}_full_batch2_275e-6_drp_0.2.pth'
        )

        print(
            f"New best model found at epoch : {epoch}"
        )

    else:

        print(
            f"Best model not found at epoch : {epoch}"
        )


    # ============================================================
    # SAVE ETA AND P FOR THIS EPOCH
    # ============================================================

    # ONE Eta value per epoch
    lambda_list.append(
        dw.Eta_val.item()
    )

    # ONE p value per epoch
    p_list.append(
        dw.p.item()
    )

    print(
        'Validation_loss:',
        mean_validation_loss
    )

    print(
        'Eta:',
        dw.Eta_val.item()
    )

    print(
        'p:',
        dw.p.item()
    )

    print(
        'Checking what is the K value:',
        K_unrolling
    )

    print(
        '\n-------------------------------------------------------------------------------------------------\n'
    )


# =================================================================
# AFTER ALL 60 EPOCHS
# =================================================================

model_details = {

    "Epoch": list(
        range(
            1,
            len(loss_Train) + 1
        )
    ),

    "Training_Loss": loss_Train,

    "Validation_Loss": loss_Val,

    "Validation_SSIM": ssim_Val,

    "Validation_HFEN": hfen_Val,

    "Validation_NRMSE": nrmse_Val,

    "Eta": lambda_list,

    "p": p_list
}


# Convert lists to DataFrame
df = pd.DataFrame(
    model_details
)


# Save CSV ONCE after all epochs
csv_path = os.path.join(
    directory,
    "training_history2.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print(
    "Training history saved to:",
    csv_path
)