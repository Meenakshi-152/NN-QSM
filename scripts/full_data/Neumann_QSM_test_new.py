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
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np

#import sys
#sys.path.append('../DeepQSM')
from myDataset import qsmdata

import numpy as np
import torch.optim as optim
import time
import tqdm
import scipy.io
import matplotlib
matplotlib.use("Agg")   # Must come BEFORE pyplot
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse
#%%

from qsm_modules.qsm_loss_modules.loss import *
from qsm_modules.qsm_data_loader.QSM_Dataset_updated import mydataloader
from qsm_modules.qsm_dw_models.model_for_dw_QSMnet_lambda_p_trainable import QSMnet
from qsm_modules.qsm_dw_models.WideResnet import WideResNet
from qsm_modules.qsm_dw_models.lmunet_deep_supervision import LightMUNet
from qsm_modules.qsm_dw_models.model_for_dw_normal_cnn_lambda_p_trainable import Dw
from qsm_modules.qsm_data_loader.utils import *
from qsm_modules.qsm_dw_models.model_for_dw_deepqsm_lambda_p_trainable import DeepQSM

#%%
parser = argparse.ArgumentParser(description='Neumann-QSM')
parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
parser.add_argument('--pretrained_path', default='', type=str, help='gpu index (start)')
parser.add_argument('--model', default='WideResNet', type=str, help='model')
args = parser.parse_args()


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
matrix_size = [176,176,160]
voxel_size = [1,  1,  1]

device_id=1


#for restoring weights
restore=False



K_unrolling=1
batch_size=1
No_samples=16800
epoch=0
epsilon=1e-5
learning_rate=275e-6
break_amount=1000
is_data_normalized=True
model='WideResNet'
#model='LightMUNet'
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
savedModels/Neumann_QSM_MODELS_dw_QSMnet/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/Apr_06_01_06_pm_model_K_4_Single_patient_dw_WideResNet/WideResNet_model_best_.pth
    if(Training_patient_no==4):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_4/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'



import os
#print(os.listdir(raw_data_path))
#print(os.listdir(data_path))
print('csv_path:',csv_path)

print("data_path:",data_path)
"""#%%
# making directory for sving models
print ('*******************************************************')
data_source = "Full_patient"
start_time=time.time()
experiments_folder="savedModels/Neumann_QSM_MODELS_dw_QSMnet/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/"

experiment_name=datetime.now().strftime("%b_%d_%I_%M_%P_")+"model_K_"+ str(K_unrolling)+"_"+data_source+'_dw_'+model;
cwd=os.getcwd()

directory=experiments_folder+"/"+experiment_name+"/"
print(directory)
print('Model will be saved to  :', directory)

#%%

# test_data = qsmdata(f'test.csv', f'Testing/phs' ,training=False)
# testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle = False, num_workers=2)
test_csv_path = "/home/guest/Documents/Siraj TM/Major Project/DeepQSM/Data Fold 1/Testing Data/phs/test1.csv"
#test_csv_path = 'test_lim_.csv'
#test_data_path = "Testing"
test_data_path = '/home/guest/Documents/Siraj TM/Major Project/DeepQSM/Data Fold 1/Testing Data'
#test_data_path = '/home/guest/Documents/Siraj TM/Major Project/DeepQSM/Data Fold 1/Limited Data/Testing 3'

testdata = mydataloader(test_csv_path, test_data_path,training=False)

testloader = DataLoader(
    testdata,
    batch_size=1,
    shuffle=False,
    num_workers=1
)


#%%

if(not is_data_normalized):
    sus_mean=0
    sus_std=1
    print('\n\n data is not normalized..................\n\n ')

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
elif(model=='WideResNet'):
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

#Loading the pretrained_model
dw.load_state_dict(torch.load(args.pretrained_path))

# # Load model weights directly from the path above
# dw.load_state_dict(
#     torch.load(pretrained_path, map_location=f'cuda:{device_id}')
# )

# print("Loaded model:", model)
# print("Loaded weights from:", pretrained_path)

print("p",dw.p)
print("lambda_val",dw.Eta_val)
#%%
#%%
if restore:
    restore_weights_path=''
    dw.load_state_dict(torch.load(restore_weights_path))
    print("we have restored weights of the model:",restore_weights_path)
    logging.warning('restore_weights_path:'+(restore_weights_path))
    print("p",dw.p)
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
#%%
ss = sobel_kernel()
ss=ss.float()
ss = ss.cuda(device_id)
# print(ss.shape)

#%%
optimizer = optim.Adam(dw.parameters(), lr=learning_rate)
#%%
loss_Train=[]
loss_Val=[]
lambda_list=[]
p_list=[]
#%%

output_dir = "/home/guest/Documents/Siraj TM/Major Project/DeepQSM/Data Fold 1/Testing Data/phs/susc_pred_neumann"
os.makedirs(output_dir, exist_ok=True)
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

# Folder to save plots
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

#%%
dw.eval()
# validation code is here
#running_val_Loss=0
inference_times = []
with torch.no_grad():
    for i, data in tqdm.tqdm(enumerate(testloader)):
        #try:
        #phs, N_dif, p_id = data
        phs, msk, sus, file_name = data
        dk_repeat = dk.repeat(batch_size,1,1,1,1)

        phs=phs.cuda(device_id)
        msk = msk.cuda(device_id)
        sus = sus.cuda(device_id)
        #sus=sus.cuda(device_id)
        
        # if(i==break_amount):
        #     break;

        # print('\n phs.shape,msk.shape,sus.shape',phs.shape,msk.shape,sus.shape)
        
        # initialization ....
        # taking x_0 = A^Hb
        #print(dk_repeat.shape)
        #print("dk_repeat.shape =", dk_repeat.shape)
        #print("phs.shape       =", phs.shape)
        
        torch.cuda.synchronize(device_id)
        start_inference = time.perf_counter()
        
        with autocast(device_type='cuda'):
        
            B_0_complex = torch.fft.ifftn(dk_repeat*torch.fft.fftn(phs,dim=[2,3,4]),dim=[2,3,4])
            #print(B_0_complex.shape)
            B_0_complex=dw.Eta_val*(B_0_complex)
            B_0_complex=B_0_complex.cuda(device_id)
            # initialize with zeros....
            B_0_complex=B_0_complex*msk

            B_k_complex_sum=torch.zeros_like(B_0_complex)
            B_k_complex_sum=B_0_complex.clone()
            #print('B_k_complex_sum.shape',B_k_complex_sum.shape)

            B_k_complex=B_0_complex
            B_k_complex=B_k_complex*msk
            #print('B_k_complex.shape',B_k_complex.shape)

            B_k_set=[B_0_complex.clone()]

            for k in range(K_unrolling):
                # print('k:',k)
                # term-1 calculation
                #X_T_X=torch.fft.ifftn(dk_repeat*dk_repeat*torch.fft.ifftn(B_k_complex,dim=[2,3,4]),dim=[2,3,4])
                X_T_X = torch.fft.ifftn(
                                            dk_repeat * dk_repeat *
                                            torch.fft.fftn(
                                                B_k_complex,
                                                dim=[2, 3, 4]
                                            ),
                                            dim=[2, 3, 4]
                                        )
                
                
                #print(X_T_X.shape)
                
                
                term_1_complex=B_k_complex- dw.Eta_val*X_T_X
                #print('term_1_complex.shape',term_1_complex.shape)
                # term-2 calculation adding regularization
                B_k_real=z_complex_to_z_real(B_k_complex)

                B_k_real=(B_k_real-sus_mean)/sus_std;
                R_X_real_term=dw(B_k_real)
                # print("Input dtype :", B_k_real.dtype)
                # print("Output dtype:", R_X_real_term.dtype)
                R_X_real_term=R_X_real_term*sus_std+sus_mean

                R_X_complex_term=z_real_to_z_complex(R_X_real_term)
                
                #print(R_X_complex_term.shape)

                #updated B_k_complex
                B_k_complex=term_1_complex-dw.Eta_val*R_X_complex_term
                B_k_complex=B_k_complex*msk
                
                B_k_set.append(B_k_complex.clone())
                B_k_complex_sum=B_k_complex_sum+B_k_complex
            #print(B_k_complex_sum.shape)
            x_k_complex = B_k_complex_sum * msk
        
        # Stop inference timer
        torch.cuda.synchronize(device_id)
        end_inference = time.perf_counter()

        inference_time = end_inference - start_inference
        inference_times.append(inference_time)

        print(f"Case {i+1}: Inference time = {inference_time:.6f} seconds")
        #print(x_k_complex.real.shape)
        N_dif = torch.tensor([[0,0,0]])
        susc_pred = crop_data(x_k_complex.real.detach().cpu(), N_dif[0])
        
        #susc_pred = x_k_complex.real.detach().cpu()
        
        ##additional

        # # Convert tensors to numpy
        # phase = phs[0, 0].detach().cpu().numpy()
        # cosmos = sus[0, 0].detach().cpu().numpy()
        # pred = susc_pred.numpy()

        # # Take middle slice (change axis if needed)
        # slice_idx = phase.shape[2] // 2

        # phase_slice = phase[:, :, slice_idx].T
        # cosmos_slice = cosmos[:, :, slice_idx].T
        # pred_slice = pred[:, :, slice_idx].T

        # # Phase scaling
        # vmin_phase = np.percentile(phase_slice, 2)
        # vmax_phase = np.percentile(phase_slice, 98)

        # # COSMOS + Prediction scaling
        # vmin = -0.1
        # vmax = 0.1

        # # Error scaling (separate)
        # # error_slice = np.abs(pred_slice - cosmos_slice)
        # # error_slice = 3 * error_slice   # mild amplification (NOT too much)
        # # vmin_err = 0
        # # vmax_err = 0.1

        # # Plot
        # # fig, axes = plt.subplots(1, 4, figsize=(8, 20))

        # # titles = ["Phase", "COSMOS", "Neumann QSM", "Error Map"]
        # # images = [phase_slice, cosmos_slice, pred_slice, error_slice]

        # # for i in range(4):
        # #     ax = axes[i]

        # #     if i == 0:  # Phase
        # #         im = ax.imshow(phase_slice, cmap='gray',
        # #                 vmin=vmin_phase, vmax=vmax_phase,
        # #                 interpolation='none')

        # #     elif i == 1:  # COSMOS
        # #         im = ax.imshow(cosmos_slice, cmap='gray',
        # #                 vmin=vmin, vmax=vmax,
        # #                 interpolation='none')

        # #     elif i == 2:  # Prediction
        # #         im = ax.imshow(pred_slice, cmap='gray',
        # #                 vmin=vmin, vmax=vmax,
        # #                 interpolation='none')

        # #     else:  # Error map
        # #         #ax.imshow(error_slice, cmap='gray', interpolation='none')
        # #         im = ax.imshow(error_slice, cmap='gray',
        # #                 vmin=vmin_err, vmax=vmax_err)
                

        # #     ax.set_title(titles[i])
        # #     ax.axis('off')
            
        # #     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # #     cbar.ax.tick_params(labelsize=8)

        # # # plt.tight_layout()
        # # # plt.savefig(f"result_k{k}.png", dpi=600, bbox_inches='tight')
        # # # plt.show()
        
        
        # # plt.tight_layout()

        # # # Save one image per test volume
        # # plot_name = os.path.join(
        # #     plot_dir,
        # #     f"result_{i:03d}_{file_name.item()}.png"
        # #     )

        # # plt.savefig(plot_name, dpi=600, bbox_inches='tight')
        # # plt.close()
        
        # # Plot ONLY axial error slice
        # # ---------------------------------------------------------
        # # Axial difference
        # # ---------------------------------------------------------
        # # ---------------------------------------------------------
        # # ---------------------------------------------------------
        # # Axial slices
        # # ---------------------------------------------------------
        # pred_slice = pred[:, :, pred.shape[2] // 2].T
        # cosmos_slice = cosmos[:, :, cosmos.shape[2] // 2].T

        # # ---------------------------------------------------------
        # # Error
        # # ---------------------------------------------------------
        # diff = pred - cosmos
        # error = 2*np.abs(diff)

        # error_slice = error[:, :, error.shape[2] // 2].T

        # # ---------------------------------------------------------
        # # Frequency spectrum of PREDICTION
        # # ---------------------------------------------------------
        # F_pred = np.fft.fftn(pred)
        # F_pred = np.fft.fftshift(F_pred)

        # frequency_pred = np.log1p(np.abs(F_pred))

        # z = frequency_pred.shape[2] // 2
        # frequency_pred_slice = frequency_pred[:, :, z].T


        # # ---------------------------------------------------------
        # # ---------------------------------------------------------
        # # Frequency spectrum of ERROR
        # # ---------------------------------------------------------

        # # Signed reconstruction error
        # error_signed = pred - cosmos

        # # 3D FFT of signed error
        # F_error = np.fft.fftn(error_signed)

        # # Shift zero frequency to the center
        # F_error = np.fft.fftshift(F_error)

        # # Magnitude spectrum with log compression
        # frequency_error = np.log1p(np.abs(F_error))

        # # Middle axial frequency slice
        # z_error = frequency_error.shape[2] // 2
        # frequency_error_slice = frequency_error[:, :, z_error].T


        # # ---------------------------------------------------------
        # # Plot: Prediction / Error / Frequency of Prediction
        # # ---------------------------------------------------------
        # fig, axes = plt.subplots(
        #     4, 1,
        #     figsize=(6, 20)
        # )

        # # ---------------------------------------------------------
        # # 1. Prediction
        # # ---------------------------------------------------------
        # im1 = axes[0].imshow(
        #     pred_slice,
        #     cmap='gray',
        #     vmin=-0.1,
        #     vmax=0.1,
        #     interpolation='none'
        # )

        # axes[0].set_title("Neumann QSM Prediction")
        # axes[0].axis('off')

        # plt.colorbar(
        #     im1,
        #     ax=axes[0],
        #     fraction=0.046,
        #     pad=0.04
        # )


        # # ---------------------------------------------------------
        # # 2. Error
        # # ---------------------------------------------------------
        # im2 = axes[1].imshow(
        #     error_slice,
        #     cmap='gray',
        #     vmin=0,
        #     vmax=0.05,
        #     interpolation='none'
        # )

        # axes[1].set_title("Axial Error Map")
        # axes[1].axis('off')

        # plt.colorbar(
        #     im2,
        #     ax=axes[1],
        #     fraction=0.046,
        #     pad=0.04
        # )
        
        # freq_vmin = min(
        #     np.percentile(frequency_pred_slice, 1),
        #     np.percentile(frequency_error_slice, 1)
        # )

        # freq_vmax = max(
        #     np.percentile(frequency_pred_slice, 99),
        #     np.percentile(frequency_error_slice, 99)
        # )


        # # ---------------------------------------------------------
        # # 3. Frequency of Prediction
        # # ---------------------------------------------------------
        # im3 = axes[2].imshow(
        #     frequency_pred_slice,
        #     cmap='gray',
        #     vmin=freq_vmin,
        #     vmax=freq_vmax,
        #     interpolation='none'
        # )
        # axes[2].set_title("Frequency Spectrum of Prediction")
        # axes[2].axis('off')

        # plt.colorbar(
        #     im3,
        #     ax=axes[2],
        #     fraction=0.046,
        #     pad=0.04
        # )
        
        # # ---------------------------------------------------------
        # # 4. Frequency of Error
        # # ---------------------------------------------------------

        # # Use robust contrast enhancement
        # vmin_freq_error = np.percentile(frequency_error_slice, 1)
        # vmax_freq_error = np.percentile(frequency_error_slice, 99)

        # im4 = axes[3].imshow(
        #     frequency_error_slice,
        #     cmap='gray',
        #     vmin=freq_vmin,
        #     vmax=freq_vmax,
        #     interpolation='none'
        # )

        # axes[3].set_title("Frequency Spectrum of Error")
        # axes[3].axis('off')

        # plt.colorbar(
        #     im4,
        #     ax=axes[3],
        #     fraction=0.046,
        #     pad=0.04
        # )


        # plt.tight_layout()

        # plot_name = os.path.join(
        #     plot_dir,
        #     f"error_{file_name.item()}_{k}.png"
        # )

        # plt.savefig(
        #     plot_name,
        #     dpi=1800,
        #     bbox_inches='tight'
        # )

        # plt.close()
        #print(f"Saved: {plot_name}")
        
        ########
        
        #print(f"mean: {susc_pred.mean()}, max: {susc_pred.max()}, min: {susc_pred.min()}")
        #print(f"mean: {sus.mean()}, max: {sus.max()}, min: {sus.min()}")
        mdic = {'suscpred':susc_pred}
        #filename =  f"../../Siraj TM/Major Project/DeepQSM/Data/Testing Data/susc_pred_neumann/net-{str(file_name.item())}.mat"
        filename =  f"/home/guest/Documents/Siraj TM/Major Project/DeepQSM/Data Fold 1/Testing Data/phs/susc_pred_neumann/net-{str(file_name.item())}.mat"
        scipy.io.savemat(filename, mdic)
        #loss=total_loss_l1(chi=x_k_complex.real, y=sus, b=phs, d=dk, m=msk, sobel=ss)'''
        #running_val_Loss += loss.item()

        #except Exception as e: 
        #    print(e)
        #    print('error at',i)
#%%
print('\n-------------------------------------------------------------------------------------------------\n')
if len(inference_times) > 0:
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)

    print("\n==========================================")
    print("INFERENCE TIME")
    print("==========================================")
    print(f"Number of cases : {len(inference_times)}")
    print(f"Mean time       : {mean_time:.6f} seconds")
    print(f"Std deviation   : {std_time:.6f} seconds")
    print(f"Mean time       : {mean_time*1000:.3f} ms")
    print(f"Throughput      : {1/mean_time:.3f} cases/sec")
    print("==========================================")