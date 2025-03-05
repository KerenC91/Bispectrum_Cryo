#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:46:07 2024

@author: kerencohen2
"""

import os


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import sys

sys.path.append(parent_dir)
sys.path.append(f'{parent_dir}/config')

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from utils.utils import clculate_bispectrum_efficient, align_to_reference, BatchAligneToReference, BispectrumCalculator
from train_main import get_model, read_org, read_dataset_from_baseline, create_dataset, load_model_safely
from utils.compare_to_baseline import read_tensor_from_matlab
from hparams import hparams
from torch.utils.data import Dataset, DataLoader
import numpy as np
from draw_comp_params import draw_comp_params, draw_comp_args
# torch.manual_seed(234)


# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Init args
args = draw_comp_args
params = draw_comp_params
N = args.N
K = args.K
if params.mode[0] == 'opt':
    params.read_baseline = False

# Set baeline data path
baseline_data_path = os.path.join('../data', params.baseline_data_folder)
# Set model path
model_path = os.path.join(os.path.join('../output', params.test_folder),'ckp.pt')
# Set output folder path
output_path = os.path.join('../output', params.test_folder)

if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists(baseline_data_path):
    print(f'error, baseline_data_path does not exist: {baseline_data_path}')
    exit(1)

# Init helpers
bs_calc = BispectrumCalculator(K, N, device).to(device)
aligner = BatchAligneToReference(device).to(device)
    
# Init functions
def read_org(folder, k, K, label='x_true'):
    if K > 1:
        sample_path = os.path.join(folder, f'{label}_{k+1}.csv')
        target = read_tensor_from_matlab(sample_path, True)  
    else:
        sample_path = os.path.join(folder, f'{label}.csv')
        target = read_tensor_from_matlab(sample_path, True)    
    return target  

       
def plot_output_debug2(target, output, folder, from_matlab=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig_path = f'{folder}/x_vs_x_rec.png'     
  
    plt.figure()
    plt.title('Comparison between original signal and its reconstructions')
    plt.plot(output, label='tested', color='tab:orange')
    # if from_matlab is not None:
    #     plt.plot(from_matlab, label='baseline', color='tab:green')
    plt.plot(target, label='org', color='tab:blue')
    plt.ylabel('signal')
    plt.xlabel('time')
    plt.legend()
    plt.savefig(fig_path)        
    plt.close()
         
def switch_criterion(bs_pred, bs_gt):
    sh = bs_pred.shape
    reversed_bs_pred = torch.flip(bs_pred, dims=(1,))
    loss1 = torch.mean(
                torch.norm((bs_pred - bs_gt).view(sh[0], sh[1], -1), dim=(0, 2)) / \
                    torch.norm(bs_gt.view(sh[0], sh[1], -1), dim=(0, 2)))
    loss2 = torch.mean(
                torch.norm((reversed_bs_pred - bs_gt).view(sh[0], sh[1], -1), dim=(0, 2)) / \
                    torch.norm(bs_gt.view(sh[0], sh[1], -1), dim=(0, 2)))
    # Get the index for the minimal loss
    i = np.argmin(np.array([loss1.item(), loss2.item()]))
    # Get the minimal loss
    loss = torch.min(loss1, loss2)
    switch = (i != 0)
    
    return loss, switch  

def switch_criterion_l1_aligned(pred, target):
    criterion = torch.nn.L1Loss()  
    # for sum method only
    sh = pred.shape
    reversed_pred = torch.flip(pred, dims=(1,))
    
    pred, _ = aligner(pred, target)
    loss1 = criterion(pred, target)
    reversed_pred, _ = aligner(reversed_pred, target)
    loss2 = criterion(reversed_pred, target)
    # get the index for the minimal loss
    i = np.argmin(np.array([loss1.item(), loss2.item()]))
    # get the minimal loss
    loss = torch.min(loss1, loss2)
    switch = (i != 0)
    
    return loss, switch

def switch_criterion_mse_aligned(pred, target):
    criterion = torch.nn.MSELoss()  
    # for sum method only
    sh = pred.shape
    reversed_pred = torch.flip(pred, dims=(1,))
    
    pred, _ = aligner(pred, target)
    loss1 = criterion(pred, target)
    reversed_pred, _ = aligner(reversed_pred, target)
    loss2 = criterion(reversed_pred, target)
    # get the index for the minimal loss
    i = np.argmin(np.array([loss1.item(), loss2.item()]))
    # get the minimal loss
    loss = torch.min(loss1, loss2)
    switch = (i != 0)
    
    return loss, switch

def switch_position(pred, target):
    switch = False
    if params.f1 != 0: #loss_sc
        bs_pred, pred = bs_calc(pred, "sum")
        bs_target, target = bs_calc(target, "sum")
        _, switch = switch_criterion(bs_pred, bs_target)
    if params.f2 != 0: #loss_l1_aligned
        _, switch = switch_criterion_l1_aligned(pred, target)
    if params.f3 != 0: #loss_l1_mse
        _, switch = switch_criterion_mse_aligned(pred, target)
    if switch:
        pred = torch.flip(pred, dims=(-2,))
    
    return pred

# Load the model

model = get_model(device, args, is_distributed=False, use_transformers=params.use_transformers)
# map_location = f"cuda:{device}"
_ = load_model_safely(device, model, model_path, args, device)
model.eval()
model.to(device)

# Create Validation Dataset
dataset = create_dataset(device, 
                         params.data_size, 
                         K, 
                         N, 
                         params.read_baseline, 
                         params.mode, 
                         baseline_data_path, 
                         params.data_type,
                         params.normalize
                         )

dataloader = DataLoader(dataset=dataset,
                        batch_size=1,
                        pin_memory=False,
                        shuffle=False
                        )

# Set baseline data
baseline = read_dataset_from_baseline(baseline_data_path, 
                                      params.data_size, 
                                      K, 
                                      N, 
                                      'x_est')
if params.normalize:
    y = torch.fft.fft(baseline, dim=-1)
    y /= torch.norm(y, dim=-1).unsqueeze(2)
    baseline = torch.fft.ifft(y, dim=-1)
    baseline = baseline.type(torch.float32)
        
# loop over signals
avg_err = 0.
avg_min_err_between_signals = 0.
for idx, (source, target) in dataloader:

    i = idx.item()
    source = source.to(device)
    target = target.to(device)
    
    # Set output folder per sample
    folder_write = os.path.join(output_path, f'sample{i}')
    if not os.path.exists(folder_write):
        os.mkdir(folder_write)
  
    # Pass the sample's bispectrum through the model
    output = model(source)
    
    if K == 2 and params.check_k1_k2_distance:
        # Calculate distance between s0 and s1, to verify the cnn does not learn the same signal
        # Option 1 - no switch
        output, _ = aligner(output, target)
        min_err_between_signals = torch.norm(output.squeeze(0)[0] - output.squeeze(0)[1]) / \
                                                torch.norm(output.squeeze(0)[0])
        # Option 2 - switch
        output = switch_position(output, target)
        output, _ = aligner(output, target)
        
        min_err_between_signals = min(min_err_between_signals, 
                                      torch.norm(output.squeeze(0)[0] - output.squeeze(0)[1]) / \
                                      torch.norm(output.squeeze(0)[0]))
        # print(f'min_err_between_signals={min_err_between_signals}')
        avg_min_err_between_signals += min_err_between_signals

        # Align output 1 to output 0 --> output 1 aligned
        output_1_aligned, _ = align_to_reference(output.squeeze(0)[1], 
                                        output.squeeze(0)[0])
        
        target_0_aligned, _ = align_to_reference(target.squeeze(0)[0], 
                                        output.squeeze(0)[0])
        target_1_aligned, _ = align_to_reference(target.squeeze(0)[1], 
                                        output_1_aligned)
        
        target = torch.stack([target_0_aligned, target_1_aligned], dim=0)
        output = torch.stack([output.squeeze(0)[0], output_1_aligned], dim=0)

        output_avg = (output[0] + output[1]) / 2
        target_avg = (target[0] + target[1]) / 2
        rel_error_X = torch.norm(target - output) / torch.norm(target)
        
        # # fig 1
        # fig_path = os.path.join(folder_write, 'comp_s1_pred_s2_pred_preds_avg.jpg')
        # plt.figure(figsize=(9, 5))
        # plt.title(f'Comparison between s1_pred, s2_pred, pred_avg, sample{i}, rel_mse={rel_error_X:.03f}')
        # plt.plot(output[0].cpu().detach().numpy(), label='s1_pred', color='tab:orange')
        # plt.plot(output[1].cpu().detach().numpy(), label='s2_pred', color='tab:red')
        # plt.plot(output_avg.cpu().detach().numpy(), label='avg_pred', color='tab:blue', linestyle='dashed')
        # plt.plot(target_avg.cpu().detach().numpy(), label='target_avg', color='tab:green')
    
        # plt.ylabel('signal')
        # plt.xlabel('time')
        # plt.legend()
        # plt.savefig(fig_path)        
        # plt.close()
        
        # # fig 2
        # fig_path = os.path.join(folder_write, 'comp_s1_s2_s1_pred_s2_pred.jpg')
        # plt.figure(figsize=(9, 5))
        # plt.title(f'Comparison between s1, s2, s1_pred, s2_pred, sample{i}')
        # plt.plot(target[0].cpu().detach().numpy(), label='s1', color='tab:blue')
        # plt.plot(target[1].cpu().detach().numpy(), label='s2', color='tab:green')
        # plt.plot(output[0].cpu().detach().numpy(), label='s1_pred', color='tab:orange', linestyle='dashed')
        # plt.plot(output[1].cpu().detach().numpy(), label='s2_pred', color='tab:red', linestyle='dashed')
    
        # plt.ylabel('signal')
        # plt.xlabel('time')
        # plt.legend()
        # plt.savefig(fig_path)        
        # plt.close()
        
        # fig 3
        fig_path = os.path.join(folder_write, 'comp_s1_s1_pred.jpg')
        plt.figure(figsize=(9, 5))
        plt.title(f'Comparison between s1, s1_pred, sample{i}')
        plt.plot(target[0].cpu().detach().numpy(), label='s1', color='tab:blue')
        plt.plot(output[0].cpu().detach().numpy(), label='s1_pred', color='tab:orange', linestyle='dashed')
    
        plt.ylabel('signal')
        plt.xlabel('time')
        plt.legend()
        plt.savefig(fig_path)        
        plt.close()
        
        # fig 4
        fig_path = os.path.join(folder_write, 'comp_s2_s2_pred.jpg')
        plt.figure(figsize=(9, 5))
        plt.title(f'Comparison between s2, s2_pred, sample{i}')
        plt.plot(target[1].cpu().detach().numpy(), label='s2', color='tab:green')
        plt.plot(output[1].cpu().detach().numpy(), label='s2_pred', color='tab:red', linestyle='dashed')
    
        plt.ylabel('signal')
        plt.xlabel('time')
        plt.legend()
        plt.savefig(fig_path)        
        plt.close()
    else:
        if K == 1:
            output, _ = aligner(output, target)
        else:
            output = switch_position(output, target)
        # Plot org, baseline and output for comparison
        for j in range(K):
            # set folder to write to
            folder_k = os.path.join(folder_write, f'{j+1}')
            if not os.path.exists(folder_k):
                os.mkdir(folder_k)
            # Draw the baseline vs model output
            plot_output_debug2(target.squeeze(0)[j].cpu().detach().numpy(), 
                                    output.squeeze(0)[j].cpu().detach().numpy(),
                                    folder_k,
                                    baseline[i][j])     
    rel_error_X = torch.norm(target - output) / torch.norm(target)
    rel_error_X_path = os.path.join(folder_write, 'rel_error_X.csv')
    np.savetxt(rel_error_X_path, [rel_error_X.item()])
    print(f'sample{i}, err={rel_error_X}')  
    avg_err += rel_error_X
    print(f'curent avg={(avg_err/(i+1)):.08f}')      

        
avg_err /= params.data_size
print(f'avg err={avg_err:.08f}')   
np.savetxt(f'{output_path}/avg_err.csv', [avg_err.item()])     
if K == 2 and params.check_k1_k2_distance:
    print(f'avg min_err_between_signals={avg_err:.08f}')        
    np.savetxt(f'{output_path}/avg_min_err_between_signals.csv', [avg_err.item()])     

