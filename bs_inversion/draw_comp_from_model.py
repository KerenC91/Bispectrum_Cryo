#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:46:07 2024

@author: kerencohen2
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from utils import clculate_bispectrum_efficient, align_to_reference
from train_main import get_model, read_org
from compare_to_baseline import read_tensor_from_matlab
from hparams import hparams

# Parse args
parser = argparse.ArgumentParser(description='Inverting the bispectrum. Pulse dataset')

parser.add_argument('--N', type=int, default=20, metavar='N',
        help='size of vector in the dataset')
parser.add_argument('--K', type=int, default=1, metavar='N',
        help='Number of signals to reconstruct from')
parser.add_argument('--maxout', action='store_true', 
                    help='True for maxout in middle layer, False for conv1 (default)')
parser.add_argument('--pow_2_channels', action='store_true', 
                    help='True for power of 2 channels, '
                    'False for 1 layer with output channel of 8 (default)')
parser.add_argument('--n_heads', type=int, default=1, 
                help='number of cnn heads')
parser.add_argument('--model', type=int, default=3,  
                    help='1 for CNNBS1 - reshape size to reduce dimension'
                    ' 2 for CNNBS2 - strided convolution to reduce dimension')
args = parser.parse_args()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set args
baseline_data_folder = 'baseline_K_1_N_20'
model_folder = 'cnn_baseline_compt_N20_bs_100_ep15000_tr_d_sz5000_val_d_sz100_model3_[\'rand\', \'none\']_n_heads1_loss_all_lr_0.0869_dynamic_lr_OneCycleLR'
N = args.N
K = args.K
# Set baeline data path
baseline_data_path = os.path.join(os.path.join(hparams.data_root, 'baseline_data'),
                                  baseline_data_folder)
# Set model path
model_path = os.path.join(os.path.join(hparams.checkpoints_root, model_folder),
                          'checkpoint_ep14900.pt')
output_path = os.path.join(os.path.join(hparams.data_root, 'tests'),
                                        'offline_comp_compt')

if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists(baseline_data_path):
    print(f'error, baseline_data_path does not exist: {baseline_data_path}')
    exit(1)
    
    
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
    plt.plot(target, label='org')
    plt.plot(output, label='tested')
    if from_matlab is not None:
        plt.plot(from_matlab, label='baseline')
    plt.ylabel('signal')
    plt.xlabel('time')
    plt.legend()
    plt.savefig(fig_path)        
    plt.close()
        
    
# load the model
model = get_model(device, args)
model.load_state_dict(torch.load(model_path))
model.eval()

data_size = len(os.listdir(baseline_data_path))

for i in range(data_size):
    folder_write = os.path.join(output_path, f'sample{i}')
    if not os.path.exists(folder_write):
        os.mkdir(folder_write)
    folder_read = os.path.join(baseline_data_path, f'sample{i}')
    for j in range(K):
        # read
        target = read_org(folder_read, j, K, 'x_true').squeeze(0)
        baseline = read_org(folder_read, j, K, 'x_est').squeeze(0)
        # calculate output
        bs = clculate_bispectrum_efficient(target)
        bs_real = bs.real.float()
        bs_imag = bs.imag.float()
        source = torch.stack([bs_real, bs_imag], dim=0).unsqueeze(0)
        # pass the baseline samples bispectrum through the model
        output = model(source).squeeze(0).squeeze(0)
        output, _ = align_to_reference(output, target)
        # set folder to write to
        folder_k = os.path.join(folder_write, f'{j+1}')
        if not os.path.exists(folder_k):
            os.mkdir(folder_k)
        # Draw the baseline vs model output
        plot_output_debug2(target.cpu().detach().numpy(), 
                               output.cpu().detach().numpy(),
                               folder_k,
                               baseline.cpu().detach().numpy())