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
from utils import clculate_bispectrum_efficient, align_to_reference, BatchAligneToReference, BispectrumCalculator
from train_main import get_model, read_org, read_dataset_from_baseline, UnitVecDataset
from compare_to_baseline import read_tensor_from_matlab
from hparams import hparams
from torch.utils.data import Dataset, DataLoader
import numpy as np
# Parse args
parser = argparse.ArgumentParser(description='Inverting the bispectrum. Pulse dataset')

parser.add_argument('--N', type=int, default=20, metavar='N',
        help='size of vector in the dataset')
parser.add_argument('--K', type=int, default=2, metavar='N',
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
baseline_data_folder = f'baseline_K_{args.K}_N_{args.N}'
model_folder = 'test_K_2_offline_ep7800'
test_folder = 'test_K_2_offline_ep7800'
N = args.N
K = args.K
# Set baeline data path
baseline_data_path = os.path.join(os.path.join(hparams.data_root, 'baseline_data'),
                                  baseline_data_folder)
# Set model path
model_path = os.path.join(os.path.join(os.path.join(hparams.data_root, 'tests'),
                          model_folder),
                          'ckp.pt')
# Set output folder path
output_path = os.path.join(os.path.join(hparams.data_root, 'tests'),
                                        test_folder)

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
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.eval()
model.to(device)

data_size = len(os.listdir(baseline_data_path))

# Create Dataset
target = read_dataset_from_baseline(baseline_data_path, data_size, K, N)
target.to(device)
bs_calc = BispectrumCalculator(K, N, device).to(device)
source, target = bs_calc(target)
source=source.to(device)
target.to(device)
dataset = UnitVecDataset(source, target)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=1,
    pin_memory=False,
    shuffle=False
)
baseline = torch.zeros(data_size, K, N)
for i in range(data_size):
    folder = os.path.join(baseline_data_path, f'sample{i}')
    for j in range(K):
        baseline[i][j] = read_org(folder, j, K, f"x_est") 
aligner = BatchAligneToReference(device).to(device)
avg_err = 0
for idx, (source, target) in dataloader:
    i = idx.item()
    folder_write = os.path.join(output_path, f'sample{i}')
    if not os.path.exists(folder_write):
        os.mkdir(folder_write)
    folder_read = os.path.join(baseline_data_path, f'sample{i}')
    # pass the baseline samples bispectrum through the model
    source = source.to(device)
    output = model(source)
    target = target.to(device)
    output, _ = aligner(output, target)
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
print(f'avg err={(avg_err / data_size):.08f}')        

