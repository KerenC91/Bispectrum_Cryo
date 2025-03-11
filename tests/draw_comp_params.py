#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:36:09 2025

@author: kerencohen2
"""

import numpy as np
import torch


class DictParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Example usage:
draw_comp_args = DictParams(
    N = 24,
    K = 2,
    maxout = False,
    pow_2_channels = False,
    n_heads = 1,
    model = 3,
    pre_residuals = 11,
    up_residuals = 3,
    post_residuals = 14,
    last_ch = 256,
    channels = [32, 8],
    pre_conv_channels = [8, 32, 256], 
    reduce_height = [4, 3, 3], 
    window_size = 6, 
    img_size = 48,                        
    patch_size = 1,                   
    depths = [6, 6],                      
    num_heads = [2, 2], 
    qkv_bias = False,
    qk_scale = False,
    drop = 0.,
    attn_drop = 0.,
    drop_path_rate = 0.1,
    norm_layer = True,
    downsample = False,
    resi_connection = '1conv',
    from_pretrained = False,
    )

draw_comp_params = DictParams(
    baseline_data_folder = f'baseline_K_{draw_comp_args.K}_N_{draw_comp_args.N}',
    test_folder = 'K2_N24_win6_bs100_ep3000_tr5000_val100_lr_1.0e-03_AdamW_OneCycleLR_average_draw',
    check_k1_k2_distance = True,
    mode = ['opt', 'none'],
    data_size=100,
    data_type = 'normal_distribution',
    normalize=False,
    f1 = 1.,  #loss_sc
    f2 = 0., #loss_l1_aligned
    f3 = 0., #loss_l1_mse
    read_baseline = True,
    use_transformers = True,
    )