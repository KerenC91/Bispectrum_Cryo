#import tensorflow as tf
import numpy as np
import torch


class OptParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Example usage:
optuna_params = OptParams(
    #####################################
    # args parameters
    #####################################
    N = 100,
    batch_size = 1,
    train_data_size = 1,
    #epochs = 7000,
    channels_model1 = [256, 8],
    channels_model2 = [256, 64],
    channels_model3 = [256, 8],
    maxout = False,
    pow_2_channels = False,

    #####################################
    # Training config
    #####################################
    # n_workers=2,
    # seed=12345,
    # batch_size=40,
    # lr=1.0 * 1e-5,
    # weight_decay=1e-5,
    # epochs=7000,
    # grad_clip_thresh=5.0,
    # checkpoint_interval=1000,
    
    #####################################
    # loss config 
    #####################################
    ##########################
    # dynamic lr
    ##########################
    lr_f = 0.1, # for all modes
    # Manual:
    epochs_lr_change =  [2000, 5000],#[3000, 5000, 6000],

    ##########################
    # CNN params
    ##########################
    last_ch = 8, # for all models: 8
    dilation_mid = 1,
    channels = [256, 64], # for model1: [256, 8], for model2: [256, 64]  
                        # layer_channels list of values on each of heads
    linear_ch = 8, # for model1: channels[-1], for model2: 8, 
    pre_conv_channels = [8, 32, 64], 
                        #layer_channels list of values on each of heads
    reduce_height = [4, 3, 3], # RELEVANT FOR MODEL2 ONLY
                    #relevant only for model2 - [count kernel stride]
                    #for reducing height in tensor: BXCXHXW to BXCX1XW
                
    ##########################
    # additional params
    ##########################
    early_stopping = 100,
    dbg_draw_rate=100,
    loss_lim = 1e-6,
    # comparison with baseline
    matlab_x_org_file = 'data_from_matlab/sample2/x_true.csv',
    py_x_rec_file = 'data_from_matlab/sample2/py_x_est.csv',
    matlab_x_rec_file = 'data_from_matlab/sample2/x_est.csv',
)

