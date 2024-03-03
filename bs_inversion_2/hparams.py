#import tensorflow as tf
import numpy as np
import torch


class HParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Example usage:
hparams = HParams(
    #####################################
    # debug parameters
    #####################################
    DEBUG = False,
    debug_model = 3,
    debug_N = 100,
    debug_last_ch = 256,# 8 for 1, 2,
    debug_pre_conv_channels = [8, 32, 256], # [8, 32, 64, debug_last_ch]
    debug_pre_residuals = 1,
    debug_up_residuals = 1,
    debug_post_residuals = 1,
    debug_n_heads = 1,
    debug_mode = 'opt',
    debug_batch_size = 1,
    debug_loss_mode = 'all',
    debug_train_data_size = 1,
    debug_val_data_size = 1,
    debug_epochs = 1,
    debug_channels_model1 = [256, 8],
    debug_channels_model2 = [256, 64],
    debug_channels_model3 = [256, 256], # [256, debug_last_ch]


    #####################################
    # Training config
    #####################################
    # n_workers=2,
    # seed=12345,
    # batch_size=40,
    # lr=1.0 * 1e-5,
    # weight_decay=1e-5,
    # epochs=50000,
    # grad_clip_thresh=5.0,
    # checkpoint_interval=1000,
    
    #####################################
    # loss config 
    #####################################
    f1=0.,#_loss_sc
    f2=0.,#_loss_log_sc
    f3=0.,#_loss_freq
    f4=0.,#loss_weighted_phase
    f5=1.,#_loss_l1
    ##########################
    # dynamic lr
    ##########################
    lr_f = 0.1, # for all modes
    # Manual:
    epochs_lr_change = [2000, 3000, 4000, 5000, 6000],
    # ReduceLROnPlateau    
    reduce_lr_threshold = 1e-3,
    reduce_lr_patience = 10,
    reduce_lr_mode='min',
    # StepLR
    step_lr_step_size = 1000,
    step_lr_gamma = 0.01,
    ##########################
    # optimizer
    ##########################
    # RMSProp
    opt_rms_prop_alpha = 0.99,
    opt_rms_prop_eps = 1e-8,
    # SGD
    
    # AdamW
    opt_adam_w_betas=(0.9, 0.999),
    opt_adam_w_eps = 1e-8,
    opt_adam_w_weight_decay=0.01,
    # Adam
    opt_adam_betas=(0.9, 0.999),
    opt_adam_eps = 1e-8,
    opt_adam_weight_decay=0.0,
    ##########################
    # CNN params
    ##########################
    last_ch = 256, # for all models: 8, for model3: 64
    dilation_mid = 1,
    n_heads = 1, # number of cnn heads
    channels = [256, 256], # for model1: [256, 8], for model2: [256, 64]  
                        # layer_channels list of values on each of heads
    linear_ch = 8, # for model1: channels[-1], for model2: 8, 
    pre_conv_channels = [8, 32, 64, 256], 
                        #layer_channels list of values on each of heads
    reduce_height = [4, 3, 3], # RELEVANT FOR MODEL2 ONLY
                    #relevant only for model2 - [count kernel stride]
                    #for reducing height in tensor: BXCXHXW to BXCX1XW
    pre_residuals = 14, 
    up_residuals = 2,    
    post_residuals = 12,

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

hparams2 = HParams(
    #####################################
    dbg_draw_rate=100,
    )

hparams3 = HParams(
    #####################################
    dbg_draw_rate=100,
    )

def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)