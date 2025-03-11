import numpy as np


class HParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

hparams = HParams(
    #####################################
    # debug parameters
    #####################################
    debug_model = 3,
    debug_N = 24,
    debug_last_ch = 128,# 8 for 1, 2,
    debug_pre_conv_channels = [8, 32, 256], # [8, 32, 64, debug_last_ch]
    debug_pre_residuals = 11,
    debug_up_residuals = 3,
    debug_post_residuals = 14,
    debug_n_heads = 1,
    debug_mode = ['opt', None],
    debug_batch_size = 1,
    debug_loss_mode = 'all',
    debug_train_data_size = 1,
    debug_val_data_size = 1,
    debug_epochs = 10000,
    debug_channels_model1 = [256, 8],
    debug_channels_model2 = [256, 64],
    debug_channels_model3 = [256, 8], # [256, debug_last_ch]
    debug_scheduler = "None",
    debug_read_baseline = 0,
    debug_comp_test_name_m = 'baseline_K_2_N_100',
    debug_K = 2,
    debug_loss_method = "sum", #{"average", "sum"}
    
    #####################################
    # data config 
    #####################################
    sigma = 0.1,
    ##########################
    # scheduler config
    ##########################
    # Manual:
    manual_lr_f = 0.1,    
    manual_epochs_lr_change = [2000, 3000, 4000, 5000, 6000],
        
    # ReduceLROnPlateau 
    reduce_lr_mode='min',
    reduce_lr_factor = 0.5,
    reduce_lr_threshold = 1e-4,
    reduce_lr_patience = 5,
    reduce_lr_cooldown = 0,

    # StepLR - every step_size epochs decrease by lr gamma factor
    step_lr_step_size = 2000,#1000, 
    step_lr_gamma = 0.94,
    
    # OneCycleLR - perform one cycle of learning. 
    # epochs and steps per epochs are defined in the code
    # cyc_lr_max_lr = 1e-2,
    cyc_lr_pct_start = 0.562,
    cyc_lr_anneal_strategy = 'cos',
    cyc_lr_three_pahse= True,
    cyc_lr_div_factor = 16,#25
	#"cyc_lr_epochs": num_epochs,
	#"cyc_lr_steps_per_epoch": len(train_loader),
    
    # CosineAnnealingLR - used as:
    # cos_ann_lr_T_max = int(num_epochs * len(train_loader) * cos_ann_lr_T_max_f)
    # performs (cos_ann_lr_T_max_f / 2) cosine periods
    cos_ann_lr_T_max_f = 0.1,
    
    # CyclicLR
    # cyclic_lr_step_size_up = int(num_epochs * len(train_loader) / 2 / cyclic_lr_step_size_up_f)
    # Performs cyclic_lr_step_size_up_f traingle periods
    cyclic_lr_base_lr=1e-4, 
    # cyclic_lr_max_lr=1e-2,
    cyclic_lr_mode="triangular",
    cyclic_lr_step_size_up_f=3,
    cyclic_lr_gamma=1,
    
    ##########################
    # optimizer
    ##########################
    # RMSProp
    opt_rms_prop_alpha = 0.99,
    # SGD
    opt_sgd_momentum = 0.9,
    opt_sgd_weight_decay = 1e-4,
    # AdamW
    opt_adam_w_betas=(0.9, 0.999),
    opt_adam_w_weight_decay=1e-2,
    opt_adam_w_eps = 1e-8,
    # Adam
    opt_adam_betas=(0.9, 0.999),
    opt_adam_eps = 1e-8,
    opt_adam_weight_decay=0.0,
    
    # all optimizer params
    opt_eps = 9.606529741408894e-07,

    ##########################
    # CNN params
    ##########################
    last_ch = 256, # last ch of pre conv. for all models: 8, for model3: 256
    dilation_mid = 1,
    #channels = [256, 256], # for model1: [256, 8], for model2: [256, 64]  
                        # layer_channels list of values on each of heads
    channels_model1 = [256, 8],
    channels_model2 = [256, 64],
    channels_model3 = [32,8],#[256, 8],
    pre_conv_channels = [8, 32],#[8, 32, 256], 
                        #layer_channels list of values on each of heads
    reduce_height = [4, 3, 3], # RELEVANT FOR MODEL2, 3 ONLY
                    #relevant only for model2 - [count kernel stride]
                    #for reducing height in tensor: BXCXHXW to BXCX1XW
    pre_residuals = 9,#11, 
    up_residuals = 8,#3,    
    post_residuals = 2,#14,
    activation = 'LeakyReLU',
    ##########################
    # additional params
    ##########################
    early_stopping = 100,
    dbg_draw_rate=100,
    loss_lim = 1e-6,
    # comparison with baseline
    data_root = '../data',
    #Additional params
    norm_bs = False,
)

