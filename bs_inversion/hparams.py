#import tensorflow as tf
import numpy as np
import torch


class HParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Example usage:
hparams = HParams(
    #####################################
    dbg_draw_rate=100,
    loss_lim = 1e-6,
    #####################################
    # Model parameters
    #####################################
#     n_heads = 1,
#     pre_residuals = 10,
#     up_residuals=0,
#     post_residuals = 3,
#     pre_conv_channels = [2,2,4],
#     layer_channels = [10 * 4 * 2, 52, 26, 13, 6],
# #[100 * 4 * 2, 256 * 2, 128 * 2, 64 * 2, 32 * 2, 16* 2, 8 * 2]

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
    fixed_sample=torch.tensor([-4.1425e-01,  1.3600e+00, -6.8926e-01, -4.1558e-01, -1.5053e+00,
         3.0317e-01, -1.6260e-03, -1.3670e+00, -1.8949e-01, -7.8125e-01,
          4.1742e-01,  6.0319e-01, -1.3874e+00, -2.3990e+00,  1.0808e+00,
          4.9770e-01, -8.4575e-01,  3.9753e-01, -7.8966e-01,  1.4190e+00,
        -5.6052e-01, -2.5053e-01, -2.3574e+00,  2.2591e-01,  3.5794e-01,
          1.1672e+00,  2.2979e-01,  5.9654e-01,  1.8660e+00, -2.5858e-01,
          1.8843e+00,  1.7055e+00,  4.8482e-01, -4.7975e-01,  4.9895e-01,
          8.2748e-01, -1.0155e+00, -6.9956e-01,  7.7198e-01,  3.3064e-01,
          1.2681e+00, -1.9337e-01,  1.2024e+00,  7.4160e-02, -1.2637e+00,
        -2.0865e+00, -1.0185e-01, -2.1991e-01, -6.9103e-02,  1.0575e+00,
          1.7725e+00, -3.5686e-01,  1.1313e+00,  1.4250e+00,  1.1318e+00,
          8.0695e-01,  7.2106e-01, -8.9464e-01,  1.2015e+00,  2.8017e-01,
        -1.4855e+00,  9.0220e-01, -6.1121e-02,  1.0083e-01, -8.9616e-02,
        -7.2925e-01,  1.0240e-01,  9.0606e-01, -3.0918e-01, -3.9682e-01,
        -1.8181e+00, -8.4176e-01, -1.1784e+00, -8.8659e-01,  1.9762e-01,
          8.5462e-03,  7.1479e-01,  2.1523e-01, -4.6860e-01, -2.3042e-01,
          1.2255e+00,  9.1218e-01, -1.4788e-01, -1.0748e+00, -2.3714e+00,
        -1.7732e-01, -2.9170e-01, -3.4579e-01, -1.1051e+00,  2.8760e-01,
          1.5630e-02,  2.1314e+00,  3.6073e-02,  5.6559e-01, -8.1566e-01,
          4.5816e-01,  1.8699e+00, -7.0245e-03,  5.0816e-01,  1.7056e-01
        ])
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)