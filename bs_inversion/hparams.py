#import tensorflow as tf
import numpy as np

class HParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Example usage:
hparams = HParams(
    #####################################
 

    #####################################
    # Data config
    #####################################
    seg_len= 81 * 256,
    file_list="training_data/files.txt",
    spec_len= 81,

    #####################################
    # Model parameters
    #####################################
    n_heads = 2,
    pre_residuals = 4,
    up_residuals=0,
    post_residuals = 12,
    pre_conv_channels = [2,2,4],
    layer_channels = [100 * 4, 256, 128, 64, 32, 16, 8],


    #####################################
    # Training config
    #####################################
    n_workers=2,
    seed=12345,
    batch_size=40,
    lr=1.0 * 1e-3,
    weight_decay=1e-5,
    epochs=50000,
    grad_clip_thresh=5.0,
    checkpoint_interval=1000,
    
    #####################################
    # loss config
    #####################################
    
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
