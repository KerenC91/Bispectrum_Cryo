#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:49:44 2024

@author: kerencohen2
"""
import torch
import argparse
from torch.cuda import device_count
import torch.multiprocessing as mp
from config.hparams import hparams
from train_main import train, train_distributed

def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


def main(args):
  replica_count = args.nprocs
  if replica_count > 1:
    if args.batch_size % replica_count != 0:
      raise ValueError(f'Batch size {args.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    args.batch_size = args.batch_size // replica_count
    port = _get_free_port()
    # mp.spawn(train_distributed, args=(args,), nprocs=replica_count)#, join=True)
    mp.spawn(train_distributed, args=(port, args, hparams), nprocs=replica_count)#, join=True)
  else:
    if torch.cuda.is_available():
        print("Running with a single GPU")
    else:
        print("GPU is not available, using CPU")
    train(args, hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inverting the Bispectrum.')
    parser.add_argument('--N', type=int, default=10, metavar='N',
            help='size of vector in the dataset')
    parser.add_argument('--K', type=int, default=1, metavar='N',
            help='Number of signals to reconstruct from')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
            help='batch size')
    parser.add_argument('--epochs', type=int, default=5000, metavar='N',
            help='number of epochs to run')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='f',
            help='learning rate (initial for dynamic lr, otherwise fixed)')  
    parser.add_argument('--optimizer', type=str, default="AdamW",  
                        help='The options are \"Adam\"\, \"SGD\"\, \"RMSprop\"\, \"AdamW\"\n'
                        'Please update relevant parameters in parameters file.') 
    parser.add_argument('--scheduler', type=str, default='None',
            help='\'StepLR\', \'ReduceLROnPlateau\', \'OneCycleLR\','
            ' \'CosineAnnealingLR\', \'CyclicLR\', \'Manual\'. '
            'Update configurtion parametes accordingly. '
            'default: \'None\' - no change in lr') 
    parser.add_argument('--scheduler_from_start', action='store_true', 
                        help='In case of loading from checkpoint, if set, start scheduler from scratch.'
                        ' Else, resume scheduler from checkpoint.') 
    # data
    parser.add_argument('--data_type', type=str, default="normal_distribution", 
                        help='one out of \"normal_distribution\", \"gaussian_pulse\". '
                        'gaussian_pulse does not have baseline data to read from.') 
    parser.add_argument('--mode', type=str, nargs='+', default=['opt', None],
            help= '[mode, add], mode in {\'rand\'\,\'opt\'}, add (optioanl) in {\'shift\', \'circular_shifts\'}'
                '\'rand\': Create random data during training.\n'
                    '\'opt\': Create a fixed dataset'
                    '\'shift\': Randomly shift the signal.\n'
                    '\'circular_shifts\': shift the signal circularly for every bbatch') 
    parser.add_argument('--train_data_size', type=int, default=5000, metavar='N',
            help='the size of the train data') 
    parser.add_argument('--val_data_size', type=int, default=100, metavar='N',
            help='the size of the validate data')  
    # baseline data
    parser.add_argument('--comp_test_name_m', type=str, default='',
            help='baseline data folder results for comparison') 
    parser.add_argument('--read_baseline', type=int, default=0, 
                        help='0: no action, 1: read from matlab to training set'
                        '2: read from matlab to validation set')
    parser.add_argument('--normalize', action='store_true',
                        help='normalizing data for True, else False (default)')
    # wandb
    parser.add_argument('--wandb', action='store_true', help='Log data using wandb') 
    parser.add_argument('--wandb_proj_name', type=str, default='BS_G_inv_multi_gpu', 
                        help='wandb project name')
    parser.add_argument('--wandb_run_id', type=str, default="",
                        help='run id to resume running. If not provided - new run.') 
    # loss
    parser.add_argument('--loss_mode', type=str, default="l1",  
                        help='\'all\' - l1, mse, rel_mse. default: \'l1\' - l1 loss.'
                        'Note: the training loss is always l1') 
    parser.add_argument('--loss_method', type=str, default="average",  
                        help='one of \'average\', \'sum\'.'
                        'Note: the training loss is always l1') 
    parser.add_argument('--loss_criterion', type=str, default="l1", 
                        help='one out of \"l1\", \"mse\", \"sc\".') 
    
    parser.add_argument('--clip_grad_norm', type=float, default=0.,  
                        help='If greater than 0: clip gradients norm with the clip_grad_norm value.') 
    parser.add_argument('--run_mode', type=str, default="new", 
                        help='one out of \"override\", \"resume\", \"new\" existing run '
                        'eventhough a checkpoint exists') 
    # model 
    parser.add_argument('--n_heads', type=int, default=1, 
                    help='number of cnn heads')
    parser.add_argument('--model', type=int, default=3,  
                        help='1 for CNNBS1 - reshape size to reduce dimension'
                        ' 2 for CNNBS2 - strided convolution to reduce dimension')
    parser.add_argument('--pre_residuals', type=int, default=9, 
                        help='pre residuals layers count')
    parser.add_argument('--up_residuals', type=int, default=8, 
                        help='up residuals layers count')
    parser.add_argument('--post_residuals', type=int, default=2, 
                        help='post residuals layers count')
    parser.add_argument('--last_ch', type=int, default=256, 
                        help='last_ch')
    parser.add_argument('--channels', type=int, nargs='+', 
                        default=[256, 8], 
                        help='layer_channels list of values on each of heads. '
                        'The default fits model3')
    parser.add_argument('--pre_conv_channels', type=int, nargs='+', 
                        default=[8, 32, 256], 
                        help='layer_channels list of values on each of heads')
    parser.add_argument('--reduce_height', type=int, nargs='+', default=[4, 3, 3], 
                        help='relevant only for model2 - [count kernel stride] ' 
                        'for reducing height in tensor: BXCXHXW to BXCX1XW')
    # 
    parser.add_argument('--maxout', action='store_true', 
                        help='True for maxout in middle layer, False for conv1 (default)')
    parser.add_argument('--pow_2_channels', action='store_true', 
                        help='True for power of 2 channels, '
                        'False for 1 layer with output channel of 8 (default)')
    # swin transformer
    parser.add_argument('--window_size', type=int, default=8, 
                        help='window_size')    
    parser.add_argument('--img_size', type=int, default=48, 
                        help='window_size')  
    parser.add_argument('--patch_size', type=int, default=1, 
                        help='patch size used in training SwinIR. '
                            'Just used to differentiate two different settings in Table 2 of the paper. '
                            'Images are NOT tested patch by patch.')    
    parser.add_argument('--depths', type=int, nargs='+', 
                        default=[6, 6], 
                        help='depths')    
    parser.add_argument('--num_heads', type=int, nargs='+', 
                        default=[2, 2], 
                        help='num_heads')      
    parser.add_argument('--qkv_bias', action='store_true', 
                        help='') 
    parser.add_argument('--qk_scale', action='store_true', 
                        help='')     
    parser.add_argument('--drop', type=float, default=0.,
                        help='drop')
    parser.add_argument('--attn_drop', type=float, default=0.,
                        help='attn_drop')
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help='drop_path_rate')
    parser.add_argument('--norm_layer',  action='store_false',
                        help='norm_layer')
    parser.add_argument('--downsample', action='store_true', 
                        help='downsample')
    parser.add_argument('--resi_connection', type=str, default='1conv',
                        help='resi_connection')
    #
    parser.add_argument('--comp_test_name', type=str, default='test',
            help='folder test name to save results into') 
    parser.add_argument('--save_every', type=int, default=100, metavar='N',
            help='save checkpoint every <save_every> epoch')
    parser.add_argument('--early_stopping', action='store_true', 
                        help='early stopping after early_stopping times. '
                        'Update early_stopping in configuration') 
    parser.add_argument('--plotting_off', action='store_true', 
                        help='If set, do not plot data samples at the end. Can draw '
                        'offline using saved checkpoint and initial samples.') 
    parser.add_argument('--suffix', type=str, default='',
            help='suffix to add to the name of the cnn yml file') 
    # debug
    parser.add_argument('--debug', action='store_true', 
                        help='debugging mode. Use debug params from hparams.')
    parser.add_argument('--log_level', type=int, default=0, 
                        help='0: info, 1: warning, '
                        '2: debug, 3: detailed debug')
    # distributed training
    parser.add_argument('--nprocs', default=torch.cuda.device_count(), type=int, 
                        help='nprocs, default is the number of available gpus on the machine')
    args = parser.parse_args()
    main(args)
