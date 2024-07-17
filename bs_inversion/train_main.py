import torch.optim as optim
import time 
import os
import wandb
import torch 
from torch.utils.data import Dataset, DataLoader
import argparse
from utils import BispectrumCalculator, rand_shift_signal
from model1 import CNNBS, HeadBS1
from model2 import HeadBS2
from model3 import HeadBS3
from hparams import hparams
import numpy as np
from trainer import Trainer
import sys
from torch import nn
from compare_to_baseline import read_tensor_from_matlab


#torch.set_printoptions(precision=15)
#torch.set_default_dtype(torch.float64)
# Set the same seed for reproducibility
#torch.manual_seed(1234)


                

class UnitVecDataset(Dataset):
    
    def __init__(self, source, target):
        self.target = target
        self.source = source
        self.data_size = self.__len__()
            
        
    def __len__(self):
        return self.target.size(0)
    
    def __getitem__(self, idx):

        return idx, (self.source[idx], self.target[idx])

def read_noisy(folder):
    # Needs update
    sample_path = os.path.join(folder, 'data.csv')
    target = read_tensor_from_matlab(sample_path, True) 
    shifts = int(np.loadtxt(os.path.join(folder, 'shifts.csv'), delimiter=" "))
    target = torch.roll(target, -shifts)
    
    return target    

def read_org(folder, k, K, label='x_true'):
    if K > 1:
        sample_path = os.path.join(folder, f'{label}_{k+1}.csv')
        target = read_tensor_from_matlab(sample_path, True)  
    else:
        sample_path = os.path.join(folder, f'{label}.csv')
        target = read_tensor_from_matlab(sample_path, True)    
    return target  

def set_read_func(folder_matlab):
    if 'noisy' in folder_matlab:
        f = read_noisy
    else:
        f = read_org
    return f

def read_dataset_from_baseline(folder_matlab, data_size, K, N):
    read_func = set_read_func(folder_matlab)
    data_size = min(data_size, len(os.listdir(folder_matlab)))
    target = torch.zeros(data_size, K, N)

    print(f'The updated data size is {data_size}')

    for i in range(data_size):
        folder = os.path.join(folder_matlab, f'sample{i}')
        for j in range(K):
            target[i][j] = read_func(folder, j, K)   
    
    return target
   
def create_dataset(device, data_size, K, N, read_baseline, mode, 
                   folder_matlab):
    bs_calc = BispectrumCalculator(K, N, device).to(device)
    print(f'read_baseline={read_baseline}, mode={mode}')
    if read_baseline: # in val dataset
        target = read_dataset_from_baseline(folder_matlab, data_size, K, N)
    else:
        if mode[0] == 'opt':
            # Create random dataset
            target = torch.randn(data_size, K, N)
        elif mode[0] == 'rand':
            # Initialize dataset to zeros and create data on the fly 
            target = torch.zeros(data_size, K, N)
    target.to(device)
    source, target = bs_calc(target)
    if mode[0] == 'opt' and mode[1] == 'shift' and not read_baseline:
            target, shifts = rand_shift_signal(target, K, N, data_size)
    dataset = UnitVecDataset(source, target)

    return dataset

def set_activation(activation_name):
    #['ELU', 'LeakyReLU', 'ReLU', 'Softsign', 'Tanh'])
   
    if activation_name == 'ELU':
        activation = nn.ELU()
    elif activation_name == 'ReLU':
        activation = nn.ReLU()
    elif activation_name == 'Softsign':
        activation = nn.Softsign()
    elif activation_name == 'Tanh':
        activation = nn.Tanh()  
    else: #'LeakyReLU':
        activation = nn.LeakyReLU()
        
    return activation

def update_reduce_height_cnt(k, s, Hin):
    """
    Calculate the number of layers for reducing height, based on k, s

    Parameters
    ----------
    k : int
        height kernel size.
    s : int
        height stride size.
    Hin : int
        height of the input signal.
        
    Returns
    -------
    cnt : int
        Number of reduce_height layers to perform.
    k : int
        height kernel size for each layer.
    s : int
        height stride size for each layer.

    """
    if Hin < k:
        print(f'Error! Hin={Hin} is smaller or equal to k={k}')
        sys.exit(1)
    H = Hin
    cnt = 0
    add_conv_2 = False
    while H > 1:
        H = int((H - k) / s) + 1
        cnt += 1 
        if H == 2:
            add_conv_2 = True
            break
    print(f'reduce_height=[{cnt},{k},{s},{add_conv_2}]')
    
    return cnt, k, s, add_conv_2
    
def get_model(device, args):
    if args.model == 2:
        head_class = HeadBS2
        channels = hparams.channels_model2
    elif args.model == 3:
        head_class = HeadBS3 
        channels = hparams.channels_model3
    else:
        head_class = HeadBS1
        channels = hparams.channels_model1
 

    hparams.pre_conv_channels[-1] = hparams.last_ch
    channels[-1] = hparams.last_ch
    cnt, k, s = hparams.reduce_height
    reduce_height = update_reduce_height_cnt(k, s, args.N)
    activation = set_activation(hparams.activation)
    model = CNNBS(
        device=device,
        input_len=args.N,
        signals_count = args.K,
        n_heads=args.n_heads,
        channels=channels,
        b_maxout = args.maxout,
        pre_conv_channels=hparams.pre_conv_channels,
        pre_residuals=hparams.pre_residuals,
        up_residuals=hparams.up_residuals,
        post_residuals=hparams.post_residuals,
        pow_2_channels=args.pow_2_channels,
        reduce_height=reduce_height,
        head_class = head_class,
        linear_ch=hparams.last_ch,
        activation=activation
        )
    return model

def set_debug_args(args):
    args.N = hparams.debug_N				
    hparams.pre_conv_channels = hparams.debug_pre_conv_channels
    hparams.pre_residuals = hparams.debug_pre_residuals
    hparams.up_residuals = hparams.debug_up_residuals
    hparams.post_residuals = hparams.debug_post_residuals
    hparams.n_heads = hparams.debug_n_heads
    args.model = hparams.debug_model
    args.mode = hparams.debug_mode
    args.batch_size = hparams.debug_batch_size
    args.loss_mode = hparams.debug_loss_mode
    args.comp_test_name_m = hparams.debug_comp_test_name_m
    args.comp_test_name = 'debug'
    if args.model == 2:
        hparams.channels = hparams.debug_channels_model2
    elif args.model == 3:
        hparams.channels = hparams.debug_channels_model3
    else:
       hparams.channels = hparams.debug_channels_model1
    args.train_data_size = hparams.debug_train_data_size
    args.val_data_size = hparams.debug_val_data_size
    print('WARNING!! DEBUG value is True!')
    args.epochs = hparams.debug_epochs
    hparams.last_ch = hparams.debug_last_ch
    args.read_baseline = hparams.debug_read_baseline
    args.scheduler = hparams.debug_scheduler
    args.K = hparams.debug_K
    args.loss_method = hparams.debug_loss_method
    return args
    
    
def prepare_data_loader(dataset, args):
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=False
    )
    
    return dataloader

def print_model_summary(args, model):
    # Get model summary as a string
    mid_layer ='maxout' if args.maxout == True else 'conv1'
    print(f'mid_layer is {mid_layer}')
    print(args)
    print(hparams)

def init(args):
    if args.read_baseline:
        # Set folder to write test data to
        folder_python = os.path.join(os.path.join(hparams.data_root, 'tests'), 
                                   args.comp_test_name)
        if not os.path.exists(folder_python):
            os.mkdir(folder_python)
        else:
            print(f'run {args.comp_test_name} already exists')
        # Set folder to read baseline data from
        folder_matlab = os.path.join(os.path.join(hparams.data_root, 'baseline_data'), 
                                                 args.comp_test_name_m)
        if not os.path.exists(folder_matlab):
            print('Error! folder_matlab does not exist\n'
                  f'path={folder_matlab}')    
            exit(1)
    else:
        folder_matlab = ''
        folder_python = ''
        
    return folder_matlab, folder_python

def set_optimizer(args, model):
    
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=hparams.opt_sgd_momentum,
                                    weight_decay=hparams.opt_sgd_weight_decay)
    elif args.optimizer == 'RMSProp':
        optimizer = torch.optim.RMSProp(model.parameters(), lr=args.lr, 
                                        alpha=hparams.opt_rms_prop_alpha,
                                        eps=hparams.opt_eps)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=hparams.opt_adam_w_betas,
                                      eps=hparams.opt_adam_w_eps,
                                      weight_decay=hparams.opt_adam_w_weight_decay)
    else: # Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                      betas=hparams.opt_adam_betas,
                                      eps=hparams.opt_adam_eps,
                                      weight_decay=hparams.opt_adam_weight_decay)
        
    return optimizer


def set_scheduler(scheduler_name, optimizer, epochs, len_trainloader):
    scheduler = None
    if scheduler_name != 'None':
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',
                factor=hparams.reduce_lr_factor,
                threshold=hparams.reduce_lr_threshold,
                patience=hparams.reduce_lr_patience,
                cooldown=hparams.reduce_lr_cooldown)
        elif scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=hparams.step_lr_step_size,
                gamma=hparams.step_lr_gamma)
        elif scheduler_name == 'OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=hparams.cyc_lr_max_lr,
                steps_per_epoch=len_trainloader,
                epochs=epochs,
                pct_start=hparams.cyc_lr_pct_start,
                anneal_strategy=hparams.cyc_lr_anneal_strategy)#,
                #three_pahse=hparams.cyc_lr_three_pahse)  
        elif scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=epochs * len_trainloader * hparams.cos_ann_lr_T_max_f) 
        elif scheduler_name == 'CyclicLR':        
            scheduler = optim.lr_scheduler.CyclicLR(
                optimizer=optimizer,
                mode=hparams.cyclic_lr_mode,
                base_lr=hparams.cyclic_lr_base_lr, 

                max_lr=hparams.cyclic_lr_max_lr,
                step_size_up=int(epochs * len_trainloader / 2 / hparams.cyclic_lr_step_size_up_f),
                gamma=hparams.cyclic_lr_gamma) 

        return scheduler
    

    
def update_suffix(args, debug):
    if debug == True:
        args.suffix += 'debug'
    args.suffix += f'{args.comp_test_name}'
    args.suffix += f'_N{args.N}_bs_{args.batch_size}_ep{args.epochs}'\
                    f'_tr_d_sz{args.train_data_size}_val_d_sz{args.val_data_size}'\
                    f'_model{args.model}_{args.mode}_n_heads{args.n_heads}'\
                    f'_loss_{args.loss_mode}_lr_{args.lr}'
    if args.scheduler != 'None':
        args.suffix += f'_dynamic_lr_{args.scheduler}'
    if hparams.dilation_mid > 1:
        args.suffix += f'_dilation_mid{hparams.dilation_mid}'
    
    return args

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Set debug flag
    DEBUG = hparams.DEBUG
    # Set wandb flag
    wandb_flag = args.wandb
    
    if DEBUG ==  True:
        args = set_debug_args(args)

    args = update_suffix(args, DEBUG)

    # Initialize args
    folder_matlab, folder_python = init(args)
    # Initialize model and optimizer
    model = get_model(device, args)
    optimizer = set_optimizer(args, model)
    # print and save model
    if args.log_level >= 2:
    	print_model_summary(args, model)

    # Set train dataset and dataloader
    print('Set train data')
    read_baseline_train = True if args.read_baseline == 1 else False

    train_dataset = create_dataset(device, args.train_data_size, args.K, args.N,
                                   read_baseline_train, args.mode,
                                   folder_matlab)

    train_loader = prepare_data_loader(train_dataset, args)
    # Set validation dataset and dataloader 
    print('Set validation data')
    read_baseline_val = True if args.read_baseline == 2 else False

    val_dataset = create_dataset(device, args.val_data_size, args.K, args.N,
                                 read_baseline_val, ['opt', 'none'],
                                 folder_matlab)
    val_loader = prepare_data_loader(val_dataset, args)
    
    scheduler = set_scheduler(args.scheduler, optimizer, args.epochs, len(train_loader))
    # if exists, load from checkpoint
    ckp_path = os.path.join(f'{folder_python}', 'ckp.pt')

    if os.path.exists(ckp_path):
        print('checkpoint found, loading...')
        checkpoint = torch.load(ckp_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.scheduler_from_start: 
            scheduler = set_scheduler(args.scheduler, optimizer, args.epochs - epoch, len(train_loader))
        else:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if epoch >= args.epochs:
            print(f'Error! epoch={epoch} must be smaller then args.epochs={args.epochs}')
            exit(1)
    else:
        epoch = 0
    # Initialize trainer
    trainer = Trainer(model=model, 
                      train_loader=train_loader, 
                      val_loader=val_loader, 
                      train_dataset=train_dataset, 
                      val_dataset=val_dataset, 
                      wandb_flag=wandb_flag,
                      device=device,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      scheduler_name=args.scheduler,
                      folder_matlab=folder_matlab,
                      folder_python=folder_python,
                      start_epoch=epoch,
                      args=args)
    
    start_time = time.time()
    run = None
    if wandb_flag:
        wandb.login()
        if args.wandb_run_id == '':
            run = wandb.init(project=args.wandb_proj_name,
               	           name = f"{args.suffix}",
               	           config=args)
            wandb.log({"cmd_line": sys.argv})
            wandb.save('hparams.py')
            wandb.save("train_main.py")
            wandb.save(f"model{args.model}.py")     
        else: #resume run
            run_id = args.wandb_run_id
            resume_mode = "must"
            run = wandb.init(project=args.wandb_proj_name, 
                             id=run_id, 
                             resume=resume_mode)
    # Train and evaluate
    trainer.run()
    end_time = time.time()
        
    print(f"Time taken to train in {os.path.basename(__file__)}:", 
          end_time - start_time, "seconds")

          
if __name__ == "__main__":
    # Add arguments to parser
    parser = argparse.ArgumentParser(description='Inverting the bispectrum. Pulse dataset')

    parser.add_argument('--N', type=int, default=10, metavar='N',
            help='size of vector in the dataset')
    parser.add_argument('--K', type=int, default=1, metavar='N',
            help='Number of signals to reconstruct from')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
            help='batch size')
    parser.add_argument('--wandb_proj_name', type=str, default='BS_G_inv_multi_gpu', metavar='N',
            help='wandb project name')
    parser.add_argument('--save_every', type=int, default=100, metavar='N',
            help='save checkpoint every <save_every> epoch')
    parser.add_argument('--epochs', type=int, default=5000, metavar='N',
            help='number of epochs to run')
    parser.add_argument('--train_data_size', type=int, default=5000, metavar='N',
            help='the size of the train data') 
    parser.add_argument('--val_data_size', type=int, default=100, metavar='N',
            help='the size of the validate data')  
    parser.add_argument('--scheduler', type=str, default='None',
            help='\'StepLR\', \'ReduceLROnPlateau\', \'OneCycleLR\','
            ' \'CosineAnnealingLR\', \'CyclicLR\', \'Manual\'. '
            'Update configurtion parametes accordingly. '
            'default: \'None\' - no change in lr') 
    parser.add_argument('--scheduler_from_start', action='store_true', 
                        help='In case of loading from checkpoint, if set, start scheduler from scratch.'
                        ' Else, resume scheduler from checkpoint.') 
    parser.add_argument('--lr', type=float, default=1e-2, metavar='f',
            help='learning rate (initial for dynamic lr, otherwise fixed)')  
    parser.add_argument('--mode', type=str, nargs='+', default=['opt'],
            help= '[mode, add], mode in {\'rand\'\,\'opt\'}, add (optioanl) in {\'shift\', \'circular_shifts\'}'
                '\'rand\': Create random data during training.\n'
                    '\'opt\': Create a fixed dataset'
                    '\'shift\': Randomly shift the signal.\n'
                    '\'circular_shifts\': shift the signal circularly for every bbatch') 
    parser.add_argument('--suffix', type=str, default='',
            help='suffix to add to the name of the cnn yml file')  
    parser.add_argument('--comp_test_name', type=str, default='',
            help='test name') 
    parser.add_argument('--comp_test_name_m', type=str, default='',
            help='test name matlab') 
    parser.add_argument('--log_level', type=int, default=0, 
                        help='0: info, 1: warning, '
                        '2: debug, 3: detailed debug')
    ##---- model parameters
    parser.add_argument('--n_heads', type=int, default=1, 
                    help='number of cnn heads')
    parser.add_argument('--model', type=int, default=3,  
                        help='1 for CNNBS1 - reshape size to reduce dimension'
                        ' 2 for CNNBS2 - strided convolution to reduce dimension')
    # for CNNBS2
    parser.add_argument('--reduce_height', type=int, nargs='+', default=[4, 3, 3], 
                        help='relevant only for model2 - [count kernel stride]'
                        'for reducing height in tensor: BXCXHXW to BXCX1XW')
    parser.add_argument('--loss_mode', type=str, default="l1",  
                        help='\'all\' - l1, mse, rel_mse. default: \'l1\' - l1 loss.'
                        'Note: the training loss is always l1') 
    parser.add_argument('--loss_method', type=str, default="average",  
                        help='one of \'average\', \'sum\'.'
                        'Note: the training loss is always l1') 
    parser.add_argument('--read_baseline', type=int, default=0, 
                        help='0: no action, 1: read from matlab to training set'
                        '2: read from matlab to validation set')

    #evaluates to False if not provided, else True
    parser.add_argument('--wandb', action='store_true', 
                        help='Log data using wandb') 
    parser.add_argument('--wandb_run_id', type=str, default="",
                        help='run id to resume running. If not provided - new run.') 
    parser.add_argument('--maxout', action='store_true', 
                        help='True for maxout in middle layer, False for conv1 (default)')
    parser.add_argument('--pow_2_channels', action='store_true', 
                        help='True for power of 2 channels, '
                        'False for 1 layer with output channel of 8 (default)')
    parser.add_argument('--normalize', action='store_true',
                        help='normalizing data for True, else False (default)')
    parser.add_argument('--early_stopping', action='store_true', 
                        help='early stopping after early_stopping times. '
                        'Update early_stopping in configuration') 
    parser.add_argument('--optimizer', type=str, default="AdamW",  
                        help='The options are \"Adam\"\, \"SGD\"\, \"RMSprop\"\, \"AdamW\"\n'
                        'Please update relevant parameters in parameters file.') 
    

    # Parse arguments
    args = parser.parse_args()

    main(args)