import torch.optim as optim
import time 
import os
import wandb
import torch 
from torch.utils.data import Dataset, DataLoader
import argparse
from utils.utils import BispectrumCalculator, rand_shift_signal, create_gaussian_pulse
from models.model1 import CNNBS, HeadBS1
from models.model2 import HeadBS2
from models.model3 import HeadBS3
from config.hparams import hparams
import numpy as np
from trainer import Trainer
import sys
from torch import nn
from utils.compare_to_baseline import read_tensor_from_matlab
import random 
#torch.set_printoptions(precision=15)
#torch.set_default_dtype(torch.float64)
# Set the same seed for reproducibility

torch.manual_seed(234)


                

class BispectrumDataset(Dataset):
    
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

def read_dataset_from_baseline(folder_matlab, data_size, K, N, label='x_true'):
    read_func = set_read_func(folder_matlab)
    data_size = min(data_size, len(os.listdir(folder_matlab)))
    target = torch.zeros(data_size, K, N)

    print(f'The updated data size is {data_size}')

    for i in range(data_size):
        folder = os.path.join(folder_matlab, f'sample{i}')
        for j in range(K):
            target[i][j] = read_func(folder, j, K, label)   
    
    return target
   
def create_dataset(device, data_size, K, N, read_baseline, mode, 
                   folder_matlab, data_type, normalize=False):
    bs_calc = BispectrumCalculator(K, N, device).to(device)
    print(f'read_baseline={read_baseline}, mode={mode}')
    if read_baseline: # in val dataset
        if data_type == 'gaussian_pulse':
            print("Error! Gaussian pulse does not have data to read from.")
            sys.exit(1)
        target = read_dataset_from_baseline(folder_matlab, data_size, K, N)
    else:
        if mode[0] == 'opt':
            # Create random dataset
            if data_type == 'gaussian_pulse':
                eff_data_size = data_size * K
                target = torch.zeros(eff_data_size, N)
                n_per_side = N / 2.0
                percentage = 0.1
                mean = np.random.uniform(-n_per_side, n_per_side, eff_data_size) - percentage * n_per_side
                std = np.random.uniform(0.0, 1000.0, eff_data_size)
                for i in range(eff_data_size):
                    target[i], _ = create_gaussian_pulse(mean[i], std[i], N)
                target = target.view(data_size, K, N)
            else: # normal distribution
                target = torch.randn(data_size, K, N)
        elif mode[0] == 'rand':
            # Initialize dataset to zeros and create data on the fly 
            target = torch.zeros(data_size, K, N)
    if normalize:
        y = torch.fft.fft(target, dim=-1)
        y /= torch.norm(y, dim=-1).unsqueeze(2)
        target = torch.fft.ifft(y, dim=-1)
        target = target.type(torch.float32)
    target.to(device)
    source, target = bs_calc(target)
    if mode[0] == 'opt' and mode[1] == 'shift' and not read_baseline:
            target, shifts = rand_shift_signal(target, K, N, data_size)
    dataset = BispectrumDataset(source, target)

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
        # channels = hparams.channels_model2
    elif args.model == 3:
        head_class = HeadBS3 
        # channels = hparams.channels_model3
    else:
        head_class = HeadBS1
        # channels = hparams.channels_model1
    
    channels = args.channels
    args.pre_conv_channels[-1] = args.last_ch
    channels[-1] = args.last_ch
    cnt, k, s = args.reduce_height
    reduce_height = update_reduce_height_cnt(k, s, args.N)
    activation = set_activation(hparams.activation)
    model = CNNBS(
        device=device,
        input_len=args.N,
        signals_count = args.K,
        n_heads=args.n_heads,
        channels=channels,
        b_maxout = args.maxout,
        pre_conv_channels=args.pre_conv_channels,
        pre_residuals=args.pre_residuals,
        up_residuals=args.up_residuals,
        post_residuals=args.post_residuals,
        pow_2_channels=args.pow_2_channels,
        reduce_height=reduce_height,
        head_class = head_class,
        linear_ch=args.last_ch,
        activation=activation,
        #
        window_size = args.window_size,
        img_size = args.img_size,
        patch_size = args.patch_size,
        depths = args.depths,
        num_heads = args.num_heads,
        qkv_bias = args.qkv_bias,
        qk_scale = args.qk_scale,
        drop = args.drop,
        attn_drop = args.attn_drop,
        drop_path_rate = args.drop_path_rate,
        norm_layer = args.norm_layer,
        downsample = args.downsample,
        resi_connection = args.resi_connection
        #Add here!!! attention params
        )
    return model


def set_debug_args(args):
    args.N = hparams.debug_N				
    args.pre_conv_channels = hparams.debug_pre_conv_channels
    args.pre_residuals = hparams.debug_pre_residuals
    args.up_residuals = hparams.debug_up_residuals
    args.post_residuals = hparams.debug_post_residuals
    args.n_heads = hparams.debug_n_heads
    args.model = hparams.debug_model
    args.mode = hparams.debug_mode
    args.batch_size = hparams.debug_batch_size
    args.loss_mode = hparams.debug_loss_mode
    args.comp_test_name_m = hparams.debug_comp_test_name_m
    args.comp_test_name = 'debug'
    if args.model == 2:
        args.channels = hparams.debug_channels_model2
    elif args.model == 3:
        args.channels = hparams.debug_channels_model3
    else:
        args.channels = hparams.debug_channels_model1
    args.train_data_size = hparams.debug_train_data_size
    args.val_data_size = hparams.debug_val_data_size
    print('WARNING!! DEBUG value is True!')
    args.epochs = hparams.debug_epochs
    args.last_ch = hparams.debug_last_ch
    args.read_baseline = hparams.debug_read_baseline
    args.scheduler = hparams.debug_scheduler
    args.K = hparams.debug_K
    args.loss_method = hparams.debug_loss_method
    return args
    
    
def prepare_data_loader(dataset, batch_size):
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
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

    # Set folder to write test data to
    folder_python = os.path.join('output', args.comp_test_name)
    if not os.path.exists(folder_python):
        if args.run_mode == "resume" or args.run_mode == "override":
            print(f'Error! folder {folder_python} does not exist')
            sys.exit(1)
        else:#"new"
            os.mkdir(folder_python)
    else:
        if len(os.listdir(folder_python)) > 0:
            print(f'run {args.comp_test_name} already exists')
            if args.run_mode == "new":
                name_updated = False
                for trial in range(5):
                    random_number = random.randint(0, 50)
                    if not os.path.exists(f'{folder_python}_{random_number}'):
                        folder_python += f"_{random_number}"
                        print(f'run name has been updated to {folder_python}')
                        name_updated = True
                        break
                if name_updated == False:
                    print(f'Error! Could not update run name {folder_python}')
                    sys.exit(1)
    if args.read_baseline:
        # Set folder to read baseline data from
        folder_matlab = os.path.join('data', args.comp_test_name_m)
        if not os.path.exists(folder_matlab):
            print('Error! folder_matlab does not exist\n'
                  f'path={folder_matlab}')    
            sys.exit(1)
    else:
        folder_matlab = ''
        
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
    

    
def update_suffix(args):
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

    args = update_suffix(args)

    # Initialize wandb
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
            
    # Initialize args
    folder_matlab, folder_python = init(args)
    # Initialize model and optimizer
    model = get_model(device, args)
    optimizer = set_optimizer(args, model)
    # print and save model
    if args.log_level >= 2:
    	print(model)

    # Set train dataset and dataloader
    print('Set train data')
    read_baseline_train = True if args.read_baseline == 1 else False

    train_dataset = create_dataset(device, args.train_data_size, args.K, args.N,
                                   read_baseline_train, args.mode,
                                   folder_matlab, args.data_type, 
                                   args.normalize)

    train_loader = prepare_data_loader(train_dataset, args.batch_size)
    # Set validation dataset and dataloader 
    print('Set validation data')
    read_baseline_val = True if args.read_baseline == 2 else False

    val_dataset = create_dataset(device, args.val_data_size, args.K, args.N,
                                 read_baseline_val, ['opt', 'none'],
                                 folder_matlab, args.data_type,
                                 args.normalize)
    
    val_loader = prepare_data_loader(val_dataset, args.batch_size)
    
    scheduler = set_scheduler(args.scheduler, optimizer, args.epochs, len(train_loader))
    # if exists, load from checkpoint
    ckp_path = os.path.join(f'{folder_python}', 'ckp.pt')

    if os.path.exists(ckp_path):
        print('checkpoint found')
        if args.run_mode == "override":
           epoch = 0 
           print('override existing checkpoint')
        elif args.run_mode == "resume":
            print('loading checkpoint...')
            checkpoint = torch.load(ckp_path)
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.scheduler != "None":
                if args.scheduler_from_start: 
                    scheduler = set_scheduler(args.scheduler, optimizer, args.epochs - epoch, len(train_loader))
                else:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if epoch >= args.epochs:
                print(f'Error! epoch={epoch} must be smaller then args.epochs={args.epochs}')
                sys.exit(1)
    else:#new
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
                      optimizer_name=args.optimizer,
                      scheduler=scheduler,
                      scheduler_name=args.scheduler,
                      folder_matlab=folder_matlab,
                      folder_python=folder_python,
                      start_epoch=epoch,
                      args=args)
    
    start_time = time.time()
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
    parser.add_argument('--lr', type=float, default=3e-4, metavar='f',
            help='learning rate (initial for dynamic lr, otherwise fixed)')     
    parser.add_argument('--mode', type=str, nargs='+', default=['opt', None],
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
    parser.add_argument('--plotting_off', action='store_true', 
                        help='If set, do not plot data samples at the end. Can draw '
                        'offline using saved checkpoint and initial samples.') 
    parser.add_argument('--optimizer', type=str, default="AdamW",  
                        help='The options are \"Adam\"\, \"SGD\"\, \"RMSprop\"\, \"AdamW\"\n'
                        'Please update relevant parameters in parameters file.') 
    parser.add_argument('--clip_grad_norm', type=float, default=0.,  
                        help='If greater than 0: clip gradients norm with the clip_grad_norm value.') 
    parser.add_argument('--run_mode', type=str, default="new", 
                        help='one out of \"override\", \"resume\", \"new\" existing run '
                        'eventhough a checkpoint exists') 
    parser.add_argument('--data_type', type=str, default="normal_distribution", 
                        help='one out of \"normal_distribution\", \"gaussian_pulse\". '
                        'gaussian_pulse does not have baseline data to read from.') 
    parser.add_argument('--loss_criterion', type=str, default="l1", 
                        help='one out of \"l1\", \"mse\", \"sc\".') 
    # model 
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
    # Swin Transformers params
    parser.add_argument('--window_size', type=int, default=8, 
                        help='window_size')    
    parser.add_argument('--img_size', type=int, default=48, 
                        help='img_size')#seems unused!!!
    parser.add_argument('--patch_size', type=int, default=1, 
                        help='patch size used in training SwinIR. '
                            'Just used to differentiate two different settings in Table 2 of the paper. '
                            'Images are NOT tested patch by patch.')    
    # parser.add_argument('--embed_dim', type=int, default=128, 
    #                     help='embed_dim') #This is exactly last ch
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
    # Parse arguments
    args = parser.parse_args()

    main(args)
