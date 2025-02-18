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
from models.model4 import HeadBS4
from models.model5 import HeadBS5
from config.hparams import hparams
import numpy as np
from trainer import Trainer
import sys
from torch import nn
from utils.compare_to_baseline import read_tensor_from_matlab
import random 
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
#torch.set_printoptions(precision=15)
#torch.set_default_dtype(torch.float64)
# Set the same seed for reproducibility
from config.hparams import hparams
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

# torch.manual_seed(234)

               

class BispectrumDataset(Dataset):
    
    def __init__(self, source, target):
        super().__init__()
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
                   folder_matlab, data_type, window_size, normalize=False, is_distributed=False):
    if is_distributed:
        device='cpu'
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
    # if N % window_size != 0:
    #     pdb.set_trace()
    #     padding = (window_size - (N % window_size)) % window_size
    #     source = F.pad(target, (0, padding, 0, padding))
    #     print(f"Padding the bispectrum with {padding}")
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
    

def get_model(device, args, is_distributed=False, use_transformers=True):
    channels = args.channels
    args.pre_conv_channels[-1] = args.last_ch
    channels[-1] = args.last_ch
    cnt, k, s = args.reduce_height
    reduce_height = update_reduce_height_cnt(k, s, args.N)
    activation = set_activation(hparams.activation)
    
    if args.n_heads > 1:
        model = get_model_multi_head(device, args, channels, reduce_height,
                                 activation, is_distributed)
    else:
        model = get_model_simple(device, args, channels, reduce_height,
                                 activation, is_distributed, use_transformers)
    return model
    
def get_model_multi_head(device, args, channels, reduce_height,
                         activation, is_distributed=False):
    if args.model == 2:
        head_class = HeadBS2
    elif args.model == 3:
        head_class = HeadBS3 
    else:
        head_class = HeadBS1
    
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
        ).to(device)
        
    return model

def get_model_simple(device, args, channels, reduce_height,
                         activation, is_distributed=False, use_transformers=True):
    
    if use_transformers:
        model = HeadBS4(
            device=device,
            input_len=args.N,
            signals_count = args.K,
            channels=channels,
            pre_residuals=args.pre_residuals,
            pre_conv_channels=args.pre_conv_channels,
            up_residuals=args.up_residuals,
            b_maxout = args.maxout,
            post_residuals=args.post_residuals,
            pow_2_channels=args.pow_2_channels,
            reduce_height=reduce_height,
            last_ch=args.last_ch,
            activation=activation,
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
            ).to(device)
    else:
        model = HeadBS5(
            device=device,
            input_len=args.N,
            signals_count = args.K,
            channels=channels,
            pre_residuals=args.pre_residuals,
            pre_conv_channels=args.pre_conv_channels,
            up_residuals=args.up_residuals,
            b_maxout = args.maxout,
            post_residuals=args.post_residuals,
            pow_2_channels=args.pow_2_channels,
            reduce_height=reduce_height,
            last_ch=args.last_ch,
            activation=activation
            ).to(device)        
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

def load_model_safely(model, checkpoint_path):
    # Load the checkpoint
    state_dict = torch.load(checkpoint_path)
    
    try:
        # Try loading with strict=True (default behavior)
        model.load_state_dict(state_dict['model_state_dict'])
    except RuntimeError as e:
        print("⚠️ Warning: Model loading failed due to unexpected/missing keys.")
        print("Retrying with strict=False...")
        
        # Retry with strict=False to ignore mismatched keys
        model.load_state_dict(state_dict['model_state_dict'], strict=False)
        print("Model loaded successfully with strict=False.")    
    
def prepare_data_loader(dataset, batch_size, is_distributed=False):
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=is_distributed,
        shuffle=False#,
        #num_workers=os.cpu_count(),
    )
    
    return dataloader

def set_optimizer(args, model):
    
    args.lr = args.lr * args.nprocs
    
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


def set_scheduler(scheduler_name, optimizer, epochs, lr, len_trainloader):
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
                max_lr=lr,
                div_factor=hparams.cyc_lr_div_factor,
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
                max_lr=lr,
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

def train(args, params):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    _train_impl(0, args, params)

def train_distributed(args, params):
    # Apply ddp setup
    ddp_setup()
    
    device = int(os.environ["LOCAL_RANK"])
    print(f'Using GPU {device}')    

    _train_impl(device, args, params, is_distributed=True)


def init(args):
    # Set folder to write test data to
    folder_python = os.path.join('output', args.comp_test_name)
    # The folder does not exist
    if not os.path.exists(folder_python):
            os.mkdir(folder_python)

    if args.read_baseline:
        # Set folder to read baseline data from
        folder_matlab = os.path.join('data', args.comp_test_name_m)
        if not os.path.exists(folder_matlab):
            raise ValueError('Error! folder_matlab does not exist\n'
                  f'path={folder_matlab}')    
    else:
        folder_matlab = ''

    return folder_matlab, folder_python
    
    
def ddp_setup():
    device = int(os.environ["LOCAL_RANK"])
    # device = torch.device('cuda', device)
    torch.cuda.set_device(device)
    init_process_group(backend="nccl", init_method="env://")


def _train_impl(device, args, params, is_distributed=False):
    torch.backends.cudnn.benchmark = True
    # Set debug flag
    DEBUG = args.debug
    # Set wandb flag
    wandb_flag = args.wandb
    
    if DEBUG ==  True:
        args = set_debug_args(args)
    
    args = update_suffix(args)
    
    # Initialize wandb
    if device == 0:
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
                resume_mode = "must"
                run = wandb.init(project=args.wandb_proj_name, 
                                 id=args.wandb_run_id, 
                                 resume=resume_mode)
            print(f'Running with {args.nprocs} GPUs')
            
    # Initialize args
    folder_matlab, folder_python = init(args)

    # Initialize model and optimizer
    model = get_model(device, args, is_distributed)
    optimizer = set_optimizer(args, model)
    # print and save model
    if device == 0 and args.log_level >= 2:
    	print(model)
    
    # Set train dataset and dataloader
    print('Set train data')
    read_baseline_train = True if args.read_baseline == 1 else False

    train_dataset = create_dataset(device, args.train_data_size, args.K, args.N,
                                   read_baseline_train, args.mode,
                                   folder_matlab, args.data_type, 
                                   args.window_size, 
                                   args.normalize, is_distributed)
    
    train_loader = prepare_data_loader(train_dataset, args.batch_size, is_distributed)
    # Set validation dataset and dataloader 
    print('Set validation data')
    read_baseline_val = True if args.read_baseline == 2 else False
    
    val_dataset = create_dataset(device, args.val_data_size, args.K, args.N,
                                 read_baseline_val, ['opt', 'none'],
                                 folder_matlab, args.data_type,
                                 args.normalize, is_distributed)
    
    val_loader = prepare_data_loader(val_dataset, args.batch_size, is_distributed)
    
    scheduler = set_scheduler(args.scheduler, optimizer, args.epochs, args.lr, len(train_loader))
    # if exists, load from checkpoint
    ckp_path = os.path.join(f'{folder_python}', 'ckp.pt')
    
    if os.path.exists(ckp_path):
        print('checkpoint found')
        if args.run_mode == "override":
           epoch = 0 
           print('Overriding existing checkpoint')
        elif args.run_mode == "resume":
            print('Resuming existing run, loading checkpoint...')
            if is_distributed:
                # configure map_location properly
                map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
                # map_location=f"cuda:{device}"
                checkpoint = torch.load(ckp_path, map_location=map_location)
            else:
                checkpoint = torch.load(ckp_path)
            epoch = checkpoint['epoch']
            
            model.load_state_dict(checkpoint['model_state_dict'])

            # model = model.to(device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.scheduler != "None":
                if args.scheduler_from_start: 
                    scheduler = set_scheduler(args.scheduler, optimizer, args.epochs - epoch, args.lr, len(train_loader))
                else:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if epoch >= args.epochs:
                print(f'Error! epoch={epoch} must be smaller then args.epochs={args.epochs}')
                sys.exit(1)
    else:#new
        epoch = 0
    
    if is_distributed:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
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
                      args=args,
                      is_distributed=is_distributed)
    if device == 0:
        start_time = time.time()    
    
    # Train and evaluate
    trainer.run()
    
    if device == 0:
       	end_time = time.time()
        if wandb_flag:
            np.savetxt(f'{folder_python}/wandb_run_id.csv', [wandb.run.id], fmt='%s')    
        print(f"Time taken to train in {os.path.basename(__file__)}:", 
              end_time - start_time, "seconds")