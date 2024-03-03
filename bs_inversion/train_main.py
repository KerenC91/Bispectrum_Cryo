import torch.optim as optim
import time 
import os
import wandb
import torch 
from torch.utils.data import Dataset, DataLoader
import argparse
from utils import BispectrumCalculator
from model1 import CNNBS, HeadBS1
from model2 import HeadBS2
from model3 import HeadBS3
from hparams import hparams
import numpy as np
from trainer import Trainer
import sys
from torch import nn
from compare_to_baseline import read_tensor_from_matlab
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# Set the same seed for reproducibility
#torch.manual_seed(1234)

#os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"


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
    sample_path = os.path.join(folder, 'data.csv')
    target = read_tensor_from_matlab(sample_path, True) 
    shifts = int(np.loadtxt(os.path.join(folder, 'shifts.csv'), delimiter=" "))
    target = torch.roll(target, -shifts)
    
    return target    

def read_org(folder):
    sample_path = os.path.join(folder, 'x_true.csv')
    target = read_tensor_from_matlab(sample_path, True)  
    
    return target  

def set_read_func(folder_matlab):
    if 'noisy' in folder_matlab:
        f = read_noisy
    else:
        f = read_org
    return f


def create_dataset(device, data_size, N, read_baseline, mode, 
                   comp_baseline_folders):
    device='cpu'
    bs_calc = BispectrumCalculator(data_size, N, device).to(device)
    if read_baseline:
        target = torch.zeros(data_size, 1, N)
        if mode == 'opt':
            _, folder_matlab, _ = \
                    comp_baseline_folders
            data_size = min(data_size, len(os.listdir(folder_matlab)))
            print(f'data_size={data_size}')

            for i in range(data_size):

                folder = os.path.join(folder_matlab, f'sample{i}')
                read_func = set_read_func(folder_matlab)
                
                target[i] = read_func(folder)
  
        else:
            print('Error! read data from baseline mode is only possible for '
                  '\'opt\' mode. Please check your parameters.')
            sys.exit(1)
    else:
        target = torch.randn(data_size, 1, N)
    target.to(device)
    source, target = bs_calc(target)

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
        
    activation = set_activation(hparams.activation)
    model = CNNBS(
        device=device,
        input_len=args.N,
        n_heads=args.n_heads,
        channels=channels,
        b_maxout = args.maxout,
        pre_conv_channels=hparams.pre_conv_channels,
        pre_residuals=hparams.pre_residuals,
        up_residuals=hparams.up_residuals,
        post_residuals=hparams.post_residuals,
        pow_2_channels=args.pow_2_channels,
        reduce_height=hparams.reduce_height,
        head_class = head_class,
        linear_ch=hparams.last_ch,
        activation=activation
        )
    return model   
    
def prepare_data_loader(dataset, args):
    dataloader = None
    
    if args.mode =='opt':
        dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )
    
    return dataloader

def print_model_summary(args, model):
    # Get model summary as a string
    mid_layer ='maxout' if args.maxout == True else 'conv1'
    print(f'mid_layer is {mid_layer}')
    print(args)
    print(hparams)

def init(args):
    # Set wandb flag
    wandb_flag = args.wandb
    if (args.wandb_log_interval == 0):
        wandb_flag = False
    if args.read_baseline:
        folder_test = os.path.join(hparams.comp_root, args.comp_test_name)
        if not os.path.exists(folder_test):
            os.mkdir(folder_test)
        folder_testm = os.path.join(hparams.comp_root, args.comp_test_name_m)
        if not os.path.exists(folder_testm):
            print('Error! folder_testm does not exist\n'
                  f'path={folder_testm}')    
            exit(1)
        folder_matlab = os.path.join(folder_testm, 'data_from_matlab')
        if not os.path.exists(folder_testm):
            print('Error! folder_matlab does not exist\n'
                  f'path={folder_matlab}') 
            exit(1)
        folder_python = os.path.join(folder_test, 'data_from_python')
        if not os.path.exists(folder_python):
            os.mkdir(folder_python)
    else:
        folder_test = ''
        folder_matlab = ''
        folder_python = ''
    
    return wandb_flag, (folder_test, folder_matlab, folder_python)

def set_optimizer(args, model):
    
    lr = args.lr * args.nprocs
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=hparams.opt_sgd_momentum,
                                    weight_decay=hparams.opt_sgd_weight_decay)
    elif args.optimizer == 'RMSProp':
        optimizer = torch.optim.RMSProp(model.parameters(), lr=lr, 
                                        alpha=hparams.opt_rms_prop_alpha,
                                        eps=hparams.opt_eps)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      betas=hparams.opt_adam_w_betas,
                                      eps=hparams.opt_eps,
                                      weight_decay=hparams.opt_adam_w_weight_decay)
    else: # Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                      betas=hparams.opt_adam_betas,
                                      eps=hparams.opt_adam_eps,
                                      weight_decay=hparams.opt_adam_weight_decay)
        
    return optimizer


def set_scheduler(scheduler_name, optimizer, epochs):
    scheduler = None
    if scheduler_name != 'None':
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',
                factor=hparams.reduce_lr_factor,
                threshold=hparams.reduce_lr_threshold,
                patience=hparams.reduce_lr_patience,
                cool_down=hparams.reduce_lr_cool_down)
        elif scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=hparams.step_lr_step_size,
                gamma=hparams.step_lr_gamma)
        elif scheduler_name == 'OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr = hparams.cyc_lr_max_lr,
                steps_per_epoch =1,
                epochs= epochs,
                pct_start = hparams.cyc_lr_pct_start,
                anneal_strategy=hparams.cyc_lr_anneal_strategy)                
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

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355" #any free port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(device, args):
    
    # Apply ddp setup
    ddp_setup(device, args.nprocs)
    
    if torch.cuda.is_available():
        print("GPU available!")
    else:
        print("GPU not available, using CPU.")
    print(f'Using GPU {device}')
    
    args = update_suffix(args)
    wandb_flag, comp_baseline_folders = init(args)
    
    # Initialize model and optimizer
    model = get_model(device, args)
    optimizer = set_optimizer(args, model)
    scheduler = set_scheduler(args.scheduler, optimizer, args.epochs)
    # print and save model
    print_model_summary(args, model)

    # set train dataset and dataloader
    
    read_baseline_train = True if args.read_baseline == 1 else False
    train_dataset = create_dataset(device, args.train_data_size, args.N,
                                   read_baseline_train, args.mode,
                                   comp_baseline_folders)

    train_loader = prepare_data_loader(train_dataset, args)
    # set validation dataset and dataloader 
    read_baseline_val = True if args.read_baseline == 2 else False
    val_dataset = create_dataset(device, args.val_data_size, args.N,
                                 read_baseline_val, args.mode,
                                 comp_baseline_folders)
    val_loader = prepare_data_loader(val_dataset, args)
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
                      comp_baseline_folders=comp_baseline_folders,
                      args=args)
 
    # Get start time
    if trainer.device not in [-1, 0]:
        dist.barrier()
    # Only gpu 0 operating now...
    if trainer.device == 0:
        start_time = time.time()
        run = None
        if wandb_flag:
            wandb.login()
            run = wandb.init(project='GaussianBispectrumInversion',
                               name = f"{args.suffix}",
                               config=args)
            wandb.log({"cmd_line": sys.argv})
            wandb.save('hparams.py')
            wandb.save("train_main.py")
            wandb.save(f"model{args.model}.py")        
            wandb.watch(model, log_freq=100)
        dist.barrier()
    # Train and evaluate
    trainer.run()
    # Get end time 
    if trainer.device not in [-1, 0]:
        dist.barrier()
    if trainer.device == 0:
        # Only gpu 0 operating now...        
        if wandb_flag:
            folder = f'figures/cnn_{args.suffix}'
            fig_path = f'{folder}/x_vs_x_rec.png'
            #wandb.upload_file(fig_path, f"x_vs_x_rec_ep{args.epochs - 1}.png")
            artifact = wandb.Artifact("x_vs_x_rec", type="figure")
            artifact.add_file(fig_path, name="x_vs_x_rec.png")
            run.log_artifact(artifact)
        end_time = time.time()
        
        print(f"Time taken to train in {os.path.basename(__file__)}:", 
              end_time - start_time, "seconds")
        dist.barrier()
        
    destroy_process_group()          

if __name__ == "__main__":
    # Add arguments to parser
    parser = argparse.ArgumentParser(description='Inverting the bispectrum. Pulse dataset')

    parser.add_argument('--N', type=int, default=10, metavar='N',
            help='size of vector in the dataset')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
            help='batch size')
    parser.add_argument('--wandb_log_interval', type=int, default=10, metavar='N',
            help='interval to log data to wandb')
    parser.add_argument('--save_every', type=int, default=100, metavar='N',
            help='save checkpoint every <save_every> epoch')
    parser.add_argument('--epochs', type=int, default=5000, metavar='N',
            help='number of epochs to run')
    parser.add_argument('--train_data_size', type=int, default=5000, metavar='N',
            help='the size of the train data') 
    parser.add_argument('--val_data_size', type=int, default=100, metavar='N',
            help='the size of the validate data')  
    parser.add_argument('--scheduler', type=str, default='None',
            help='\'StepLR\', \'ReduceLROnPlateau\', \'Manual\'. '
            'Update configurtion parametes accordingly. '
            'default: \'None\' - no change in lr') 
    parser.add_argument('--lr', type=float, default=1e-3, metavar='f',
            help='learning rate (initial for dynamic lr, otherwise fixed)')  
    parser.add_argument('--mode', type=str, default='opt',
            help='\'rand\': Create random data during training if True.'
                    '\'opt\': Optimize on predefined data')  
    parser.add_argument('--suffix', type=str, default='',
            help='suffix to add to the name of the cnn yml file')  
    parser.add_argument('--config_mode', type=int, default=0, 
            help='0 for hparams, 2 for hparams2, 3 for hparams3') 
    parser.add_argument('--comp_test_name', type=str, default='',
            help='test name') 
    parser.add_argument('--comp_test_name_m', type=str, default='',
            help='test name matlab') 
    ##---- model parameters
    parser.add_argument('--n_heads', type=int, default=1, 
                    help='number of cnn heads')
    parser.add_argument('--model', type=int, default=1,  
                        help='1 for CNNBS1 - reshape size to reduce dimension'
                        ' 2 for CNNBS2 - strided convolution to reduce dimension')
    # for CNNBS2
    parser.add_argument('--reduce_height', type=int, nargs='+', default=[4, 3, 3], 
                        help='relevant only for model2 - [count kernel stride]'
                        'for reducing height in tensor: BXCXHXW to BXCX1XW')
    parser.add_argument('--loss_mode', type=str, default="l1",  
                        help='\'all\' - l1, mse, rel_mse. default: \'l1\' - l1 loss.'
                        'Note: the training loss is always l1') 
    parser.add_argument('--read_baseline', type=int, default=0, 
                        help='0: no action, 1: read from matlab to training set'
                        '2: read from matlab to validation set')

    #evaluates to False if not provided, else True
    parser.add_argument('--wandb', action='store_true', 
                        help='Log data using wandb') 
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
    parser.add_argument('--optimizer', type=str, default="Adam",  
                        help='The options are \"Adam\"\, \"SGD\"\, \"RMSprop\"\, \"AdamW\"\n'
                        'Please update relevant parameters in parameters file.') 
    parser.add_argument('--nprocs', default=torch.cuda.device_count(), type=int, help='nprocs, default is the number of available gpus on the machine')

    # Parse arguments
    args = parser.parse_args()

    # if args.batch_size is None:
    #     args.batch_size = int(args.num_imgs / args.nprocs)
    #args.batch_size *= args.nprocs    
    # Otherwise value is set as the user provided
    mp.spawn(main, args=(args,), nprocs=args.nprocs)
            

