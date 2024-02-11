import torch.optim as optim
import time 
import os
import wandb
from datetime import datetime
import torch 
from torch.utils.data import Dataset, DataLoader
import argparse
from utils import BispectrumCalculator
from model import CNNBS2, CNNBS1
from hparams import hparams, hparams_debug_string
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
import sys

DEBUG = False
# Set the same seed for reproducibility
torch.manual_seed(1234)

class Trainer:
    def __init__(self, model, 
                 train_loader, 
                 val_loader, 
                 wandb_flag,
                 device,
                 args):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = args.batch_size
        self.num_epochs = args.epochs
        self.wandb_log_interval = args.wandb_log_interval
        self.train_data_size = args.train_data_size
        self.val_data_size = args.val_data_size
        self.target_len = args.N
        self.save_every = args.save_every
        self.device = device 
        self.model = model.to(self.device)
        self.wandb_flag = wandb_flag
        self.normalize = args.normalize
        self.mode = args.mode
        self.epoch = 0
        self.last_loss = torch.inf
        self.early_stopping = args.early_stopping
        self.es_cnt = 0
        self.suffix = args.suffix
        self.n_heads = args.n_heads
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss_mode = args.loss_mode
        if self.loss_mode == 'all':
            self.loss_f = self._loss_all
        else:
            self.loss_f = self._loss
        if self.mode == 'opt':
            self.batch_size = 1
            self.train_data_size = 1
        self.bs_calc = BispectrumCalculator(self.batch_size, self.target_len, self.device).to(self.device)
        # if self.mode == 'opt' or self.mode == 'rand':
        #     self.target = torch.randn(self.train_data_size, 1, self.target_len)
        #     self.target.to(self.device)
        #     self.source, self.target = self.bs_calc(self.target)
        #     if self.normalize == True:
        #         mean = torch.mean(self.target)
        #         std = torch.std(self.target)
        #         self.target = (self.target - mean) / std
        

    
    def _loss(self, pred, target):
        bs_pred, pred = self.bs_calc(pred)
        if self.mode == 'opt':
            bs_target = self.source
        else:
            bs_target, target = self.bs_calc(target)
        total_loss = 0.
        if hparams.f1 != 0:
            loss_sc = self._loss_sc(bs_pred, bs_target)
            total_loss += hparams.f1 * loss_sc
        if hparams.f2 != 0:
            loss_log_sc = self._loss_log_sc(bs_pred, bs_target) 
            total_loss += hparams.f2 * loss_log_sc
        if hparams.f3 != 0:
            loss_freq = self._loss_freq(bs_pred, bs_target)
            total_loss += hparams.f3 * loss_freq
        if hparams.f4 != 0:
            loss_weighted_phase = self._loss_weighted_phase(bs_pred, bs_target)
            total_loss += hparams.f4 * loss_weighted_phase
        if hparams.f5 != 0:
            loss_l1 = self._loss_l1(pred, target)
            total_loss += hparams.f5 * loss_l1

        return total_loss

    def _loss_all(self, pred, target):
        bs_pred, pred = self.bs_calc(pred)
        bs_target, target = self.bs_calc(target)
        total_loss = 0.
        if hparams.f1 != 0:
            loss_sc = self._loss_sc(bs_pred, bs_target)
            total_loss += hparams.f1 * loss_sc
        if hparams.f2 != 0:
            loss_log_sc = self._loss_log_sc(bs_pred, bs_target) 
            total_loss += hparams.f2 * loss_log_sc
        if hparams.f3 != 0:
            loss_freq = self._loss_freq(bs_pred, bs_target)
            total_loss += hparams.f3 * loss_freq
        if hparams.f4 != 0:
            loss_weighted_phase = self._loss_weighted_phase(bs_pred, bs_target)
            total_loss += hparams.f4 * loss_weighted_phase
        if hparams.f5 != 0:
            loss_l1 = self._loss_l1(pred, target)
            total_loss += hparams.f5 * loss_l1

        loss = total_loss, \
                self._loss_MSE(pred, target), \
                self._loss_rel_MSE(pred, target)

        return loss

    def _get_params(self):
        params = []
        params += self.model.parameters()
        params += self.f
        
        return params
    
    def _loss_sc(self, bs_pred, bs_gt):
        """
        

        Parameters
        ----------
        pred : TYPE     torch complex-float, NXNX1
            rec_s - reconstructed signal.
        target : TYPE     torch complex-float, NXNX1
            s - target signal (GT).

        Returns
        -------
        TYPE    torch float
            || |BS(rec_s)| - |BS(s)| ||_F / || |BS(s)| ||_F.

        """
        # Get magnitudes
        bs_pred_mag = torch.abs(bs_pred)
        bs_gt_mag = torch.abs(bs_gt)
        return torch.norm(bs_pred_mag - bs_gt_mag) / torch.norm(bs_gt_mag)
    # target - ground truth image, source - Bispectrum of ground truth image
    # might be multiple targets and sources (batch size > 1)

    def _loss_log_sc(self, bs_pred, bs_gt, eps=1e-5):
        """
        

        Parameters
        ----------
        pred : TYPE     torch complex-float, NXNX1
            rec_s - reconstructed signal.
        target : TYPE     torch complex-float, NXNX1
            s - target signal (GT).

        Returns
        -------
        TYPE    torch float
            || log(|BS(s)| + epsilon) - log(|BS(rec_s)| + epsilon) ||_1

        """
        # Get magnitudes
        bs_pred_mag = torch.abs(bs_pred)
        bs_gt_mag = torch.abs(bs_gt)
        return torch.norm(torch.log(bs_gt_mag + eps) - torch.log(bs_pred_mag + eps), p=1)
    # target - ground truth image, source - Bispectrum of ground truth image
    # might be multiple targets and sources (batch size > 1)

    def _loss_freq(self, bs_pred, bs_gt):
        """
        

        Parameters
        ----------
        pred : TYPE     torch complex-float, NXNX1
            rec_s - reconstructed signal.
        target : TYPE     torch complex-float, NXNX1
            s - target signal (GT).

        Returns
        -------
        TYPE    torch float
            || d/dt(<BS(s)) - d/dt(<BS(rec_s)) ||_1

        """
        # Get phases
        bs_pred_phase = torch.angle(bs_pred)
        bs_gt_phase = torch.angle(bs_gt)
        
        #Get derivative phase
        bs_pred_phase_deriv = bs_pred_phase[1:] - bs_pred_phase[:-1]
        bs_gt_phase_deriv = bs_gt_phase[1:] - bs_gt_phase[:-1]
        return torch.norm(bs_gt_phase_deriv - bs_pred_phase_deriv, p=1)
    # target - ground truth image, source - Bispectrum of ground truth image
    # might be multiple targets and sources (batch size > 1)


    def _loss_weighted_phase(self, bs_pred, bs_gt):
        """
        
    
        Parameters
        ----------
        pred : TYPE     torch complex-float, NXNX1
            rec_s - reconstructed signal.
        target : TYPE     torch complex-float, NXNX1
            s - target signal (GT).
    
        Returns
        -------
        TYPE    torch float
            || |BS(s)| .* |BS(rec_s)| - Re{BS(s)} .* Re{BS(rec_s)} - Im{BS(s)} .* Im{BS(rec_s)} ||_1
    
        """
        # Get magnitudes
        bs_pred_mag = torch.abs(bs_pred)
        bs_gt_mag = torch.abs(bs_gt)
        # Get real
        bs_pred_real = bs_pred.real
        bs_gt_real = bs_gt.real
        # Get imaginary
        bs_pred_imag = bs_pred.imag
        bs_gt_imag = bs_gt.imag
        return torch.norm(bs_gt_mag * bs_pred_mag 
                          - bs_gt_real * bs_pred_real - bs_gt_imag * bs_pred_imag, p=1)
        # target - ground truth image, source - Bispectrum of ground truth image
        # might be multiple targets and sources (batch size > 1)

    def _loss_rel_MSE(self, pred, target):
        """
        

        Parameters
        ----------
        pred : TYPE     torch complex-float, NXNX1
            rec_s - reconstructed signal.
        target : TYPE     torch complex-float, NXNX1
            s - target signal (GT).

        Returns
        -------
        TYPE    torch float (normalized mse)
            || s - rec_s ||_F / || s ||_F.

        """

        return torch.mean(
            torch.norm(target - pred, dim=(-1, -2)) / 
            torch.norm(target, dim=(-1, -2)))
        # target - ground truth image, source - Bispectrum of ground truth image
        # might be multiple targets and sources (batch size > 1)

    def _loss_l1(self, pred, target):
        """
        

        Parameters
        ----------
        pred : TYPE     torch complex-float, NXNX1
            rec_s - reconstructed signal.
        target : TYPE     torch complex-float, NXNX1
            s - target signal (GT).

        Returns
        -------
        TYPE    torch float
        || s - rec_s ||_1 / len(s)

        """
        loss = torch.nn.L1Loss()
        return loss(pred, target)
    
        # target - ground truth image, source - Bispectrum of ground truth image
        # might be multiple targets and sources (batch size > 1)
        
    def _loss_MSE(self, pred, target):
        """
        

        Parameters
        ----------
        pred : TYPE     torch complex-float, NXNX1
            rec_s - reconstructed signal.
        target : TYPE     torch complex-float, NXNX1
            s - target signal (GT).

        Returns
        -------
        TYPE    torch float
        || s - rec_s ||_1 / len(s)

        """
        loss = torch.nn.MSELoss()
        return loss(pred, target)
        
    def _run_batch(self, source, target):
        # Move data to device
        target = target.to(self.device)
        source = source.to(self.device)
        # Forward pass
        _, output = self.model(source) # reconstructed signal
        self.last_output = output
        self.target = target
        # Loss calculation
        loss = self.loss_f(output, target)
        return loss
        
    def _run_batch_rand(self):
        if self.mode == 'rand':
            target = torch.randn(self.batch_size, 1, self.target_len)

            source, target = self.bs_calc(target)
        elif self.mode == 'opt':
            source, target = self.source, self.target
        else:
            print(f'error! unknown mode {self.mode}')
        # Move data to device
        target = target.to(self.device)
        source = source.to(self.device)
        # Forward pass
        _, output = self.model(source) # reconstructed signal
        self.last_output = output
        # if self.epoch % hparams.dbg_draw_rate == 0:
        #     self.plot_output_debug(target, output)
        
        # Loss calculation
        loss = self._loss(output, target)
        return loss

    def plot_output_debug(self, target, output):
        folder = f'figures/cnn_{self.suffix}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.figure()
        plt.xlabel("time [sec]")
        plt.title('1D signal')
        plt.plot(target.squeeze(0).detach().cpu().numpy(), label='x')
        plt.plot(output.squeeze(0).detach().cpu().numpy(), label='x_rec')
        plt.legend()
        plt.savefig(f'{folder}/x_vs_x_rec_ep{self.epoch}.png')
        plt.close()
   
    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        folder = f'./checkpoints/cnn_{self.suffix}'
        PATH = f"{folder}/checkpoint_ep{epoch}.pt"
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(ckp, PATH)
        
    def _run_epoch_train(self):
        total_loss = 0

        for idx, (sources, targets) in self.train_loader:
            # zero grads
            self.optimizer.zero_grad()
            # forward pass + loss computation
            loss = self._run_batch(sources, targets)
            # backward pass
            loss.backward()
            # optimizer step
            self.optimizer.step()
            # update avg loss 
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss

    def _run_epoch_train_losses_all(self):   
        total_loss = 0
        total_mse_loss = 0
        total_mse_norm_loss = 0
        
        for idx, (sources, targets) in self.train_loader:
            # zero grads
            self.optimizer.zero_grad()
            # forward pass + loss computation
            loss, mse_loss, rel_mse_loss = self._run_batch(sources, targets)
            # backward pass
            loss.backward()
            # optimizer step
            self.optimizer.step()
            # update avg loss 
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_mse_norm_loss += rel_mse_loss.item()
            
        avg_loss = total_loss / len(self.train_loader)
        avg_mse_loss = total_mse_loss / len(self.train_loader) 
        avg_mse_norm_loss = total_mse_norm_loss / len(self.train_loader) 

        return avg_loss, avg_mse_loss, avg_mse_norm_loss        


    def _run_epoch_train_rand(self):
        total_loss = 0
        total_mse_loss = 0
        total_mse_norm_loss = 0
        
        for _ in range(self.train_data_size):
            # zero grads
            self.optimizer.zero_grad()
            # forward pass + loss computation
            loss = self._run_batch_rand()
            if self.loss_mode == 'all':
                loss, mse_loss, rel_mse_loss = loss
            # backward pass
            loss.backward()
            # optimizer step
            self.optimizer.step()
            # update avg loss 
            total_loss += loss.item()
            if self.loss_mode == 'all':
                total_mse_loss += mse_loss.item()
                total_mse_norm_loss += rel_mse_loss.item()
            
        avg_loss = total_loss / self.train_data_size

        if self.loss_mode == 'all':
            avg_mse_loss = total_mse_loss / self.train_data_size 
            avg_mse_norm_loss = total_mse_norm_loss / self.train_data_size 
            
            avg_loss = avg_loss, avg_mse_loss, avg_mse_norm_loss
        
        return avg_loss
    
    def _run_epoch_validate(self):
        total_loss = 0
        nof_samples = 0
        
        for idx, (sources, targets) in self.val_loader:
            nof_samples += 1
            with torch.no_grad():
                # forward pass + loss computation
                loss = self._run_batch(sources, targets)
                # update avg loss 
                total_loss += loss.item()
            
        avg_loss = total_loss / self.val_data_size
            
        return avg_loss
    
    def _run_epoch_validate_rand(self):
        total_loss = 0
        nof_samples = 0
        
        for _ in range(self.val_data_size):
            nof_samples += 1
            with torch.no_grad():
                # forward pass + loss computation
                loss = self._run_batch_rand()
                if self.loss_mode == 'all':
                    loss, _, _ = loss
                # update avg loss 
                total_loss += loss.item()
                
        avg_loss = total_loss / self.val_data_size
        
        return avg_loss
    
    # one epoch of training           
    def train(self):
        # Set the model to training mode
        self.model.train()
        
        if self.loss_mode == 'all':
            avg_loss = self._run_epoch_train_losses_all()
        else:
            avg_loss = self._run_epoch_train()
        
        return avg_loss
    
    # one epoch of validation           
    def validate(self):
        # Set the model to evaluation mode
        self.model.eval()
        
        if self.mode == 'rand' or self.mode == 'opt':
            return 0
        else:
            avg_loss = self._run_epoch_validate()
        return avg_loss

    # one epoch of testing 
    def test(self):
        return 0
                    
    def run(self):
        for self.epoch in range(self.num_epochs):
            # train             
            train_loss = self.train()
            # validate
            val_loss = self.validate()
            # test
            test_loss = self.test()
            if self.loss_mode == 'all':
                train_loss, mse_loss, rel_mse_loss = train_loss
            # log loss with wandb
            if self.wandb_flag and self.epoch % self.wandb_log_interval == 0:
                wandb.log({"train_loss_l1": train_loss})
                wandb.log({"val_loss": val_loss})
                wandb.log({"test_loss": test_loss})
                if self.loss_mode == 'all':
                    wandb.log({"mse": mse_loss})
                    wandb.log({"relative mse": rel_mse_loss})
            # save checkpoint and log loss to cmd 
            if self.epoch % self.save_every == 0:
                print(f'-------Epoch {self.epoch}/{self.num_epochs}-------')
                print(f'Train loss l1: {train_loss:.6f}')
                print(f'Validation loss: {val_loss:.6f}')
                if self.loss_mode == 'all':
                    print(f'mse loss: {mse_loss:.6f}')
                    print(f'relative mse loss: {rel_mse_loss:.6f}')
                # save checkpoint
                self._save_checkpoint(self.epoch)
            # plot last output
            if self.epoch == self.num_epochs - 1:
                self.plot_output_debug(self.target[0], self.last_output[0])
            # stop early if early_stopping is on
            if self.early_stopping != 0:
                if self.last_loss < train_loss:
                    self.es_cnt +=1
                    if self.es_cnt == self.early_stopping:
                        print(f'Stooped at epoch {self.epoch}, after {self.es_cnt} times\n'
                              f'last_loss={self.last_loss}, curr_los={train_loss}')
                        self.plot_output_debug(self.target, self.last_output)
                        return
            # stop if loss has reached lower bound
            if train_loss < hparams.loss_lim:
                print(f'Stooped at epoch {self.epoch},\n'
                      f'curr_los={train_loss} < {hparams.loss_lim}')    
                self.last_loss = train_loss
                

class UnitVecDataset(Dataset):
    
    def __init__(self, source, target):
        self.target = target
        self.source = source
        self.data_size = self.__len__()
            
        
    def __len__(self):
        return self.target.size(0)
    
    def __getitem__(self, idx):

        return idx, (self.source[idx], self.target[idx])
    
def create_dataset(args, device, data_size):
    bs_calc = BispectrumCalculator(data_size, args.N, device).to(device)
    target = torch.randn(data_size, 1, args.N)
    target.to(device)
    source, target = bs_calc(target)
    
    dataset = UnitVecDataset(source, target)
    return dataset

def get_model(args):
    if args.model == 2:
        model = CNNBS2(
            input_len=args.N,
            n_heads=args.n_heads,
            channels=args.channels,
            b_maxout = args.maxout,
            pre_conv_channels=args.pre_conv_channels,
            pre_residuals=args.pre_residuals,
            up_residuals=args.up_residuals,
            post_residuals=args.post_residuals,
            pow_2_channels=args.pow_2_channels,
            reduce_height=args.reduce_height
            )
    else:
        model = CNNBS1(
            input_len=args.N,
            n_heads=args.n_heads,
            channels=args.channels,
            b_maxout = args.maxout,
            pre_conv_channels=args.pre_conv_channels,
            pre_residuals=args.pre_residuals,
            up_residuals=args.up_residuals,
            post_residuals=args.post_residuals,
            pow_2_channels=args.pow_2_channels
            )    
    return model

def set_debug_data(args):
    args.N = hparams.N				
    args.pre_conv_channels = hparams.pre_conv_channels
    args.pre_residuals = hparams.pre_residuals
    args.up_residuals = hparams.up_residuals
    args.post_residuals = hparams.post_residuals
    args.n_heads = hparams.n_heads
    args.model = hparams.model
    args.mode = hparams.mode
    args.batch_size = hparams.batch_size
    args.loss_mode = hparams.loss_mode
    if args.model == 2:
        args.channels = [256, 64]
    else:
       args.channels = [256, 8] 
    args.train_data_size = hparams.train_data_size
    args.val_data_size = hparams.val_data_size
    print('WARNING!! DEBUG value is True!')
    return args
    
    
def prepare_data_loader(dataset, args):
    dataloader = None
    
    if args.mode =='opt' or args.mode == 'rand':
        dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=False
    )
    else:
        pass
    
    return dataloader

def print_model_summary(args, model):
    # Get model summary as a string
    mid_layer ='maxout' if args.maxout == True else 'conv1'
    print(f'input length: {args.N}\n'
            f'model architecture:\n'
            f'n_heads={args.n_heads}\n'
            f'channels={args.channels}\n'
            f'mid layer={mid_layer}\n'
            f'Normalize={args.normalize}\n'
            f'pow_2_channels={args.pow_2_channels}\n'
            f'pre_conv_channels={args.pre_conv_channels}\n'
            f'pre_residuals={args.pre_residuals}\n'
            f'up_residuals={args.up_residuals}\n'
            f'post_residuals={args.post_residuals}\n'
            f'early_stopping={args.early_stopping}\n'
            f'reduce_height={args.reduce_height}')
    
    summary = str(model)
    
    # Save summary to file
    if not os.path.exists('models'):
        os.makedirs('models')
    with open(f'models/cnn_{args.suffix}.yml', "w") as f:
        f.write(summary)
    print(f'CNN arch yml: models/cnn_{args.suffix}.yml')
    
    
    
def init(args):
    # Set wandb flag
    wandb_flag = args.wandb
    if (args.wandb_log_interval == 0):
        wandb_flag = False
    if args.mode == 'opt':
        args.train_data_size = 1
        args.batch_size = 1
        args.val_data_size = 1
    
    return args, wandb_flag
    
def main():
    # Add arguments to parser
    parser = argparse.ArgumentParser(description='Inverting the bispectrum. Pulse dataset')

    parser.add_argument('--N', type=int, default=10, metavar='N',
            help='size of vector in the dataset')
    parser.add_argument('--batch_size', type=float, default=1, metavar='N',
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
    parser.add_argument('--lr', type=float, default=1e-3, metavar='f',
            help='the size of the test data')  
    parser.add_argument('--mode', type=str, default='opt',
            help='\'rand\': Create random data during training if True.'
                    '\'opt\': Optimize on a single input.')  
    parser.add_argument('--suffix', type=str, default='debug',
            help='suffix to add to the name of the cnn yml file')  

    ##---- model parameters
    parser.add_argument('--n_heads', type=int, default=1, 
                        help='number of cnn heads')
    parser.add_argument('--channels', type=int, nargs='+', 
                        default=[80, 64, 32, 16, 8], 
                        help='layer_channels list of values on each of heads')
    parser.add_argument('--pre_conv_channels', type=int, nargs='+', 
                        default=[2,2,4], 
                        help='layer_channels list of values on each of heads')
    parser.add_argument('--pre_residuals', type=int, default=4, 
                        help='number of cnn heads')
    parser.add_argument('--up_residuals', type=int, default=0, 
                        help='number of cnn heads')
    parser.add_argument('--post_residuals', type=int, default=12, 
                        help='number of cnn heads')
    parser.add_argument('--early_stopping', type=int, default=100,  
                        help='early stopping after <early_stopping> times')  
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
    
    # Parse arguments
    args = parser.parse_args()

    if DEBUG ==  True:
        args = set_debug_data(args)

    args, wandb_flag = init(args)
    # Initialize model and optimizer
    model = get_model(args)
    # print and save model
    print_model_summary(args, model)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set train dataset and dataloader
    train_dataset = create_dataset(args, device, args.train_data_size)
    train_loader = prepare_data_loader(train_dataset, args)
    # set validation dataset and dataloader 
    val_dataset = create_dataset(args, device, args.val_data_size)
    val_loader = prepare_data_loader(val_dataset, args)
    # Initialize trainer
    trainer = Trainer(model=model, 
                      train_loader=train_loader, 
                      val_loader=val_loader, 
                      wandb_flag=wandb_flag,
                      device=device,
                      args=args)
    
    start_time = time.time()
    if (wandb_flag):
        wandb.login()
        wandb.init(project='GaussianBispectrumInversion',
                           name = f"{args.suffix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                           config=args)
        wandb.log({"cmd_line": sys.argv})
        wandb.watch(model, log_freq=100)
    # Train and evaluate
    trainer.run()
    end_time = time.time()
        
    print(f"Time taken to train in {os.path.basename(__file__)}:", 
          end_time - start_time, "seconds")


if __name__ == "__main__":
    main()