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
import matplotlib.pyplot as plt
import numpy as np

import sys

from torch import nn
from compare_to_baseline import read_tensor_from_matlab
# Set the same seed for reproducibility
#torch.manual_seed(1234)

class Trainer:
    def __init__(self, model, 
                 train_loader, 
                 val_loader, 
                 train_dataset,
                 val_dataset,
                 wandb_flag,
                 device,
                 optimizer,
                 scheduler,
                 scheduler_name,
                 comp_baseline_folders,
                 args):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset=train_dataset
        self.val_dataset=val_dataset
        self.batch_size = args.batch_size
        self.epochs = args.epochs
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
        self.optimizer = optimizer
        self.read_baseline = args.read_baseline
        self.scheduler = scheduler
        self.scheduler_name = scheduler_name
        self.loss_mode = args.loss_mode
        if self.loss_mode == 'all':
            self.loss_f = self._loss_all
        else:
            self.loss_f = self._loss
        self.bs_calc = BispectrumCalculator(self.batch_size, self.target_len, self.device).to(self.device)
        self.folder_test, self.folder_matlab, self.folder_python = \
                        comp_baseline_folders
    
    def _loss(self, pred, target):
        bs_pred, _ = self.bs_calc(pred)
        bs_target, _ = self.bs_calc(target)
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
        self.last_target = target
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

    def plot_output_debug(self, target, output, folder, from_matlab=None):
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig_path = f'{folder}/x_vs_x_rec.png'     
      
        plt.figure()
        plt.title('Comparison between original signal and its reconstructions')
        plt.plot(target, label='org')
        plt.plot(output, label='tested')
        if from_matlab is not None:
            plt.plot(from_matlab, label='baseline')
        plt.ylabel('signal')
        plt.xlabel('time')
        plt.legend()
        plt.savefig(fig_path)        
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

    def _run_epoch_validate_losses_all(self):   
        total_loss = 0
        total_mse_loss = 0
        total_mse_norm_loss = 0
        
        for idx, (sources, targets) in self.train_loader:
            # forward pass + loss computation
            loss, mse_loss, rel_mse_loss = self._run_batch(sources, targets)
            # update avg loss 
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_mse_norm_loss += rel_mse_loss.item()
            
        avg_loss = total_loss / len(self.val_loader)
        avg_mse_loss = total_mse_loss / len(self.val_loader) 
        avg_mse_norm_loss = total_mse_norm_loss / len(self.val_loader) 

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
            
        avg_loss = total_loss / len(self.val_loader)
            
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
        
        if self.loss_mode == 'all':
            avg_loss = self._run_epoch_validate_losses_all()
        else:
            avg_loss = self._run_epoch_validate()
            
        return avg_loss

    # one epoch of testing 
    def test(self):
        return 0
    def write_python_test_results(self, dataset):#changedataloader
        dataloader = DataLoader(
                            dataset,
                            batch_size=1,
                            pin_memory=False,
                            shuffle=False
                        )
        for idx, (source, target) in dataloader:
            # Move data to device
            target = target.to(self.device)
            source = source.to(self.device)
            # Forward pass
            _, output = self.model(source) # reconstructed signal
            self.save_python_test_data(idx.item(), output, target)
            
    def save_python_test_data(self, i, x_est, x_true):
        folder = os.path.join(self.folder_python, f'sample{i}')
        if not os.path.exists(folder):
            os.mkdir(folder)
            
        rel_error_X = self._loss_rel_MSE(x_est, x_true).item()
        rel_error_X_path = os.path.join(folder, 'rel_error_X.csv')
        np.savetxt(rel_error_X_path, [rel_error_X])
        
        x_est_path = os.path.join(folder, 'x_est.csv')
        np.savetxt(x_est_path, 
                   x_est.squeeze(0).squeeze(0).cpu().detach().numpy())
        
        x_true_path = os.path.join(folder, 'x_true.csv')
        np.savetxt(x_true_path, 
                   x_true.squeeze(0).squeeze(0).cpu().detach().numpy())
        
        #read from matlab
        folder_m = os.path.join(self.folder_matlab, f'sample{i}')
        file_path = os.path.join(folder_m, 'x_est.csv')
        x_est_m = np.loadtxt(file_path, delimiter=" ")
        #save figure
        self.plot_output_debug(x_true.squeeze(0).squeeze(0).cpu().detach().numpy(), 
                               x_est.squeeze(0).squeeze(0).cpu().detach().numpy(),
                               folder,
                               x_est_m)

        
    def run(self):
        for self.epoch in range(self.epochs):
            # train             
            train_loss = self.train()
            # validate
            val_loss = self.validate()
            # test
            test_loss = self.test()
            if self.loss_mode == 'all':
                train_loss, train_mse_loss, train_rel_mse_loss = train_loss
                val_loss, val_mse_loss, val_rel_mse_loss = val_loss
            # update lr
            
            if self.scheduler_name != 'None':
                last_lr = self.optimizer.param_groups[0]['lr']
                if self.scheduler_name == 'Manual':
                    if self.epoch in hparams.manual_epochs_lr_change:
                        self.optimizer.param_groups[0]['lr'] *= hparams.manual_lr_f 
                elif self.scheduler_name == 'ReduceLROnPlateau':
                    self.scheduler.step(train_loss)
                elif self.scheduler_name == 'StepLR':
                    self.scheduler.step()
            # log loss with wandb
            if self.wandb_flag and self.epoch % self.wandb_log_interval == 0:
                wandb.log({"train_loss_l1": train_loss})
                wandb.log({"val_loss_l1": val_loss})
                wandb.log({"test_loss": test_loss})
                wandb.log({"lr": self.optimizer.param_groups[0]['lr']})
                if self.loss_mode == 'all':
                    wandb.log({"train mse": train_mse_loss})
                    wandb.log({"train relative mse": train_rel_mse_loss})
                    wandb.log({"val mse": val_mse_loss})
                    wandb.log({"val relative mse": val_rel_mse_loss})
            # save checkpoint and log loss to cmd 
            if self.epoch % self.save_every == 0:
                print(f'-------Epoch {self.epoch}/{self.epochs}-------')
                print(f'Train loss l1: {train_loss:.6f}')
                print(f'Validation loss l1: {val_loss:.6f}')
                if self.loss_mode == 'all':
                    print(f'train mse loss: {train_mse_loss:.6f}')
                    print(f'train relative mse loss: {train_rel_mse_loss:.6f}')
                    print(f'val mse loss: {val_mse_loss:.6f}')
                    print(f'val relative mse loss: {val_rel_mse_loss:.6f}')
                if self.scheduler_name != 'None':
                    print(f'lr: {last_lr}')
                # save checkpoint
                self._save_checkpoint(self.epoch)
            # plot last output
            if self.epoch == self.epochs - 1:
                folder = f'figures/cnn_{self.suffix}'
                self.plot_output_debug(self.last_target[0].squeeze(0).detach().cpu().numpy(), 
                                       self.last_output[0].squeeze(0).detach().cpu().numpy(),
                                       folder)
                if self.read_baseline != 0:
                    if self.read_baseline == 1: # train
                        self.write_python_test_results(self.train_dataset)
                    elif self.read_baseline == 2:
                        self.write_python_test_results(self.val_dataset)
                    # with open(hparams.py_x_rec_file, "w", newline="") as csvfile:
                    #     # Create a CSV writer object
                    #     writer = csv.writer(csvfile)
                    
                    #     # Write the data to the file
                    #     writer.writerows(self.last_output[0])
            # stop early if early_stopping is on
            if self.early_stopping != 0:
                if self.last_loss < train_loss:
                    self.es_cnt +=1
                    if self.es_cnt == self.early_stopping:
                        print(f'Stooped at epoch {self.epoch}, after {self.es_cnt} times\n'
                              f'last_loss={self.last_loss}, curr_los={train_loss}')
                        folder = f'figures/cnn_{self.suffix}'
                        self.plot_output_debug(self.last_target[0].squeeze(0).detach().cpu().numpy(),
                                               self.last_output[0].squeeze(0).detach().cpu().numpy(), 
                                               folder)
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
    
def create_dataset(device, data_size, N, read_baseline, mode, 
                   comp_baseline_folders):
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
                sample_path = os.path.join(folder, 'x_true.csv')
                target[i] = read_tensor_from_matlab(sample_path, True)              
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
        
def get_model(args):
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

def set_debug_data(args):
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
    args.read_baseline = 1
    
    return args
    
    
def prepare_data_loader(dataset, args):
    dataloader = None
    
    if args.mode =='opt':
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
        folder_testm = os.path.join(hparams.comp_root, hparams.comp_test_name)
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
    
    return wandb_flag, (folder_test, folder_matlab, folder_python)

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
                                      eps=hparams.opt_eps,
                                      weight_decay=hparams.opt_adam_w_weight_decay)
    else: # Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
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
            
def main():
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
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Parse arguments
    args = parser.parse_args()
    #hparams = set_hparams(args.config_mode)
    DEBUG = hparams.DEBUG

    if DEBUG ==  True:
        args = set_debug_data(args)

    args = update_suffix(args, DEBUG)
    wandb_flag, comp_baseline_folders = init(args)
    # Initialize model and optimizer
    model = get_model(args)
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
    # Train and evaluate
    trainer.run()
    if wandb_flag:
        folder = f'figures/cnn_{args.suffix}'
        fig_path = f'{folder}/x_vs_x_rec.png'
        #wandb.upload_file(fig_path, f"x_vs_x_rec_ep{args.epochs - 1}.png")
        artifact = wandb.Artifact("x_vs_x_rec", type="figure")
        artifact.add_file(fig_path, 
                          name=f"x_vs_x_rec.png")
        run.log_artifact(artifact)
    end_time = time.time()
        
    print(f"Time taken to train in {os.path.basename(__file__)}:", 
          end_time - start_time, "seconds")


if __name__ == "__main__":
    main()