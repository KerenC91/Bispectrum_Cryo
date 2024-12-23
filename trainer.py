import os
import wandb
import torch 
from utils.utils import BispectrumCalculator, BatchAligneToReference, rand_shift_signal
from config.hparams import hparams
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group, all_reduce
import gc
import sys
import pdb

class Trainer:
    def __init__(self, model, 
                        train_loader, 
                        val_loader, 
                        train_dataset,
                        val_dataset,
                        wandb_flag,
                        device,
                        optimizer,
                        optimizer_name,
                        scheduler,
                        scheduler_name,
                        folder_matlab,
                        folder_python,
                        start_epoch,
                        args,
                        is_distributed=False):
        self.device = device 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset=train_dataset
        self.val_dataset=val_dataset
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.train_data_size = args.train_data_size
        self.val_data_size = args.val_data_size
        self.target_len = args.N
        self.signals_count = args.K
        self.save_every = args.save_every
        self.model = model.to(self.device)
        if is_distributed:
            self.model = DDP(self.model, device_ids=[self.device])#, 
                             # find_unused_parameters=True)
        self.wandb_flag = wandb_flag
        self.normalize = args.normalize
        self.mode = args.mode
        self.start_epoch = start_epoch
        self.epoch = 0
        self.last_loss = torch.inf
        self.early_stopping = args.early_stopping
        self.es_cnt = 0
        self.suffix = args.suffix
        self.n_heads = args.n_heads
        self.optimizer = optimizer
        self.optimizer_name = optimizer_name
        self.read_baseline = args.read_baseline
        self.scheduler = scheduler
        self.scheduler_name = scheduler_name
        self.loss_mode = args.loss_mode
        if self.loss_mode == 'all':
            self.loss_f = self._loss_all
        else:
            self.loss_f = self._loss
        self.bs_calc = BispectrumCalculator(self.signals_count, self.target_len, self.device).to(self.device)
        self.aligner = BatchAligneToReference(self.device).to(self.device)
        self.folder_matlab = folder_matlab
        self.folder_python = folder_python
        self.is_training = True
        self.loss_method = args.loss_method
        self.plotting_off = args.plotting_off
        self.clip = args.clip_grad_norm
        self.loss_criterion = args.loss_criterion
        self.is_master = (device == 0)
        self.debug = args.debug
        
    def _loss(self, pred, target):
        total_loss = 0.

        if self.loss_criterion == "sc":
            bs_pred, _ = self.bs_calc(pred)
            bs_target, _ = self.bs_calc(target)
            loss_sc = self._loss_sc(bs_pred, bs_target)
            total_loss = loss_sc
        elif self.loss_criterion == "l1":
            loss_l1_aligned = self._loss_l1(pred, target)
            total_loss = loss_l1_aligned  
        if self.loss_criterion == "mse":
            loss_mse_aligned = self._loss_MSE(pred, target)
            total_loss = loss_mse_aligned

        return total_loss
    
    def _switch_position(self, pred, target):
        switch = False
        if self.loss_criterion == "sc":
            bs_pred, pred = self.bs_calc(pred, "sum")
            bs_target, target = self.bs_calc(target, "sum")
            _, switch = self._switch_criterion(bs_pred, bs_target)
        elif self.loss_criterion == "l1":
            _, switch = self._switch_criterion_l1_aligned(pred, target)
        elif self.loss_criterion == "mse":
            _, switch = self._switch_criterion_mse_aligned(pred, target)
        if switch:
            pred = torch.flip(pred, dims=(-2,))
        
        return pred
    
    def _loss_all(self, pred, target):          
        total_loss = 0.
        
        if self.loss_criterion == "sc":
            bs_pred, pred = self.bs_calc(pred, self.loss_method)
            bs_target, target = self.bs_calc(target, self.loss_method)  
            loss_sc = self._loss_sc(bs_pred, bs_target, self.loss_method)
            total_loss = loss_sc
        elif self.loss_criterion == "l1":
            loss_l1_aligned = self._loss_l1(pred, target)
            total_loss = loss_l1_aligned 
        elif self.loss_criterion == "mse":
            loss_mse_aligned = self._loss_MSE(pred, target)
            total_loss = loss_mse_aligned

        loss = total_loss, \
                self._loss_MSE(pred, target), \
                self._loss_rel_MSE(pred, target)

        return loss
   
    def _loss_sc(self, bs_pred, bs_gt, method="average"):
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
            || BS(rec_s) - BS(s) ||_F / || BS(s) ||_F.

        """
        if method == "sum":
            sh = bs_pred.shape
            loss = torch.mean(
                        torch.norm((bs_pred - bs_gt).view(sh[0], sh[1], -1), dim=(0, 2))**2/ \
                            torch.norm(bs_gt.view(sh[0], sh[1], -1), dim=(0, 2))**2)
            if self.debug:
                if self.wandb_flag and \
                    (self.epoch == 1 or self.epoch % self.save_every == 0):
                    if (self.is_training):
                        wandb.log({"bs_pred_minus_bs_gt_norm_avg": torch.mean(torch.norm((bs_pred - bs_gt).view(sh[0], sh[1], -1), dim=(0, 2)))})
                        wandb.log({"bs_gt_norm_avg": torch.mean(torch.norm(bs_gt.view(sh[0], sh[1], -1), dim=(0, 2)))})
        else:
            loss = torch.norm(bs_pred - bs_gt)**2 / torch.norm(bs_gt)**2
            if self.debug:
                if self.wandb_flag and \
                    (self.epoch == 1 or self.epoch % self.save_every == 0):
                    if (self.is_training):
                        wandb.log({"bs_pred_minus_bs_gt_norm": torch.norm(bs_pred - bs_gt)})
                        wandb.log({"bs_gt_norm": torch.norm(bs_gt)})
        
        if self.debug:
            if self.wandb_flag and \
                (self.epoch == 1 or self.epoch % self.save_every == 0):
                if (self.is_training):
                    wandb.log({"bs_pred": torch.norm(bs_pred)})
                    wandb.log({"bs_gt": torch.norm(bs_gt)})
        return loss
    
    def _switch_criterion(self, bs_pred, bs_gt):
        # for sum method only
        sh = bs_pred.shape
        reversed_bs_pred = torch.flip(bs_pred, dims=(1,))
        loss1 = torch.mean(
                    torch.norm((bs_pred - bs_gt).view(sh[0], sh[1], -1), dim=(0, 2))**2 / \
                        torch.norm(bs_gt.view(sh[0], sh[1], -1), dim=(0, 2)))**2
        loss2 = torch.mean(
                    torch.norm((reversed_bs_pred - bs_gt).view(sh[0], sh[1], -1), dim=(0, 2))**2 / \
                        torch.norm(bs_gt.view(sh[0], sh[1], -1), dim=(0, 2)))**2
        # get the index for the minimal loss
        i = np.argmin(np.array([loss1.item(), loss2.item()]))
        # get the minimal loss
        loss = torch.min(loss1, loss2)
        switch = (i != 0)
        
        return loss, switch
    
    def _switch_criterion_l1_aligned(self, pred, target):
        # for sum method only
        sh = pred.shape
        reversed_pred = torch.flip(pred, dims=(1,))
        
        pred, _ = self.aligner(pred, target)
        loss1 = self._loss_l1(pred, target)
        reversed_pred, _ = self.aligner(reversed_pred, target)
        loss2 = self._loss_l1(reversed_pred, target)
        # get the index for the minimal loss
        i = np.argmin(np.array([loss1.item(), loss2.item()]))
        # get the minimal loss
        loss = torch.min(loss1, loss2)
        switch = (i != 0)
        
        return loss, switch

    def _switch_criterion_mse_aligned(self, pred, target):
        # for sum method only
        sh = pred.shape
        reversed_pred = torch.flip(pred, dims=(1,))
        
        pred, _ = self.aligner(pred, target)
        loss1 = self._loss_MSE(pred, target)
        reversed_pred, _ = self.aligner(reversed_pred, target)
        loss2 = self._loss_MSE(reversed_pred, target)
        # get the index for the minimal loss
        i = np.argmin(np.array([loss1.item(), loss2.item()]))
        # get the minimal loss
        loss = torch.min(loss1, loss2)
        switch = (i != 0)
        
        return loss, switch
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
                    torch.norm(pred - target, dim=(0, 2))**2 / \
                    torch.norm(target, dim=(0, 2))**2)

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
        criterion = torch.nn.L1Loss()          

        return criterion(pred, target)
    
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
        if self.debug:
            if self.wandb_flag and \
                (self.epoch == 1 or self.epoch % self.save_every == 0):
                if (self.is_training):
                    wandb.log({"pred": torch.norm(pred)})
                    wandb.log({"gt": torch.norm(target)})
        criterion = torch.nn.MSELoss()  
        
        return criterion(pred, target)
        
    def _run_batch(self, source, target):
        # Move data to device
        target = target.to(self.device)
        source = source.to(self.device)

        # Forward pass
        output = self.model(source) # reconstructed signal
        #if (not self.is_training) or (self.is_training and self.loss_method == 'sum'):
        output = self._switch_position(output, target)
        # if not self.is_training:
        #     output, _ = self.aligner(output, target)
             
        # Loss calculation

        loss = self.loss_f(output, target)

        return loss
        
    def _run_batch_rand(self):#only in tarining
        if self.mode[1] == 'circular_shifts':
            y = torch.randn(self.target_len)
            circulant = lambda v: torch.cat([f := v, f[:-1]]).unfold(0, len(v), 1).flip(0)
            target = circulant(torch.roll(y, -1))
            target = target.unsqueeze(1)
        else:
            target = torch.randn(self.batch_size, self.signals_count, self.target_len)
        if self.normalize:
            y = torch.fft.fft(target, dim=-1)
            y /= torch.norm(y, dim=-1).unsqueeze(2)
            target = torch.fft.ifft(y, dim=-1) 
            target = target.type(torch.float32)
        source, target = self.bs_calc(target)

        if self.mode[1] == 'shift':
            target, shifts = rand_shift_signal(target, 
                                                self.target_len, 
                                                self.batch_size)
        
        # Move data to device
        target = target.to(self.device)
        source = source.to(self.device)
        # Forward pass
        output = self.model(source) # reconstructed signal
        #if self.loss_method == 'sum':
        output = self._switch_position(output, target)
        
        # Loss calculation
        loss = self.loss_f(output, target)
        return loss
            
    def plot_output_debug(self, target, output, folder, from_matlab=None):
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig_path = f'{folder}/x_vs_x_rec.png'     
      
        plt.figure()
        plt.title('Comparison between original signal and its reconstructions')
        plt.plot(output, label='tested', color='tab:orange')
        if from_matlab is not None:
            plt.plot(from_matlab, label='baseline', color='tab:green')
        plt.plot(target, label='org', color='tab:blue')
        plt.ylabel('signal')
        plt.xlabel('time')
        plt.legend()
        plt.savefig(fig_path)        
        plt.close()
            
    def _save_checkpoint(self):
        if not os.path.exists(self.folder_python):
            os.makedirs(self.folder_python)
        if self.scheduler_name != 'None':    
            torch.save({'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()}, 
                f'{self.folder_python}/ckp.pt')
        else:
            torch.save({'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()},
                f'{self.folder_python}/ckp.pt')    
        
    def _run_epoch_train(self):
        total_loss = 0

        for idx, (sources, targets) in self.train_loader:
            # zero grads
            self.optimizer.zero_grad()
            # forward pass + loss computation
            if self.mode[0] == 'opt':
                loss = self._run_batch(sources, targets)
            else:#if self.mode[0] == 'rand': 
                loss = self._run_batch_rand()
            # backward pass
            loss.backward()
            # clip gradients
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            # optimizer step
            self.optimizer.step()
            # update avg loss 
            total_loss += loss.item()
            # scheduler step after batch
            if self.scheduler_name != 'None':
                if self.scheduler_name in ['OneCycleLR', 'CosineAnnealingLR', 'CyclicLR']:
                    self.scheduler.step()
            
        avg_loss = total_loss / len(self.train_loader)
        
        # scheduler step after epoch
        if self.scheduler_name != 'None':
            if self.scheduler_name == 'Manual':
                if self.epoch in hparams.manual_epochs_lr_change:
                    self.optimizer.param_groups[0]['lr'] *= hparams.manual_lr_f 
            elif self.scheduler_name == 'StepLR':
                self.scheduler.step()
            elif self.scheduler_name == 'ReduceLROnPlateau':
                self.scheduler.step(avg_loss)
                
        return avg_loss

    def _run_epoch_train_losses_all(self):   
        total_loss = 0
        total_mse_loss = 0
        total_mse_norm_loss = 0
        
        for idx, (sources, targets) in self.train_loader:
            # zero grads
            self.optimizer.zero_grad()
            # forward pass + loss computation
            if self.mode[0] == 'opt':
                loss, mse_loss, rel_mse_loss = self._run_batch(sources, targets)
            else:#if self.mode[0] == 'rand': 
                # pdb.set_trace()
                loss, mse_loss, rel_mse_loss = self._run_batch_rand()
            # backward pass
            loss.backward()
            # clip gradients
            if self.clip:    
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            # optimizer step
            self.optimizer.step()
            torch.cuda.empty_cache()
            # update avg loss 
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_mse_norm_loss += rel_mse_loss.item()
            # scheduler step after batch
            if self.scheduler_name != 'None':
                if self.scheduler_name in ['OneCycleLR', 'CosineAnnealingLR', 'CyclicLR']:
                    self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_mse_loss = total_mse_loss / len(self.train_loader) 
        avg_mse_norm_loss = total_mse_norm_loss / len(self.train_loader) 

        # scheduler step after epoch
        if self.scheduler_name != 'None':
            if self.scheduler_name == 'Manual':
                if self.epoch in hparams.manual_epochs_lr_change:
                    self.optimizer.param_groups[0]['lr'] *= hparams.manual_lr_f 
            elif self.scheduler_name == 'StepLR':
                self.scheduler.step()
            elif self.scheduler_name == 'ReduceLROnPlateau':
                self.scheduler.step(avg_loss)
                
        return avg_loss, avg_mse_loss, avg_mse_norm_loss        

    def _run_epoch_validate_losses_all(self):   
        total_loss = 0
        total_mse_loss = 0
        total_mse_norm_loss = 0
        
        for idx, (sources, targets) in self.val_loader:
            with torch.no_grad():
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
    
    

    
    def _run_epoch_validate(self):
        total_loss = 0
        
        for idx, (sources, targets) in self.val_loader:
            with torch.no_grad():
                # forward pass + loss computation
                loss = self._run_batch(sources, targets)

                # update avg loss 
                total_loss += loss.item()
            
        avg_loss = total_loss / len(self.val_loader)
            
        return avg_loss
    

    # one epoch of training           
    def train(self):
        # Set the model to training mode
        self.model.train()
        self.is_training = True
        
        if self.loss_mode == 'all':
            avg_loss = self._run_epoch_train_losses_all()
        else:
            avg_loss = self._run_epoch_train()
            
        return avg_loss
    
    # one epoch of validation           
    def validate(self):
        # Set the model to evaluation mode
        self.model.eval()
        self.is_training = False
        
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
            pred = self.model(source) # reconstructed signal
            pred = self._switch_position(pred, target)
            pred, _ = self.aligner(pred, target)
                
            self.save_python_test_data(idx.item(), pred, target)
            
    def save_python_test_data(self, i, x_est, x_true):
        folder = os.path.join(self.folder_python, f'sample{i+1}')
        if not os.path.exists(folder):
            os.mkdir(folder)
        #read from matlab
        folder_m = os.path.join(self.folder_matlab, f'sample{i}')    
        rel_error_X = self._loss_rel_MSE(x_est, x_true).item()
        rel_error_X_path = os.path.join(folder, 'rel_error_X.csv')
        np.savetxt(rel_error_X_path, [rel_error_X])
        for k in range(self.signals_count):
            folder_k = os.path.join(folder, f'{k+1}')
            if not os.path.exists(folder_k):
                os.mkdir(folder_k)

            x_est_path = os.path.join(folder_k, f'x_est.csv')
            np.savetxt(x_est_path, 
                       x_est.squeeze(0)[k].cpu().detach().numpy())
            
            x_true_path = os.path.join(folder_k, f'x_true.csv')
            np.savetxt(x_true_path, 
                       x_true.squeeze(0)[k].cpu().detach().numpy())
            
            file_path = os.path.join(folder_m, f'x_est_{k+1}.csv')
            x_est_m = np.loadtxt(file_path, delimiter=" ")
            #save figure
            self.plot_output_debug(x_true.squeeze(0)[k].cpu().detach().numpy(), 
                                   x_est.squeeze(0)[k].cpu().detach().numpy(),
                                   folder_k,
                                   x_est_m)

        
    def run(self):
        for self.epoch in range(self.start_epoch + 1, self.epochs + 1):
            # train             
            train_loss = self.train()
            # validate
            val_loss = self.validate()

            if self.loss_mode == 'all':
                train_loss, train_mse_loss, train_rel_mse_loss = train_loss
                val_loss, val_mse_loss, val_rel_mse_loss = val_loss
            # update lr
            last_lr = self.optimizer.param_groups[0]['lr']

            if self.is_master:
                # log loss with wandb
                if self.wandb_flag and \
                    (self.epoch == 1 or self.epoch % self.save_every == 0):
                    wandb.log({"train_loss": train_loss})
                    wandb.log({"val_loss": val_loss})
                    wandb.log({"lr": self.optimizer.param_groups[0]['lr']})
                    if self.loss_mode == 'all':
                        wandb.log({"train mse": train_mse_loss})
                        wandb.log({"train relative mse": train_rel_mse_loss})
                        wandb.log({"val mse": val_mse_loss})
                        wandb.log({"val relative mse": val_rel_mse_loss})
                # save checkpoint and log loss to cmd 
                if self.epoch == 1 or self.epoch % self.save_every == 0:
                    print(f'-------Epoch {self.epoch}/{self.epochs}-------')
                    print(f'Total Train loss: {train_loss:.6f}')
                    print(f'Total Validation loss: {val_loss:.6f}')
                    if self.loss_mode == 'all':
                        print(f'train mse loss: {train_mse_loss:.6f}')
                        print(f'train relative mse loss: {train_rel_mse_loss:.6f}')
                        print(f'val mse loss: {val_mse_loss:.6f}')
                        print(f'val relative mse loss: {val_rel_mse_loss:.6f}')
                    if self.scheduler_name != 'None':
                        print(f'lr: {last_lr}')
                    # save checkpoint
                    self._save_checkpoint()
                # plot outputs on last epoch
                if self.epoch == self.epochs and self.plotting_off == False:
                    if self.read_baseline != 0:
                        if self.read_baseline == 1: # train
                            self.write_python_test_results(self.train_dataset)
                        elif self.read_baseline == 2:
                            self.write_python_test_results(self.val_dataset)

            # stop early if early_stopping is on
            if self.early_stopping:
                if self.last_loss < train_loss:
                    self.es_cnt +=1
                    if self.es_cnt == hparams.early_stopping:
                        if self.is_master:
                            print(f'Stooped at epoch {self.epoch}, after {self.es_cnt} times\n'
                                  f'last_loss={self.last_loss}, curr_los={train_loss}')
                        folder = f'figures/cnn_{self.suffix}'
                        break
            # stop if loss has reached lower bound
            if self.loss_mode == 'all' and train_mse_loss < hparams.loss_lim:
                if self.is_master:
                    print(f'-------Epoch {self.epoch}/{self.epochs}-------')
                    print(f'Total Train loss: {train_loss:.6f}')
                    print(f'Total Validation loss: {val_loss:.6f}')
                    if self.loss_mode == 'all':
                        print(f'train mse loss: {train_mse_loss:.6f}')
                        print(f'train relative mse loss: {train_rel_mse_loss:.6f}')
                        print(f'val mse loss: {val_mse_loss:.6f}')
                        print(f'val relative mse loss: {val_rel_mse_loss:.6f}')
                    if self.scheduler_name != 'None':
                        print(f'lr: {last_lr}')
    
                    print(f'Stooped at epoch {self.epoch},\n'
                          f'curr_los={train_loss} < {hparams.loss_lim}')    
                    self.last_loss = train_loss
                break
        
        # test
        with torch.no_grad():
            test_loss = self.test()
            if self.is_master:
                print(f'Test loss l1: {test_loss:.6f}')
