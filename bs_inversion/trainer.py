import os
import wandb
import torch 
from utils import BispectrumCalculator
from hparams import hparams
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group, all_reduce


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
        self.device = device 
        self.nprocs = args.nprocs
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
        self.model = model.to(device)
        self.model = DDP(self.model, device_ids=[self.device], 
                         find_unused_parameters=True)
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
        self.bs_calc = BispectrumCalculator(self.target_len, self.device).to(self.device)
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
        output = self.model(source) # reconstructed signal
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
        output = self.model(source) # reconstructed signal
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
        ckp = self.model.module.state_dict()
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
            total_loss += loss
            
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss

    def _run_epoch_train_losses_all(self):   
        total_loss = 0
        total_mse_loss = 0
        total_mse_norm_loss = 0
        
        b_sz = len(next(iter(self.train_loader))[0])
        
        # print(f'GPU{self.device}: b_sz={b_sz}')
        for idx, (sources, targets) in self.train_loader:
            # print(f'GPU{self.device}: sources.shape={sources.shape}')
            # print(f'GPU{self.device}: targets.shape={targets.shape}')
            # print(f'GPU{self.device}: idx={idx}')

            # zero grads
            self.optimizer.zero_grad()
            # forward pass + loss computation
            loss, mse_loss, rel_mse_loss = self._run_batch(sources, targets)
            # backward pass
            loss.backward()
            # optimizer step
            self.optimizer.step()
            # update avg loss 
            total_loss += loss
            total_mse_loss += mse_loss
            total_mse_norm_loss += rel_mse_loss
            
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
            total_loss += loss
            total_mse_loss += mse_loss
            total_mse_norm_loss += rel_mse_loss
            
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
            total_loss += loss
            if self.loss_mode == 'all':
                total_mse_loss += mse_loss
                total_mse_norm_loss += rel_mse_loss
            
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
                total_loss += loss
            
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
                total_loss += loss
                
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
            output = self.model(source) # reconstructed signal
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
            self.train_loader.sampler.set_epoch(self.epoch)
            train_loss = self.train()
            # validate
            self.val_loader.sampler.set_epoch(self.epoch)
            val_loss = self.validate()


            if self.loss_mode == 'all':
                train_loss, train_mse_loss, train_rel_mse_loss = train_loss
                val_loss, val_mse_loss, val_rel_mse_loss = val_loss
                # Get loss from all processes
                all_reduce(train_loss, op=dist.ReduceOp.SUM)
                all_reduce(train_mse_loss, op=dist.ReduceOp.SUM)
                all_reduce(train_rel_mse_loss, op=dist.ReduceOp.SUM)
                
                all_reduce(val_loss, op=dist.ReduceOp.SUM)
                all_reduce(val_mse_loss, op=dist.ReduceOp.SUM)
                all_reduce(val_rel_mse_loss, op=dist.ReduceOp.SUM)
            else:
                # Get loss from all processes
                all_reduce(train_loss, op=dist.ReduceOp.SUM)
                all_reduce(val_loss, op=dist.ReduceOp.SUM) 

                
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
            # Only gpu 0 operating now...
            if self.device == 0: 
                # update losses
                if self.loss_mode == 'all':
                    # Get loss from all processes
                    train_loss /= self.nprocs
                    train_mse_loss /= self.nprocs
                    train_rel_mse_loss /= self.nprocs
                    
                    val_loss /= self.nprocs
                    val_mse_loss /= self.nprocs
                    val_rel_mse_loss /= self.nprocs
                else:
                    # Get loss from all processes
                    train_loss /= self.nprocs
                    val_loss /= self.nprocs
                # log loss with wandb
                if self.wandb_flag and self.epoch % self.wandb_log_interval == 0:
                    wandb.log({"train_loss_l1": train_loss.item()})
                    wandb.log({"val_loss_l1": val_loss.item()})
                    wandb.log({"lr": self.optimizer.param_groups[0]['lr']})
                    if self.loss_mode == 'all':
                        wandb.log({"train mse": train_mse_loss.item()})
                        wandb.log({"train relative mse": train_rel_mse_loss.item()})
                        wandb.log({"val mse": val_mse_loss.item()})
                        wandb.log({"val relative mse": val_rel_mse_loss.item()})
                # save checkpoint and log loss to cmd 
                if self.epoch % self.save_every == 0:
                    print(f'-------Epoch {self.epoch}/{self.epochs}-------')
                    print(f'Train loss l1: {train_loss.item():.6f}')
                    print(f'Validation loss l1: {val_loss.item():.6f}')
                    if self.loss_mode == 'all':
                        print(f'train mse loss: {train_mse_loss.item():.6f}')
                        print(f'train relative mse loss: {train_rel_mse_loss.item():.6f}')
                        print(f'val mse loss: {val_mse_loss.item():.6f}')
                        print(f'val relative mse loss: {val_rel_mse_loss.item():.6f}')
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
                                  f'last_loss={self.last_loss.item()}, curr_los={train_loss.item()}')
                            folder = f'figures/cnn_{self.suffix}'
                            self.plot_output_debug(self.last_target[0].squeeze(0).detach().cpu().numpy(),
                                                   self.last_output[0].squeeze(0).detach().cpu().numpy(), 
                                                   folder)
                            return
                # stop if loss has reached lower bound
                if train_loss.item() < hparams.loss_lim:
                    print(f'Stooped at epoch {self.epoch},\n'
                          f'curr_los={train_loss.item()} < {hparams.loss_lim}')    
                    self.last_loss = train_loss
        # test
        test_loss = self.test()
        print(f'Test loss l1: {test_loss.item():.6f}')