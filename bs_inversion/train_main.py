import torch.optim as optim
import time 
import os
import wandb
from datetime import datetime
import torch 
from torch.utils.data import Dataset, DataLoader
import argparse
from utils import calculate_bispectrum_power_spectrum_efficient
from model import CNNBS, CNNBSSingleHead
from hparams import hparams, hparams_debug_string
import matplotlib.pyplot as plt
import numpy as np

DEBUG = True

class Trainer:
    def __init__(self, model, 
                 train_loader, 
                 val_loader, 
                 batch_size, 
                 optimizer,
                 wandb_flag,
                 args):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.num_epochs = args.epochs
        self.optimizer = optimizer
        self.wandb_log_interval = args.wandb_log_interval
        self.train_data_size = args.train_data_size
        self.val_data_size = args.val_data_size
        self.norm_factor = self.train_data_size
        self.target_len = args.N
        self.save_every = args.save_every
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model = model.to(self.device)
        self.wandb_flag = wandb_flag
        self.mode = args.mode
        if self.mode == 'rand':
            self.batch_size = 1
        elif self.mode == 'opt':
            self.single_target = hparams.fixed_sample[:self.target_len]
            self.single_source, self.single_target = self.create_rand_data(target=self.single_target)
            self.train_data_size = 1
        self.epoch = 0
        self.last_loss = torch.inf
        self.early_stopping = args.early_stopping
        self.es_cnt = 0
        self.suffix = args.suffix
    
    def _loss(self, pred, target):
        bs_pred, _, _ = calculate_bispectrum_power_spectrum_efficient(pred)
        bs_target, _, _ = calculate_bispectrum_power_spectrum_efficient(target)
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
            || s - rec_s ||_F / || s ||_F.

        """
        return torch.norm(target - pred) / torch.norm(target)
        # target - ground truth image, source - Bispectrum of ground truth image
        # might be multiple targets and sources (batch size > 1)
        
    def _run_batch(self, source, target):
        # Move data to device
        target = target.to(self.device)
        source = source.to(self.device)
        # Forward pass
        output = self.model(source) # reconstructed image
        # Loss calculation
        loss = self._loss(output, target)
        return loss

    def create_rand_data(self, target):
        # Create data
        bs, ps, f = calculate_bispectrum_power_spectrum_efficient(target)
        bs_real = bs.real.float()
        bs_imag = bs.imag.float()
        source = torch.stack([bs_real, bs_imag], dim=-1)
        source = source.permute(2, 0, 1)
        target = target.unsqueeze(0)   
        
        return source, target
        
    def _run_batch_rand(self):
        if self.mode == 'rand':
            source, target = self.create_rand_data(target=torch.randn(self.target_len)  )
        elif self.mode == 'opt':
            source, target = self.single_source, self.single_target
        else:
            print(f'error! unknown mode {self.mode}')
        # Move data to device
        target = target.to(self.device)
        source = source.to(self.device)
        # Forward pass
        _, output = self.model(source) # reconstructed signal
        
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
            loss = self._run_batch()
            # backward pass
            loss.backward()
            # optimizer step
            self.optimizer.step()
            # update avg loss 
            total_loss += loss.item()
            
            
        avg_loss = total_loss / self.train_data_size
        
        return avg_loss
    
    def _run_epoch_train_rand(self):
        total_loss = 0
                
        for _ in range(self.train_data_size):
            # zero grads
            self.optimizer.zero_grad()
            # forward pass + loss computation
            loss = self._run_batch_rand()
            # backward pass
            loss.backward()
            # optimizer step
            self.optimizer.step()
            # update avg loss 
            total_loss += loss.item()
            
        avg_loss = total_loss / self.train_data_size
            
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
                # update avg loss 
                total_loss += loss.item()
            
        avg_loss = total_loss / self.val_data_size
            
        return avg_loss
               
    def train(self):
        # Set the model to training mode
        self.model.train()
        
        if self.mode == 'rand' or self.mode == 'opt':
            avg_loss = self._run_epoch_train_rand()
        else:
            avg_loss = self._run_epoch_train()
        
        return avg_loss
              
    def validate(self):
        # Set the model to evaluation mode
        self.model.eval()
        
        if self.mode == 'rand':
            avg_loss = self._run_epoch_validate_rand()
        else:
            avg_loss = self._run_epoch_validate()
        return avg_loss

    def test(self):
        return 0
                    
    def run(self):
        for self.epoch in range(self.num_epochs):                
            train_loss = self.train()
            if self.mode != 'opt':
                val_loss = self.validate()
                test_loss = self.test()
            
            if self.wandb_flag and self.epoch % self.wandb_log_interval == 0:
                wandb.log({"train_loss": train_loss})
                if self.mode != 'opt':
                    wandb.log({"val_loss": val_loss})
                    #wandb.log({"test_loss": test_loss})
            if self.epoch % self.save_every == 0:
                print(f'-------Epoch {self.epoch}/{self.num_epochs}-------')
                print(f'Train loss: {train_loss:.3f}')
                if self.mode != 'opt':
                    print(f'Validation loss: {val_loss:.3f}')

                self._save_checkpoint(self.epoch)
            
            if self.early_stopping != 0:
                if self.last_loss < train_loss:
                    self.es_cnt +=1
                    if self.es_cnt == self.early_stopping:
                        print(f'Stooped at epoch {self.epoch}, after {self.es_cnt} times\n'
                              f'last_loss={self.last_loss}, curr_los={train_loss}')
                        return
            if train_loss < hparams.loss_lim:
                print(f'Stooped at epoch {self.epoch},\n'
                      f'curr_los={train_loss} < {hparams.loss_lim}')
    
                self.last_loss = train_loss
                
      
def main(debug_args=None):
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
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
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
    #evaluates to False if not provided, else True
    parser.add_argument('--wandb', action='store_true', 
                        help='Log data using wandb')     
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
    #evaluates to True if not provided, else False
    parser.add_argument('--early_stopping', type=int, default=100,  
                        help='early stopping after <early_stopping> times')  
    
    
    # Parse arguments
    args = parser.parse_args()

    if DEBUG ==  True:
        args.N = 100
        args.channels = [1600, 8]
        args.pre_conv_channels = [2, 2, 4, 8]
        args.pre_residuals = 5 
        args.up_residuals = 4 
        args.post_residuals = 1
    # Set wandb flag
    wandb_flag = args.wandb
    if (args.wandb_log_interval == 0):
        wandb_flag = False
    
    # Initialize model and optimizer
    model = CNNBS(
        input_len=args.N,
        n_heads=1,
        channels=args.channels,
        pre_conv_channels=args.pre_conv_channels,
        pre_residuals=args.pre_residuals,
        up_residuals=args.up_residuals,
        post_residuals=args.post_residuals
        )
    # Get model summary as a string
    print(f'input length: {args.N}')
    print(f'model architecture:\n'
          f'n_heads={args.n_heads}\n'
          f'channels={args.channels}\n'
          f'pre_conv_channels={args.pre_conv_channels}\n'
          f'pre_residuals={args.pre_residuals}\n'
          f'up_residuals={args.up_residuals}\n'
          f'post_residuals={args.post_residuals}\n'
          f'early_stopping: {args.early_stopping}\n')
    
    summary = str(model)
    
    # Save summary to file
    if not os.path.exists('models'):
        os.makedirs('models')
    with open(f'models/cnn_{args.suffix}.yml', "w") as f:
        f.write(summary)
    print(f'CNN arch yml: models/cnn_{args.suffix}.yml')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Initialize trainer
    # If train_loader and test_loader are set to None, 
    # the data is created during training
    trainer = Trainer(model=model, 
                      train_loader=None, 
                      val_loader=None, 
                      batch_size=1,                     
                      optimizer=optimizer,
                      wandb_flag=wandb_flag,
                      args=args)
    
    start_time = time.time()
    if (wandb_flag):
        wandb.login()
        wandb.init(project='GaussianBispectrumInversion',
                           name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                           config=args)
        wandb.watch(model, log_freq=100)
    # Train and evaluate
    trainer.run()
    end_time = time.time()
    if args.mode == 'opt':
        target = hparams.fixed_sample
        source, target = trainer.create_rand_data(target)
        # Move data to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        target = target.to(device)
        source = source.to(device)
        # Forward pass
        _, output = trainer.model(source)
        trainer.plot_output_debug(target, output)
        
    print(f"Time taken to train in {os.path.basename(__file__)}:", 
          end_time - start_time, "seconds")


if __name__ == "__main__":
    main()