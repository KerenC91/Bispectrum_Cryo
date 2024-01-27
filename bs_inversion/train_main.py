import torch.optim as optim
import time 
import os
import wandb
from datetime import datetime
import torch 
from torch.utils.data import Dataset, DataLoader
import argparse
from utils import calculate_bispectrum_power_spectrum_efficient
from model import CNNBS
from hparams import hparams


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
        self.rand_data = args.rand_data
        if self.rand_data == True:
            self.batch_size = 1
        
    def _loss_sc(self, pred, target):
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
        bs_pred, _, _ = calculate_bispectrum_power_spectrum_efficient(pred)
        bs_gt, _, _ = calculate_bispectrum_power_spectrum_efficient(target)
        # Get magnitudes
        bs_pred_mag = torch.abs(bs_pred)
        bs_gt_mag = torch.abs(bs_gt)
        return torch.norm(bs_pred_mag - bs_gt_mag) / torch.norm(bs_gt_mag)
    # target - ground truth image, source - Bispectrum of ground truth image
    # might be multiple targets and sources (batch size > 1)

    def _loss_log_sc(self, pred, target, eps=1e-5):
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
        bs_pred, _, _ = calculate_bispectrum_power_spectrum_efficient(pred)
        bs_gt, _, _ = calculate_bispectrum_power_spectrum_efficient(target)
        # Get magnitudes
        bs_pred_mag = torch.abs(bs_pred)
        bs_gt_mag = torch.abs(bs_gt)
        return torch.norm(torch.log(bs_gt_mag + eps) - torch.log(bs_pred_mag + eps), p=1)
    # target - ground truth image, source - Bispectrum of ground truth image
    # might be multiple targets and sources (batch size > 1)

    def _loss_freq(self, pred, target):
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
        bs_pred, _, _ = calculate_bispectrum_power_spectrum_efficient(pred)
        bs_gt, _, _ = calculate_bispectrum_power_spectrum_efficient(target)
        # Get phases
        bs_pred_phase = torch.angle(bs_pred)
        bs_gt_phase = torch.angle(bs_gt)
        
        #Get derivative phase
        bs_pred_phase_deriv = bs_pred_phase[1:] - bs_pred_phase[:-1]
        bs_gt_phase_deriv = bs_gt_phase[1:] - bs_gt_phase[:-1]
        return torch.norm(bs_gt_phase_deriv - bs_pred_phase_deriv, p=1)
    # target - ground truth image, source - Bispectrum of ground truth image
    # might be multiple targets and sources (batch size > 1)


    def _loss_weighted_phase(self, pred, target):
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
        bs_pred, _, _ = calculate_bispectrum_power_spectrum_efficient(pred)
        bs_gt, _, _ = calculate_bispectrum_power_spectrum_efficient(target)
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
        loss = self._loss_l1(output, target)
        return loss

    def _run_batch_rand(self):
        # Create data
        target = torch.randn(self.target_len)     
        bs, ps, f = calculate_bispectrum_power_spectrum_efficient(target)
        bs_real = bs.real.float()
        bs_imag = bs.imag.float()
        source = torch.stack([bs_real, bs_imag], dim=-1)
        source = source.permute(2, 0, 1)
        target = target.unsqueeze(0)
        # Move data to device
        target = target.to(self.device)
        source = source.to(self.device)
        # Forward pass
        _, output = self.model(source) # reconstructed signal
        # Loss calculation
        loss = self._loss_l1(output, target)
        return loss

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = f"./checkpoints/checkpoint_ep{epoch}.pt"
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
        
        if self.rand_data:
            avg_loss = self._run_epoch_train_rand()
        else:
            avg_loss = self._run_epoch_train()
        
        return avg_loss
              
    def validate(self):
        # Set the model to evaluation mode
        self.model.eval()
        
        if self.rand_data:
            avg_loss = self._run_epoch_validate_rand()
        else:
            avg_loss = self._run_epoch_validate()
        return avg_loss

    def test(self):
        return 0
                    
    def run(self):
        for epoch in range(self.num_epochs):                
            train_loss = self.train()
            val_loss = self.validate()
            test_loss = self.test()
            
            if self.wandb_flag and epoch % self.wandb_log_interval == 0:
                wandb.log({"train_loss": train_loss})
                wandb.log({"val_loss": val_loss})
                #wandb.log({"test_loss": test_loss})
            if epoch % self.save_every == 0:
                print(f'-------Epoch {epoch}/{self.num_epochs}-------')
                print(f'Train loss: {train_loss:.3f}')
                print(f'Validation loss: {val_loss:.3f}')

                self._save_checkpoint(epoch)
      
def main():
    # Add arguments to parser
    parser = argparse.ArgumentParser(description='Inverting the bispectrum. Pulse dataset')

    parser.add_argument('--N', type=int, default=100, metavar='N',
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
    #evaluates to True if not provided, else False
    parser.add_argument('--rand_data', action='store_false',
            help='Create random data during training if True.')  
    #evaluates to False if not provided, else True
    parser.add_argument('--wandb', action='store_true', 
                        help='Log data using wandb')     
    
    # Parse arguments
    args = parser.parse_args()

    # Set wandb flag
    wandb_flag = args.wandb
    if (args.wandb_log_interval == 0):
        wandb_flag = False
    
    # Initialize model and optimizer
    model = CNNBS()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
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
    print(f"Time taken to train in {os.path.basename(__file__)}:", 
          end_time - start_time, "seconds")


if __name__ == "__main__":
    main()