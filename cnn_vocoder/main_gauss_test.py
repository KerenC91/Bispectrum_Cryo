import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time 
import os
import wandb
from datetime import datetime
import torch 
from torch.utils.data import Dataset, DataLoader
import argparse
from bispectrum_calculation.bispectrum_calc_org import create_gaussian_pulse, calculate_bispectrum_power_spectrum_efficient
import cnn_vocoder as vc


def create_dataset(N):
    targets = F.one_hot(torch.arange(0, N)).type(torch.float64)
    sources = torch.zeros((N, 2, N, N))
    # target - ground truth image, source - Bispectrum of ground truth image
    for i in range(N):
        bs, px, f = torch.tensor(
            calculate_bispectrum_power_spectrum_efficient(targets[i], dt), 
                          dtype=torch.complex64)
        sources[i] = bs
        
    return targets, sources


class UnitVecDataset(Dataset):
    
    def __init__(self, targets, sources):
        self.targets = targets
        self.sources = sources
        self.data_size = self.__len__()
            
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):

        return self.targets[idx], self.sources[idx], idx
      


    
class Trainer:
    def __init__(self, model, 
                 train_loader, 
                 test_loader, 
                 batch_size, 
                 num_epochs, 
                 optimizer,
                 args):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.wandb_log_interval = args.wandb_log_interval
        self.norm_factor = 1
        
    # target - ground truth image, source - Bispectrum of ground truth image
    # might be multiple targets and sources (batch size > 1)
    def _run_batch(self, source, target):
        output = self.model(source) # reconstructed image
        loss = F.cross_entropy(output, target)
        return loss
    
    def _run_epoch_train(self):
        loss_sum = 0
        self.optimizer.zero_grad()
        
        for targets, sources, idx in self.train_loader:
            loss_sum += self._run_batch(sources, targets)
        loss_sum.backward()
        self.optimizer.step()

        return loss_sum

    def _run_epoch_test(self):
        loss_sum = 0
        
        for targets, sources, idx in self.test_loader:
            loss_sum += self._run_batch(sources, targets)

        return loss_sum
        
    def train(self):
        for epoch in range(0, self.num_epochs):
            loss_sum = self._run_epoch_train()
            
            if epoch % self.wandb_log_interval == 0:
                wandb.log({"train_loss": loss_sum / self.norm_factor})

    def test(self):
         for epoch in range(0, self.num_epochs):
             running_vloss = 0.0
             # Set the model to evaluation mode
             self.model.eval()
             
             # Disable gradient computation and reduce memory consumption.
             with torch.no_grad():
                test_loss = self._run_epoch_test()
                
                if epoch % self.wandb_log_interval == 0:
                    wandb.log({"test_loss": test_loss / self.norm_factor})
     
        
      

def main():
    
    parser = argparse.ArgumentParser(description='Inverting the bispectrum. Pulse dataset')

    parser.add_argument('--N', type=int, default=10, metavar='N',
            help='size of dataset')
    parser.add_argument('--train_test_split', type=float, default=0.75, metavar='N',
            help='split fraction to train and test')
    parser.add_argument('--batch_size', type=float, default=1, metavar='N',
            help='batch size')
    parser.add_argument('--wandb_log_interval', type=float, default=10, metavar='N',
            help='interval to log data to wandb')
    
    args = parser.parse_args()
    
    targets, sources = create_dataset(args.N)
    test_index = int(args.train_test_split * len(targets))
    
    train_dataset = UnitVecDataset(targets=targets[:test_index], 
                             sources=sources[:test_index])
    test_dataset = UnitVecDataset(targets=targets[test_index:], 
                             sources=sources[:test_index:])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = vc.CNNVocoder(n_heads=1)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, 
                      train_loader=train_loader, 
                      test_loader=test_loader, 
                      batch_size=1,                     
                      num_epochs=1000,
                      optimizer=optimizer,
                      args=args)
    
    start_time = time.time()
    wandb.login()
    wandb.init(project='GaussianBispectrumInversion',
                       name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                       config=args)
    wandb.watch(model, log_freq=0.1*args.N)
    trainer.train()
    end_time = time.time()
    print(f"Time taken to train in {os.path.basename(__file__)}:", 
          end_time - start_time, "seconds")

if __name__ == "__main__":
    main()