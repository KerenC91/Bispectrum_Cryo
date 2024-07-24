from optuna import Trial, Study, create_study
from torch import nn, optim
from torchvision import datasets, transforms
import torch.optim as optim
import time 
import os
import wandb
from datetime import datetime
import torch 
from torch.utils.data import Dataset, DataLoader
import argparse
from utils import BispectrumCalculator, BatchAligneToReference
from model1 import CNNBS, HeadBS1
from model2 import HeadBS2
from model3 import HeadBS3
from optuna_params import optuna_params
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
import sys
import csv
import random
from train_main import set_activation, update_reduce_height_cnt, create_dataset, prepare_data_loader

# Set the same seed for reproducibility
torch.manual_seed(1234)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set helpers
bs_calc = BispectrumCalculator(optuna_params.K, optuna_params.N, device).to(device)
aligner = BatchAligneToReference(device).to(device)
                

class UnitVecDataset(Dataset):
    
    def __init__(self, source, target):
        self.target = target
        self.source = source
        self.data_size = self.__len__()
            
        
    def __len__(self):
        return self.target.size(0)
    
    def __getitem__(self, idx):

        return idx, (self.source[idx], self.target[idx])

def loss_sc(bs_pred, bs_gt, method="average"):

     if method == "sum":
         sh = bs_pred.shape
         loss = torch.mean(
                     torch.norm((bs_pred - bs_gt).view(sh[0], sh[1], -1), dim=(0, 2)) / \
                         torch.norm(bs_gt.view(sh[0], sh[1], -1), dim=(0, 2)))
     else:
         loss = torch.norm(bs_pred - bs_gt) / torch.norm(bs_gt)

     return loss
    
def loss_l1(pred, target):
    criterion = torch.nn.L1Loss()          

    return criterion(pred, target)

    
def loss_MSE(pred, target):
    criterion = torch.nn.MSELoss()  
    
    return criterion(pred, target)
   
def loss_rel_MSE(pred, target):
    return torch.mean(
                torch.norm(pred - target, dim=(0, 2))**2 / \
                torch.norm(target, dim=(0, 2))**2)
    
def loss_all(pred, target):
    bs_pred, pred = bs_calc(pred, optuna_params.loss_method)
    bs_target, target = bs_calc(target, optuna_params.loss_method)            
    total_loss = 0.
    
    loss = loss_sc(bs_pred, bs_target, optuna_params.loss_method)

    loss = loss, \
            loss_MSE(pred, target), \
            loss_rel_MSE(pred, target)

    return loss
    

def try_model(trial, device):
    model_num = 3#trial.suggest_int("model_num", 1, 3)
    
    pre_residuals = trial.suggest_int("pre_residuals", 0, 14)
    up_residuals = trial.suggest_int("up_residuals", 0, 14)   
    post_residuals = trial.suggest_int("post_residuals", 0, 14)
    n_heads = trial.suggest_int("n_heads", 1, 5)
    activation_name = trial.suggest_categorical('activation', 
                                           ['ELU',
                                            'LeakyReLU',
                                            'ReLU',
                                            'Softsign',
                                            'Tanh'])
   
      
    
    if model_num == 2:
        head_class = HeadBS2
        channels = optuna_params.channels_model2
    elif model_num == 3:
        head_class = HeadBS3 
        channels = optuna_params.channels_model3
    else:
        head_class = HeadBS1
        channels = optuna_params.channels_model1
    
    activation = set_activation(activation_name) 
        
    # last_ch_power = trial.suggest_int("last_ch_power", 5, 10)
    # last_ch = int(2**last_ch_power)
    # optuna_params.last_ch = last_ch
    optuna_params.pre_conv_channels[-1] = optuna_params.last_ch
    channels[-1] = optuna_params.last_ch
    cnt, k, s = optuna_params.reduce_height
    reduce_height = update_reduce_height_cnt(k, s, optuna_params.N)
    
    model = CNNBS(
        device=device,
        input_len=optuna_params.N,
        signals_count = optuna_params.K,
        n_heads=n_heads,
        channels=channels,
        b_maxout = optuna_params.maxout,
        pre_conv_channels=optuna_params.pre_conv_channels,
        pre_residuals=pre_residuals,
        up_residuals=up_residuals,
        post_residuals=post_residuals,
        pow_2_channels=optuna_params.pow_2_channels,
        reduce_height=reduce_height,
        head_class = head_class,
        linear_ch=optuna_params.last_ch,
        activation=activation
        )
    return model
     

def try_scheduler(trial, scheduler_name, optimizer, epochs):
    # ReduceLROnPlateau
    factor = trial.suggest_categorical("factor", 
                                       [1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4, 0.5])
    patience = trial.suggest_int("patience", 3, 10)
    threshold = trial.suggest_categorical("threshold", 
                                          [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    cooldown = trial.suggest_int("cooldown", 0, 10)
    # StepLR
    step_size = trial.suggest_categorical("step_size", 
                                          [100, 500, 1000, 2000, 3000]) 
    # OneCycleLR + StepLR
    gamma = trial.suggest_categorical("gamma", [1e-3, 1e-2, 1e-1])
    # OneCycleLR
    max_lr = trial.suggest_categorical("max_lr", [1e-3, 1e-2, 0.1])
    pct_start = trial.suggest_categorical("pct_start", [0.3, 0.4, 0.5, 0.6])
    anneal_strategy=trial.suggest_categorical("anneal_strategy", ["cos", "linear"])
    # CosineAnnealingLR
    #T_max=epochs * 1 * cos_ann_lr_T_max_f
    cos_ann_lr_T_max_f = trial.suggest_categorical("cos_ann_lr_T_max_f", 
                                                   [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    # CyclicLR
    cyclic_lr_mode = trial.suggest_categorical( "cyclic_lr_mode",
                                               ["triangular", "triangular2", "exp_range"])
    cyclic_lr_base_lr = trial.suggest_categorical("cyclic_lr_base_lr", 
                                                   [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    cyclic_lr_diff_lr_f = trial.suggest_categorical("cyclic_lr_diff_lr_f", 
                                                   [2, 10, 100, 1000])
    cyclic_lr_max_lr = np.min([1, cyclic_lr_base_lr * cyclic_lr_diff_lr_f])
    #step_size_up=int(epochs * 1 / 2 / cyclic_lr_step_size_up_f)
    cyclic_lr_step_size_up_f = trial.suggest_categorical("cyclic_lr_step_size_up_f", 
                                                   [2, 3, 5, 10, 100, 1000])
    cyclic_lr_gamma = trial.suggest_categorical("cyclic_lr_gamma", 
                                                   [0.1, 1, 2, 3])
            
    scheduler = None
    if scheduler_name != 'None':
        if scheduler_name == 'ReduceLROnPlateau':
             scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                 optimizer=optimizer,
                 mode='min',
                 factor=factor,
                 threshold=threshold,
                 patience=patience, 
                 cooldown=cooldown)
        elif scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=step_size,
                gamma=gamma)
        elif scheduler_name == 'OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr = max_lr,
                steps_per_epoch =1,
                epochs= epochs,
                pct_start = pct_start,
                anneal_strategy=anneal_strategy)   
        elif scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=epochs * 1 * cos_ann_lr_T_max_f) 
        elif scheduler_name == 'CyclicLR':        
            scheduler = optim.lr_scheduler.CyclicLR(
                optimizer=optimizer,
                mode=cyclic_lr_mode,
                base_lr=cyclic_lr_base_lr, 
                max_lr=cyclic_lr_max_lr,
                step_size_up=int(epochs * 1 / 2 / cyclic_lr_step_size_up_f),
                gamma=cyclic_lr_gamma)             
        return scheduler

      
def try_optimizer(trial, model):
    
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", 
                                                             "SGD", 
                                                             "RMSprop", 
                                                             "AdamW"])
    # all optimizer params
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1.)
    eps = trial.suggest_float("eps", 1e-10, 1e-6)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.6)
    # RMSprop
    alpha = trial.suggest_float("alpha", 0.45, 0.99)
    #SGD
    momentum = trial.suggest_float("momentum", 0, 0.9)      
    #Adam
    beta1 = trial.suggest_float("beta1", 0.8, 0.8999)
    beta2 = trial.suggest_float("beta2", 0.9, 0.9999)
    
    # Create optimizer and scheduler based on trial suggestions
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=eps,
                               betas=(beta1, beta2), weight_decay=weight_decay)   
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, 
                                        alpha=alpha,
                                        eps=eps)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                              weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                      betas=(beta1, beta2),
                                      eps=eps,
                                      weight_decay=weight_decay)
        
    return optimizer

            
def objective(trial: Trial, epochs):

    # Initialize model and optimizer
    model = try_model(trial, device)
    model.to(device)
    optimizer = try_optimizer(trial, model)
    # set scheduler
    scheduler_name = trial.suggest_categorical("scheduler", 
                                              [#"None",
                                               #"Manual",
                                               "ReduceLROnPlateau", 
                                               "OneCycleLR",
                                               "StepLR",
                                               "CosineAnnealingLR",
                                               "CyclicLR"])
    scheduler = try_scheduler(trial, scheduler_name, optimizer, epochs)
      
    # set train dataset and dataloader
    train_dataset = create_dataset(device, optuna_params.train_data_size, optuna_params.K,
                                   optuna_params.N, optuna_params.read_baseline, 
                                   optuna_params.mode, optuna_params.folder_matlab)
    train_loader = prepare_data_loader(train_dataset, 
                                       batch_size=optuna_params.batch_size)

    # Start training    
    opt_loss = 0.
    
    # Train         
    model.train()

    for epoch in range(epochs):

        total_loss = 0
        total_mse_loss = 0
        total_mse_norm_loss = 0
        
        for idx, (sources, targets) in train_loader:
            # zero grads
            optimizer.zero_grad()
            # forward pass + loss computation
            # Move data to device
            targets = targets.to(device)
            sources = sources.to(device)
            # Forward pass
            output = model(sources) # reconstructed signal
            output, _ = aligner(output, targets)
            # Loss calculation
            loss, mse_loss, rel_mse_loss = loss_all(output, targets)
            # backward pass
            loss.backward()
            # optimizer step
            optimizer.step()
            # update avg loss 
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_mse_norm_loss += rel_mse_loss.item()
            # scheduler step after batch
            if scheduler_name != 'None':
                if scheduler_name in ['OneCycleLR', 'CosineAnnealingLR', 'CyclicLR']:
                    scheduler.step()
        avg_loss = total_loss / len(train_loader)
        avg_mse_loss = total_mse_loss / len(train_loader) 
        avg_mse_norm_loss = total_mse_norm_loss / len(train_loader) 
       
        # scheduler step after epoch
        if scheduler_name != 'None':
            if scheduler_name == 'Manual':
                if epoch in hparams.manual_epochs_lr_change:
                    optimizer.param_groups[0]['lr'] *= hparams.manual_lr_f 
            elif scheduler_name == 'StepLR':
                scheduler.step()
            elif scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(avg_loss)
        # seek for best loss
        opt_loss = avg_loss # optimize l1_loss
        # if train_loss < best_loss:
        #     best_loss = train_loss
        # else:
        #     break

    return opt_loss
    

# class Objective:
#     def __init__(self, min_x, max_x):
#         # Hold this implementation specific arguments as the fields of the class.
#         self.min_x = min_x
#         self.max_x = max_x

#     def __call__(self, trial):
#         # Calculate an objective value by using the extra arguments.
#         x = trial.suggest_float("x", self.min_x, self.max_x)
#         return (x - 2) ** 2

if __name__ == "__main__":
    # Add arguments to parser
    parser = argparse.ArgumentParser(
        description='Optuna Test - Inverting the bispectrum. Pulse dataset')

    parser.add_argument('--n', type=int, default=10, metavar='N',
            help='The umber of trials to run the test')
    parser.add_argument('--epochs', type=int, default=7000, metavar='N',
            help='The umber of epochs per trial')
    args = parser.parse_args()
    study = create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args.epochs), n_trials=args.n)