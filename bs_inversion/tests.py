import torch
import torch.nn as nn
import math
from utils import calculate_bispectrum_power_spectrum_efficient, align_to_reference
import torch
import torch.nn as nn
import os
import numpy as np
import torch
import random


def positional_encoding(x):
    max_length = x.size(2)#10 # length of sequence, here length of signal
    d_model = x.size(-1)
    n = 10000.0
    pe = torch.zeros(max_length, d_model)    

    # create position column   
    k = torch.arange(0, max_length).unsqueeze(1)  

    # calc divisor for positional encoding 
    div_term = torch.exp(                                 
            torch.arange(0, d_model, 2) * -(math.log(n) / d_model)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)    

    # calc cosine on odd indices   
    pe[:, 1::2] = torch.cos(k * div_term)  

    # add dimension     
    pe = pe.unsqueeze(0).unsqueeze(0)
    stacked_tensor = torch.cat([x, pe], dim=1)
    
    return stacked_tensor    

def positional_encoding2(x):
    max_length = x.size(2)#10 # length of sequence, here length of signal
    d_model = x.size(-1)
    n = 10000.0
    pe = torch.zeros(max_length, d_model)    

    # create position column   
    k = torch.arange(0, max_length).unsqueeze(1)  

    # calc divisor for positional encoding 
    div_term = torch.exp(                                 
            torch.arange(0, d_model, 2) * -(math.log(n) / d_model)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)    

    # calc cosine on odd indices   
    pe[:, 1::2] = torch.cos(k * div_term)  

    # add dimension     
    pe = pe.unsqueeze(0).unsqueeze(0)
    pe = pe.repeat(1, 2, 1, 1)
    stacked_tensor = torch.cat([x, pe], dim=2)
    
    return stacked_tensor 
    
def duplicate_and_expand(x, d_model):
    # x is of shape 1D n [ 1, 90,  6] n = 3, d_model = 3
    expanded_x = x.repeat(1, d_model, 1) #torch.ones((1, d_model, 1)) * x
    # shape 3D 1Xd_modelXn
    # tensor([[[ 1, 90,  6],
    #          [ 1, 90,  6],
    #          [ 1, 90,  6]]])
    expanded_x = expanded_x.T.squeeze(1).flatten()
    # shape 1D (d_modelXn)
    #tensor([ 1.,  1.,  1., 90., 90., 90.,  6.,  6.,  6.])
    return expanded_x

    
class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
    """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
    # inherit from Module
    super().__init__()     

    # initialize dropout                  
    self.dropout = nn.Dropout(p=dropout)      

    # create tensor of 0s
    pe = torch.zeros(max_length, d_model)    

    # create position column   
    k = torch.arange(0, max_length).unsqueeze(1)  

    # calc divisor for positional encoding 
    div_term = torch.exp(                                 
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)    

    # calc cosine on odd indices   
    pe[:, 1::2] = torch.cos(k * div_term)  

    # add dimension     
    pe = pe.unsqueeze(0)          

    # buffers are saved in state_dict but not trained by the optimizer                        
    self.register_buffer("pe", pe)                        

  def forward(self, x):
    """
    Args:
      x:        embeddings (batch_size, seq_length, d_model)
    
    Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
    """
    # add positional encoding to the embeddings
    x = x + self.pe[:, : x.size(1)].requires_grad_(False) 

    # perform dropout
    return self.dropout(x)



def test_positional_encoding(type=1):
    # create signal
    x = torch.randn(1, 2, 10, 10)
    if type == 1:
        pe = positional_encoding(x)
    else:
        pe = positional_encoding2(x)


def test_strided_conv_height():
    batch_size = 1
    in_channels = 2
    in_height = 10
    in_width = 10
    out_channels = 2
    out_height = 100
    out_width = 100
    kernel_size = 3
    stride = 2

    return strided_conv_height(batch_size, 
                            in_channels, 
                            in_height, 
                            in_width, 
                            out_channels,
                            out_height,
                            out_width,
                            kernel_size,
                            stride)


def strided_conv_height(batch_size, 
                        in_channels, 
                        in_height, 
                        in_width, 
                        out_channels,
                        out_height,
                        out_width,
                        kernel_size,
                        stride):
    x = torch.randn(batch_size, in_channels, in_height, in_width)
    print(f'x.shape = {x.shape}')

    # Height conv
    m = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, 
        kernel_size=(kernel_size, 1), stride=(stride, 1))
    
    y = m(x)
    print(f'y.shape = {y.shape}')
    return y


class VectorProcessor(nn.Module):
    def __init__(self, f, batch_size, target_len):
        super().__init__()
        self.f = f
        self.batch_size = batch_size
        self.target_len = target_len
        
    def _create_rand_data(self, target):
        # Create data
        bs, ps, f = self.f(target)
        bs_real = bs.real.float()
        bs_imag = bs.imag.float()
        source = torch.stack([bs_real, bs_imag], dim=1)
        
        return source, target.unsqueeze(0) 
       
    def forward(self, target):
        # Iterate over the batch dimension using indexing
        source = torch.zeros(self.batch_size, 2, self.target_len, self.target_len)
        target = torch.randn(self.batch_size, 1, self.target_len)

        for i in range(self.batch_size):
            source[i], target[i] = self._create_rand_data(target[i])
        return source, target  # Stack processed vectors




def test_VectorProcessor():
    batch_size = 2
    target_len = 10
    target = torch.randn(batch_size, 1, target_len)
    # Create the custom layer and apply it
    processor = VectorProcessor(f=calculate_bispectrum_power_spectrum_efficient,
                                batch_size=batch_size, 
                                target_len=target_len)
    processed_tensor = processor(target) 
    print('test_VectorProcessor done')
    

def test_vectorize():
    d_model = 4
    x = torch.tensor([1, 90, 6])
    print(x.shape)
    y = duplicate_and_expand(x, d_model)
    print(y.shape)
    
    x = torch.tensor([[1, 90, 6], [3, 5, 1]])
    #duplicate_and_expand(x, d_model)
    transformed_tensor = \
        x.apply_(lambda v: duplicate_and_expand(v.squeeze(1), d_model))
     
def read_tensor_from_matlab(file):
    x = np.loadtxt(file, delimiter=" ")
    x = torch.tensor(x).unsqueeze(1).unsqueeze(0)
    return x

def read_test_from_matlab(test_i=1,
                          folder='/scratch/home/kerencohen2/Git/HeterogeneousMRA/baseline_test'):
    sub_folder = f'test_{test_i}'
    home_folder = os.path.join(folder, sub_folder)
    
    x_true = read_tensor_from_matlab(os.path.join(home_folder, 'x_true.csv'))
    data = read_tensor_from_matlab(os.path.join(home_folder, 'data.csv'))
    shifts = float(np.loadtxt(os.path.join(home_folder, 'shifts.csv'), delimiter=" "))
    x_est = read_tensor_from_matlab(os.path.join(home_folder, 'x_est.csv'))
    p_est = float(np.loadtxt(os.path.join(home_folder, 'p_est.csv'), delimiter=" "))
    rel_error_X = float(np.loadtxt(os.path.join(home_folder, 'rel_error_X.csv'), delimiter=" "))
    tv_error_p = float(np.loadtxt(os.path.join(home_folder, 'tv_error_p.csv'), delimiter=" "))

    return x_true, data, shifts, x_est, p_est, rel_error_X, tv_error_p

def test_bs_correlation():
    n = 20
    n_shifts = 10
    dt = 1
    x = torch.randn(n)
    Bx_efficient, _, _ = \
        calculate_bispectrum_power_spectrum_efficient(x, dt)
    
    # Verify the bispectrum is invariant under translations
    mse_avg = 0
    mse_thresh = 1e-8
    
    for i in range(n_shifts):
        shift = random.randint(0, n - 1)
        # Performing cyclic shift over the signal
        shifted_x = torch.roll(x, shift)
        # Calculating Bispectrum of the shifted signal
        shifted_Bx, _, _ = \
            calculate_bispectrum_power_spectrum_efficient(shifted_x, dt)
        # Calculate the mse between Bx and shifted_Bx
        mse = torch.abs(torch.mean((Bx_efficient - shifted_Bx) ** 2)).item()
        mse_avg +=mse
        #print(f'mse={mse}')
        if mse > mse_thresh:
            print(f"Error! Bispectrums don't match. MSE error = {mse}")

    print(f"done! average mse is {mse_avg / n_shifts}")
    bs_x = calculate_bispectrum_power_spectrum_efficient(x)
    
def test_signals_correlation():
    n = 20
    shift = 5
    xref = torch.randn(n)
    x = torch.roll(xref, shift)
    x_aligned_reshaped, ind = align_to_reference(x, xref)
    print(f"done! ind={ind}, shift={shift}")
    
    
if __name__ == "__main__":

    #test_bs_correlation()
    test_signals_correlation()
    #read_test_from_matlab()
    #test_VectorProcessor()
    #test_strided_conv_height()
    #test_type = 2
    #test_positional_encoding(test_type)
    print('done')