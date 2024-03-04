import torch
import numpy as np
import torch
import torch.nn as nn
import math

# add positional encoding as concatanation of data: [1, 2, 10, 10] --> [1, 2, 20, 10]
def perform_positional_encoding2(x, d_model):
    max_length = x.size(-1)#10 # length of sequence, here length of signal
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


# add positional ecoding as additional channel: [1, 2, 10, 10] --> [1, 3, 10, 10]
def perform_positional_encoding(x, d_model):
    max_length = x.size(2)#10 # length of sequence, here length of signal
    #d_model = x.size(-1)
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

def read_csv_from_matlab(file):
    x = np.loadtxt(file, delimiter=" ")
    x = torch.tensor(x).unsqueeze(0).unsqueeze(0)
    return x

def calculate_bispectrum_power_spectrum_efficient(x, dt=1.):
    """
    

    Parameters
    ----------
    x : torch N size, float
        signal.
    dt : float
        time resolution.

    Returns
    -------
    Bx : torch NXNX1 size, complex-float
        Bispectrum.
    Px : torch NX1 size, float (could be complex)
        Power spectrum.
    f : float
        frequency resolution.

    """
    # Get signal's length
    N = len(x)
    # Calculate DFT(x)
    y = torch.fft.fft(x)
    # Shift the DFT to the center
    y_shifted = torch.fft.fftshift(y)
    # Calculate the Power spectrum of x
    Px = y_shifted * torch.conj(y_shifted).T
    if torch.all(torch.isreal(Px)).item() == True:
        Px = Px.real # change to float type
    else:
        #print('Px is complex')
        pass
    # Calculate the Bispectrum
    Bx = clculate_bispectrum_efficient(y, shifted=False)

    # Calculate frequency resolution
    f = np.fft.fftshift(np.fft.fftfreq(N, dt))

    return Bx, Px, f
   
def clculate_bispectrum_efficient(y, shifted=True):
    """
    

    Parameters
    ----------
    y : torch, torch NX1 size, float (could be complex) 
        The DFT of the signal.
    shifted : Bool, optional
        True if the DFT is already shifted, False if not, and then 
        performing the shift manually. The default is True.
        note: for shifted = True the results are significantly different, 
        hence using shifted=False
    Returns
    -------
    Bx : torch NXNX1 size, complex-float
        Bispectrum.

    """
    circulant = lambda v: torch.cat([f := v, f[:-1]]).unfold(0, len(v), 1).flip(0)
    Bx = y.unsqueeze(1) *\
        torch.conj(y).T.unsqueeze(0) * circulant(torch.roll(y, -1))
    if shifted == False:
        # Shift the Bispectrum to center
        Bx = torch.fft.fftshift(Bx)
    return Bx


class BispectrumCalculator(nn.Module):
    def __init__(self, target_len, device):
        super().__init__()
        self.calculator = calculate_bispectrum_power_spectrum_efficient
        self.target_len = target_len
        self.device = device
        self.channels = 2
        self.height = target_len
        self.width = target_len
        
    def _create_data(self, target):
        # Create data
        target = target.clone()
        # print(f'GPU{self.device}: target.shape={target.shape}')
        bs, ps, f = self.calculator(target)
        # print(f'GPU{self.device}: bs.shape={bs.shape}')
        bs_real = bs.real.float()
        bs_imag = bs.imag.float()
        # print(f'GPU{self.device}: got here 2')
        source = torch.stack([bs_real, bs_imag], dim=1)
               
        return source, target.unsqueeze(0) 
    # target: signal 1Xtarget_len
    # source: bs     2Xtarget_lenXtarget_len
    def forward(self, target):
        batch_size = target.shape[0]
        # Iterate over the batch dimension using indexing
        source = torch.zeros(batch_size, self.channels, self.height, self.width).to(self.device)

        for i in range(batch_size):
            # print(f'GPU{self.device}: i={i}, source.shape={source.shape}')
            source[i], target[i] = self._create_data(target[i])
        return source, target  # Stack processed vectors