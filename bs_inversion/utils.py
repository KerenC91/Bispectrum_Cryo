import torch
import numpy as np
import torch
import torch.nn as nn
import math

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
    Bx = clculate_bispectrum_efficient(x)

    # Calculate frequency resolution
    f = np.fft.fftshift(np.fft.fftfreq(N, dt))

    return Bx, Px, f
   
def clculate_bispectrum_efficient(x):
    """
    

    Parameters
    ----------
    x : torch N size, float
        signal.
    Returns
    -------
    Bx : torch NXNX1 size, complex-float
        Bispectrum.

    """
    y = torch.fft.fft(x)
    circulant = lambda v: torch.cat([f := v, f[:-1]]).unfold(0, len(v), 1).flip(0)
    # Bx = (y.unsqueeze(1) *\
    #     torch.conj(y).T.unsqueeze(0)) * circulant(torch.roll(y, -1))
    C = circulant(torch.roll(y, -1))
    Bx = y.unsqueeze(1) @ y.conj().unsqueeze(0)
    Bx = Bx * C
    return Bx


class BispectrumCalculator(nn.Module):
    def __init__(self, target_len, device):
        super().__init__()
        self.calculator = clculate_bispectrum_efficient
        self.target_len = target_len
        self.device = device
        self.channels = 2
        self.height = target_len
        self.width = target_len
        
    def _create_data(self, target):
        # Create data
        target = target.clone()
        bs = self.calculator(target.squeeze(0))
        bs_real = bs.real.float()
        bs_imag = bs.imag.float()
        source = torch.stack([bs_real, bs_imag], dim=0)
               
        return source, target 
    # target: signal 1Xtarget_len
    # source: bs     2Xtarget_lenXtarget_len
    def forward(self, target):
        batch_size = target.shape[0]
        # Iterate over the batch dimension using indexing
        source = torch.zeros(batch_size, self.channels, self.height, self.width).to(self.device)

        for i in range(batch_size):
            source[i], target[i] = self._create_data(target[i])
        return source, target  # Stack processed vectors



class BatchAligneToReference(nn.Module):
    def __init__(self, device):
        super().__init__()
        self._align = align_to_reference
        self.device = device
        
    def forward(self, x, xref):
        batch_size = x.shape[0]
        # Iterate over the batch dimension using indexing
        x_aligned = torch.zeros_like(x).to(self.device)
        inds = torch.zeros(batch_size).to(self.device)
        
        for i in range(batch_size):
            aligned, inds[i] = \
                self._align(x[i].squeeze(0), xref[i].squeeze(0))
            x_aligned[i] = aligned.unsqueeze(0)
        return x_aligned, inds  # Stack processed vectors
    
   
def align_to_reference(x, xref):
    """
    Aligns a signal (x) to a reference signal (xref) using circular shift.
    
    Args:
        x: A numpy array of the signal to be aligned.
        xref: A numpy array of the reference signal.
    
    Returns:
        A numpy array of the aligned signal.
    """
    
    # Check if input arrays have the same size
    assert x.shape == xref.shape, "x and xref must have identical size"
    assert len(x.shape) == 1, "x shape is greater than 1 dim"
    org_shape = x.shape
    
    # Reshape to column vectors
    x = x.flatten()
    xref = xref.flatten()
    
    # Compute FFTs
    x_fft = torch.fft.fft(x)
    xref_fft = torch.fft.fft(xref)
    
    # Compute correlation using inverse FFT of complex conjugate product
    correlation_x_xref = torch.real(torch.fft.ifft(torch.conj(x_fft) * xref_fft))
    
    # Find index of maximum correlation
    ind = torch.argmax(correlation_x_xref).item()
    
    # Perform circular shift
    x_aligned = torch.roll(x, ind)
    
    return x_aligned.reshape(org_shape), ind
           
def rand_shift_signal(target, target_len, batch_size):
    target = target.squeeze(1)
    
    shifts = np.random.randint(low=0, 
                               high=target_len, 
                               size=batch_size)
    
    rows, column_indices = np.ogrid[:target.shape[0], :target.shape[1]]

    # Always use a negative shift, so that column_indices are valid.
    #shifts[shifts < 0]= target.shape[1]
    #shifts += target.shape[1]
    column_indices = (column_indices + shifts[:, np.newaxis]) % target.shape[1]
    
    target = target[rows, column_indices]

    target = target.unsqueeze(1)
    
    return target, shifts


