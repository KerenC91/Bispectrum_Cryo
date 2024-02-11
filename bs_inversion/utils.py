import torch
import numpy as np
import torch
import torch.nn as nn

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
    def __init__(self, batch_size, target_len, device):
        super().__init__()
        self.calculator = calculate_bispectrum_power_spectrum_efficient
        self.batch_size = batch_size
        self.target_len = target_len
        self.device = device
        
    def _create_data(self, target):
        # Create data
        target = target.clone()
        bs, ps, f = self.calculator(target)
        bs_real = bs.real.float()
        bs_imag = bs.imag.float()
        source = torch.stack([bs_real, bs_imag], dim=1)
        
        return source, target.unsqueeze(0) 
    # target: signal 1Xtarget_len
    # source: bs     2Xtarget_lenXtarget_len
    def forward(self, target):
        # Iterate over the batch dimension using indexing
        source = torch.zeros(self.batch_size, 2, self.target_len, self.target_len).to(self.device)

        for i in range(self.batch_size):
            source[i], target[i] = self._create_data(target[i])
        return source, target  # Stack processed vectors