import torch
import numpy as np


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