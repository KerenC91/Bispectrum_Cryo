import numpy as np
import random
import torch
from stingray.bispectrum import Bispectrum
from stingray import lightcurve
import matplotlib.pyplot as plt
from scipy.signal import gausspulse

def calculate_bispectrum_power_spectrum_efficient(x, fs, nfft=1024):
    N = len(x)
    # DFT(x)
    y = torch.fft.fft(x)
    y_shifted = torch.fft.fftshift(y)
    # Power spectrum
    Px = y_shifted * torch.conj(y_shifted).T
    Px = Px.real # imagionary value is 0, just for changing types for python
    # Bispectrum
    circulant = lambda v: torch.cat([f := v, f[:-1]]).unfold(0, len(v), 1).flip(0)
    Bx = y.unsqueeze(1) * torch.conj(y).T.unsqueeze(0) * circulant(torch.roll(y, -1))
    Bx = torch.fft.fftshift(Bx)
    #f = fs * np.arange(-nfft / 2, nfft / 2 - 1, 1) / nfft
    f = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))

    return Bx, Px, f

def calculate_bispectrum(x):
    N = len(x)
    y = torch.fft.fft(x)
    Bx = torch.zeros((N, N), dtype=torch.complex64)
    for k1 in range(N):
        for k2 in range(N):
            Bx[k1, k2] = y[k1] * torch.conj(y[k2]) * y[(k2 - k1) % N]
    Bx = torch.fft.fftshift(Bx)
    return Bx

def calculate_bispectrum2(x):
    c3 = calculate_coeff3(x)
    return np.fft.fft2(c3)
def calculate_coeff3(x):
    N = len(x)
    c3 = np.zeros((N, N), dtype=complex)
    for n1 in range(N):
        for n2 in range(N):
            val = 0
            for n in range(N):
                val += x[n] * x[(n - n1) % N].conjugate() * x[(n + n2) % N]
                val /= N
            c3[n1, n2] = val
    return c3

#   N = 100   # Signal length of N samples
#   signal_type = 'Gaussian' # {'UnitVec', 'Gaussian'}
#   mean = N / 2  # Center at mean
#   std = 1   # Standard deviation of std
def test(n=100, signal_type='Gaussian', n_shifts=10, mse_thresh=1e-13, fftshift=True, perform_old_new_test = True, **params):
    # Create the signal
    params=params['params']
    fs = n
    if signal_type == 'Gaussian':
        mean = params['mean']# any real number
        std = params['std'] #>0
        amplitude = 1 / (np.sqrt(2 * np.pi) * std)
        suffix = f'{signal_type}_m{mean}_s{std}'
        #r = std + np.linspace(mean - std, mean + std, n)  # Create evenly spaced x values
        if fs != 0:
            # mean has to be in the range of (-0.5, 0.5)
            t = np.arange(-0.5, 0.5, 1 / fs)# fs = n
        else:
            t = np.arange(0, n, 1)
        x = amplitude * np.exp(-(t - mean) ** 2 / (2 * std ** 2))  # Gaussian function

        x = torch.tensor(x)
        #x = torch.tensor(np.random.normal(mean, std, n)) # x is in R_N here
    else: # default
        shift = params['shift']
        suffix = f'{signal_type}_sh{shift}'
        x = torch.zeros(n)
        x[shift] = 1
    if fftshift == True:
        suffix += '_fftshift'

    suffix += f'_fs{fs}'

    # Calculate the power
    # Calculate the Bispectrum
    Bx_efficient, Px, f = calculate_bispectrum_power_spectrum_efficient(x, fs)
    if perform_old_new_test == True:
        Bx = calculate_bispectrum(x)
        diff = torch.mean((torch.tensor(Bx) - Bx_efficient) ** 2)
        if np.abs(diff) > mse_thresh:
            print(f'Warning: absolute mse of regular calculation and efficient: {np.abs(diff)}')

    # Figures
    # Plot the 1D signal
    plt.figure()
    plt.xlabel("time [sec]")
    plt.title('1D signal')
    plt.plot(t, x, label='1D signal')
    plt.legend()
    plt.savefig(f'./figures/fs_scale/{suffix}__x.png')
    plt.close()
    # Plot the Power spectrum
    plt.figure()
    plt.xlabel("Freq [Hz]")
    plt.title('The Power spectrum')
    plt.plot(f, Px, label='Power spectrum')
    plt.legend()
    plt.savefig(f'./figures/fs_scale/{suffix}__Px.png')
    plt.close()
    # Plot magnitude and phase of the Bispectrum
    magnitude = torch.abs(Bx_efficient)
    phase = torch.angle(Bx_efficient)

    fig, ax = plt.subplots()
    shw = ax.imshow(magnitude)
    cbar = plt.colorbar(shw, ax=ax)
    cbar.set_label(f"Bispectrum magnitude")
    plt.savefig(f'./figures/fs_scale/{suffix}__Bx_mag.png')
    plt.close()

    fig, ax = plt.subplots()
    shw = ax.imshow(phase)
    cbar = plt.colorbar(shw, ax=ax)
    cbar.set_label(f"Bispectrum phase")
    plt.savefig(f'./figures/fs_scale/{suffix}__Bx_phase.png')
    plt.close()

    # Verify the bispectrum is invariant under translations
    mse_avg = 0
    for i in range(n_shifts):
        shift = random.randint(0, n - 1)
        # Performing cyclic shift over the signal
        shifted_x = torch.roll(x, shift)
        # Calculating Bispectrum of the shifted signal
        shifted_Bx, _, _ = calculate_bispectrum_power_spectrum_efficient(shifted_x, fs)
        # Calculate the mse between Bx and shifted_Bx
        mse = torch.abs(torch.mean((Bx_efficient - shifted_Bx) ** 2)).item()
        mse_avg +=mse
        #print(f'mse={mse}')
        if mse > mse_thresh:
            print(f"Error! Bispectrums don't match. MSE error = {mse}")

    print(f"done! average mse is {mse_avg / n_shifts}")

if __name__ == "__main__":
    N = 1000
    n_shifts = 10
    mse_thresh = 1e-16
    fftshift = True
    perform_old_new_test = False
    # mean should be around -0.5 to 0.5
    print(f'Performing {n_shifts} random shifts over the signal, with absolute mse threshold of {mse_thresh}')
    #test(n=N, signal_type='UnitVec', params={'shift': 5})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.0, 'std': 1e-10})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.0, 'std': 1e-5})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.0, 'std': 1e-3})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.0, 'std': 1})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.0, 'std': 5})

    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-8, 'std': 1e-10})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-8, 'std': 1e-5})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-8, 'std': 1e-3})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-8, 'std': 1})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-8, 'std': 5})

    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-5, 'std': 1e-10})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-5, 'std': 1e-5})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-5, 'std': 1e-3})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-5, 'std': 1})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-5, 'std': 5})

    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-3, 'std': 1e-10})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-3, 'std': 1e-5})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-3, 'std': 1e-3})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-3, 'std': 1})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 1e-3, 'std': 5})

    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.1, 'std': 1e-10})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.1, 'std': 1e-5})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.1, 'std': 1e-3})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.1, 'std': 1})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.1, 'std': 5})

    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.499, 'std': 1e-10})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.499, 'std': 1e-5})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.499, 'std': 1e-3})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.499, 'std': 1})
    test(n=N, signal_type='Gaussian', n_shifts=n_shifts, mse_thresh=mse_thresh, fftshift=True, params={'mean': 0.499, 'std': 5})