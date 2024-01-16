import os.path

import numpy as np
import random
import torch
from stingray.bispectrum import Bispectrum
from stingray import lightcurve
import matplotlib.pyplot as plt
from scipy.signal import gausspulse
from scipy import signal

def calculate_bispectrum_power_spectrum_efficient(x, dt):
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
    f = np.fft.fftshift(np.fft.fftfreq(N, dt))

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

def create_gaussian_pulse(mean, std, n, amplitude=None):
    if amplitude is None:
        amplitude = 1 / (np.sqrt(2 * np.pi) * std)
    else:
        amplitude = amplitude
    # r = std + np.linspace(mean - std, mean + std, n)  # Create evenly spaced x values
    t = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
    dt = t[1] - t[0]
    fs = 1. / dt
    x = amplitude * np.exp(-(t - mean) ** 2 / (2 * std ** 2))  # Gaussian function

    return x, t, dt, fs
def create_two_gaussians(mean1, mean2, std1, std2, n):

    x1 = create_gaussian_pulse(mean1, std1, n)
    x2 = create_gaussian_pulse(mean2, std2, n)

    return x1 + x2

def calculate_bispectrum2(x):
    c3 = calculate_coeff3(x)
    return np.fft.fft2(c3)
def calculate_coeff3(x):
    N = len(x)
    c3 = np.zeros((N, N), dtype=complex)
    for n1 in np.linspace(-(N - 1) / 2, (N - 1) / 2, N, dtype=int):
        for n2 in np.linspace(-(N - 1) / 2, (N - 1) / 2, N, dtype=int):
            val = 0
            for n in np.linspace(-(N - 1) / 2, (N - 1) / 2, N, dtype=int):
                #val += x[n] * torch.conj(x[(n - n1) % N]) * x[(n + n2) % N]
                val += x[n] * torch.conj(x[int((n - n1 + N/2) % N - N/2)]) * x[int((n + n2 + N/2) % N - N/2)]
                val /= N
            c3[n1, n2] = val
    return c3

#   N = 100   # Signal length of N samples
#   signal_type = 'Gaussian' # {'UnitVec', 'Gaussian'}
#   mean = N / 2  # Center at mean
#   std = 1   # Standard deviation of std
def test(n, signal_type, n_shifts, mse_thresh, folder, fftshift=True, calc_c3 = False, norm=True, **params):
    perform_old_new_test = False
    suffix = ''
    # Create the signal
    params = params['params']
    if norm == True:
        suffix +='_norm'
    if signal_type == 'Gaussian1':
        mean = params['mean']# any real number
        std = params['std'] #>0
        suffix = f'{signal_type}_m{mean}_s{std}'

        x, t, dt, fs = create_gaussian_pulse(mean, std, n)
        if norm == True:
            x /= np.abs(x.max())
        print(f'Running test {signal_type}, with (mean, std) = ({mean}, {std})')

    elif signal_type == 'Gaussian2':
        mean1 = params['mean1']  # any real number
        std1 = params['std1']  # >0
        mean2 = params['mean2']  # any real number
        std2 = params['std2']  # >0
        suffix = f'{signal_type}_m1_{mean1}_s1_{std1}_m2_{mean2}_s2_{std2}'
        x1, t1, dt1, fs1 = create_gaussian_pulse(mean1, std1, n)
        x2, t2, dt2, fs2 = create_gaussian_pulse(mean2, std2, n)
        if norm == True:
            x1 /= np.abs(x1.max())
            x2 /= np.abs(x2.max())
        t = t1
        dt = dt1
        x = x1 + x2
        print(f'Running test {signal_type}, with (mean1, std1) = ({mean1}, {std1}), (mean2, std2) = ({mean2}, {std2})')
    elif signal_type == 'Gaussian3':
        mean1 = params['mean1']  # any real number
        std1 = params['std1']  # >0
        mean2 = params['mean2']  # any real number
        std2 = params['std2']  # >0
        amplitude1 = params['amplitude1']  # >0
        amplitude2 = params['amplitude2']  # >0
        if ((amplitude1 is None) or (amplitude2 is None)):
            suffix = f'{signal_type}_m1_{mean1}_s1_{std1}_m2_{mean2}_s2_{std2}'
        else:
            suffix = f'{signal_type}_m1_{mean1}_s1_{std1}_amp1_{amplitude1}_m2_{mean2}_s2_{std2}_amp2_{amplitude2}'
        x1, t1, dt1, fs1 = create_gaussian_pulse(mean1, std1, n)
        x2, t2, dt2, fs2 = create_gaussian_pulse(mean2, std2, n)
        if norm == True:
            x1 /= np.abs(x1.max())
            x2 /= np.abs(x2.max())
        x1 *= amplitude1
        x2 *= amplitude2
        noise = torch.randn(n)
        if noise is not None:
            suffix += '_noise'
        t = t1
        dt = dt1
        x1 = 100 * torch.tensor(x1)
        x2 = 0.5 * torch.tensor(x2)
        #x = torch.randn(N)#x1 + torch.multiply(x2, noise)
        x = x1 + noise
        x /= x.max()
        print(f'Running test {signal_type}, with (mean1, std1) = ({mean1}, {std1}), (mean2, std2) = ({mean2}, {std2})')
    elif signal_type == 'rand_sin':
        t = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
        dt = t[1] - t[0]
        # x1 = np.zeros(n)
        # rng = t[int(n / 4):int(n / 2)]
        # amp = 1
        # x1[rng] = amp * np.abs(signal.sawtooth(2 * np.pi * (rng) / len(rng))) # triangular
        # rng2 = t[int(n / 4):int(0.365* n)]
        # x1[rng2] = 0
        #x2 = np.ones(n)#np.abs(signal.sawtooth(t / 500))

        length = n  # Total length of the signal
        m = 40  # Center location of the triangle
        d = 20  # Width of the triangle
        x2 = np.zeros(length)  # Start with a zero-filled signal

        # Ascending side (left half of the triangle)
        x2[m - d // 2:m] = np.linspace(0, 1, d // 2)

        # Descending side (right half of the triangle)
        x2[m:m + d // 2] = np.linspace(1, 0, d // 2)[::-1]

        x1, t1, dt1, fs1 = create_gaussian_pulse(200., 2., n) # gaussian


        x = x1 + x2
        suffix = f'mix'
    else: # default
        shift = params['shift']
        suffix = f'{signal_type}_sh{shift}'
        x = torch.zeros(n)
        x[shift] = 1
        print(f'Running test {signal_type}, with pick in {shift} place)')
    suffix += f'_N{N}'
    folder_path = f'./figures/{folder}/{suffix}'
    x = torch.tensor(x)
    # Calculate the power
    # Calculate the Bispectrum
    Bx_efficient, Px, f = calculate_bispectrum_power_spectrum_efficient(x, dt)
    if perform_old_new_test == True:
        Bx = calculate_bispectrum(x)
        diff = torch.mean((torch.tensor(Bx) - Bx_efficient) ** 2)
        if np.abs(diff) > mse_thresh:
            print(f'Warning: absolute mse of regular calculation and efficient: {np.abs(diff)}')

    if calc_c3 == True:
        c3 = calculate_coeff3(x)
    # Figures
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Plot the 1D signal
    plt.figure()
    plt.xlabel("time [sec]")
    plt.title('1D signal - all')
    if signal_type != 'Gaussian1':
        plt.plot(t, x1, label='x1')
        plt.plot(t, x2, label='x2')
    plt.plot(t, x, label='x total')
    plt.legend()
    plt.savefig(f'{folder_path}/{suffix}__x_all.png')
    plt.close()

    plt.figure()
    plt.xlabel("time [sec]")
    plt.title('1D signal')
    plt.plot(t, x, label='x total')
    plt.legend()
    plt.savefig(f'{folder_path}/{suffix}__x.png')
    plt.close()
    # Plot the c3
    if calc_c3 == True:
        fig, ax = plt.subplots()
        shw = ax.imshow(np.real(c3))
        cbar = plt.colorbar(shw, ax=ax)
        cbar.set_label(f"C3")
        plt.savefig(f'{folder_path}/{suffix}__C3.png')
        plt.close()
    # Plot the Power spectrum
    plt.figure()
    plt.xlabel("Freq [Hz]")
    plt.title('The Power spectrum')
    plt.plot(f, Px, label='Power spectrum')
    plt.legend()
    plt.savefig(f'{folder_path}/{suffix}__Px.png')
    plt.close()

    # Plot real and imag of the Bispectrum
    real = Bx_efficient.real
    imag = Bx_efficient.imag
    fig, ax = plt.subplots()
    shw = ax.imshow(torch.log(real))
    cbar = plt.colorbar(shw, ax=ax)
    cbar.set_label(f"Bispectrum Real")
    plt.savefig(f'{folder_path}/{suffix}__Bx_real.png')
    plt.close()

    fig, ax = plt.subplots()
    shw = ax.imshow(torch.log(imag))
    cbar = plt.colorbar(shw, ax=ax)
    cbar.set_label(f"Bispectrum Imaginary")
    plt.savefig(f'{folder_path}/{suffix}__Bx_imag.png')
    plt.close()

    # Plot magnitude and phase of the Bispectrum
    magnitude = torch.abs(Bx_efficient)
    phase = torch.angle(Bx_efficient)

    fig, ax = plt.subplots()
    shw = ax.imshow(torch.log(magnitude))
    cbar = plt.colorbar(shw, ax=ax)
    cbar.set_label(f"Bispectrum magnitude")
    plt.savefig(f'{folder_path}/{suffix}__Bx_mag.png')
    plt.close()

    fig, ax = plt.subplots()
    shw = ax.imshow(torch.log(phase))
    cbar = plt.colorbar(shw, ax=ax)
    cbar.set_label(f"Bispectrum phase")
    plt.savefig(f'{folder_path}/{suffix}__Bx_phase.png')
    plt.close()

    # Verify the bispectrum is invariant under translations
    mse_avg = 0
    for i in range(n_shifts):
        shift = random.randint(0, n - 1)
        # Performing cyclic shift over the signal
        shifted_x = torch.roll(x, shift)
        # Calculating Bispectrum of the shifted signal
        shifted_Bx, _, _ = calculate_bispectrum_power_spectrum_efficient(shifted_x, dt)
        # Calculate the mse between Bx and shifted_Bx
        mse = torch.abs(torch.mean((Bx_efficient - shifted_Bx) ** 2)).item()
        mse_avg +=mse
        #print(f'mse={mse}')
        if mse > mse_thresh:
            print(f"Error! Bispectrums don't match. MSE error = {mse}")

    print(f"done! average mse is {mse_avg / n_shifts}")

def test2(mean1, mean2, std1, std2, n, folder_path, suffix):
    x1, t1, dt1, fs1 = create_gaussian_pulse(mean1, std1, n)
    x2, t2, dt2, fs2 = create_gaussian_pulse(mean2, std2, n)
    x = x1 + x2
    # Plot the 1D signal
    plt.figure()
    plt.xlabel("time [sec]")
    plt.title('1D signal')
    plt.plot(t1, x1, label='x1')
    plt.plot(t1, x2, label='x2')
    plt.plot(t1, x, label='x')
    plt.legend()
    plt.savefig(f'{folder_path}/{suffix}__x.png')
    plt.close()

if __name__ == "__main__":

    #test2(mean1=0.0, mean2=20., std1=2.1, std2=1.1, n=100, folder_path=f'./figures/test2', suffix='')



    N = 1000
    n_shifts = 10
    mse_thresh = 1e-16
    fftshift = True
    folder = "results_new"
    print(f'Performing {n_shifts} random shifts over the signal, with absolute mse threshold of {mse_thresh}')
    norm = True
    # test(n=N, signal_type='rand_sin', n_shifts=n_shifts, mse_thresh=mse_thresh, folder=folder,
    #      params={})
    test(n=N, signal_type='Gaussian3', n_shifts=n_shifts, mse_thresh=mse_thresh, folder=folder, calc_c3=False, norm=True,
         params={'mean1': -50.0, 'std1': 0.1, 'mean2': 300., 'std2': 0.1, 'amplitude1': 0.5, 'amplitude2': 1})
    # test(n=N, signal_type='Gaussian2', n_shifts=n_shifts, mse_thresh=mse_thresh, folder=folder, calc_c3=False,
    #      params={'mean1': 0.0, 'std1': 0.1, 'mean2': 400., 'std2': 1.1})
    # test(n=N, signal_type='Gaussian2', n_shifts=n_shifts, mse_thresh=mse_thresh, folder=folder, calc_c3=False,
    #      params={'mean1': 0.0, 'std1': 0.1, 'mean2': 400., 'std2': 5.1})
    # test(n=N, signal_type='Gaussian2', n_shifts=n_shifts, mse_thresh=mse_thresh, folder=folder, calc_c3=False,
    #      params={'mean1': 0.0, 'std1': 0.1, 'mean2': 400., 'std2': 50.1})
    # test(n=N, signal_type='Gaussian2', n_shifts=n_shifts, mse_thresh=mse_thresh, folder=folder, calc_c3=False,
    #      params={'mean1': -100.0, 'std1': 0.1, 'mean2': 400., 'std2': 0.1})
    # test(n=N, signal_type='Gaussian2', n_shifts=n_shifts, mse_thresh=mse_thresh, folder=folder, calc_c3=False,
    #      params={'mean1': -100.0, 'std1': 50, 'mean2': 400., 'std2': 100.})
    # test(n=N, signal_type='Gaussian2', n_shifts=n_shifts, mse_thresh=mse_thresh, folder=folder, calc_c3=False,
    #      params={'mean1': -100.0, 'std1': 500, 'mean2': 400., 'std2': 0.1})
    #test(n=N, signal_type='Gaussian2', n_shifts=n_shifts, mse_thresh=mse_thresh, folder=folder,
    #     params={'mean1': 0.0, 'std1': 50., 'mean2': 20., 'std2': 1.1})
    #test(n=N, signal_type='Gaussian2', n_shifts=n_shifts, mse_thresh=mse_thresh, folder=folder,
    #     params={'mean1': 0.0, 'std1': 115, 'mean2': 20., 'std2': 1.1})
    # test(n=N, signal_type='Gaussian1', n_shifts=n_shifts, mse_thresh=mse_thresh, folder=folder, calc_c3=True,
    #      params={'mean': 0.0, 'std': 0.1})


