import numpy as np
import matplotlib.pyplot as plt

N = 11#1 + 2**8
k = 2 * np.linspace(0, N - 1, N, endpoint=True) / N - 1

V = np.exp(-abs(k))

# Vf1 = np.fft(np.fftshift(V))
# Vf2 = np.fft(np.ifftshift(V))
# Vf3 = np.ifft(np.fftshift(V))
# Vf4 = np.ifft(np.ifftshift(V))
Vf5 = np.fft.fft(V)
# Vf6 = np.ifft(V)

# Plot the 1D signal
plt.figure()
plt.xlabel("time [sec]")
plt.title('V')
plt.plot(k, V, label='V')
plt.legend()
plt.close()

dk = k[1] - k[0]
f = np.fft.fftfreq(N, dk)

plt.figure()
plt.xlabel("f [Hz]")
plt.title('Vf5')
plt.plot(f, Vf5, label='Vf5')
plt.legend()
plt.close()