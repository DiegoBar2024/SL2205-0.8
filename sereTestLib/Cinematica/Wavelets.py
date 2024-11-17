import matplotlib.pyplot as plt
import scaleogram as scg
import seaborn as sns
import numpy as np
import sys
import pywt

# Cantidad de muestras
N = 600

# Período de muestreo
T = 1.0 / 1500.0

## Vector de tiempos
x = np.linspace(0.0, N * T, N, endpoint = False)

## Función
y = np.cos(50.0 * 2.0 * np.pi * x)

coef, scales_freq = pywt.cwt(data = y, scales = [1, 10], wavelet = 'cmor1.5-1', sampling_period = T)

coef = np.abs(coef[:-1, :-1])

fig, axs = plt.subplots(2, 1)
pcm = axs[0].pcolormesh(x, scales_freq, coef)
axs[0].set_yscale("log")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency (Hz)")
axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
fig.colorbar(pcm, ax=axs[0])
plt.show()

# def plot_wavelet(time, data, wavelet, title, ax):
#     widths = np.geomspace(1, 1024, num=75)
#     cwtmatr, freqs = pywt.cwt(
#         data, widths, wavelet, sampling_period=np.diff(time).mean()
#     )
#     cwtmatr = np.abs(cwtmatr[:-1, :-1])
#     pcm = ax.pcolormesh(time, freqs, cwtmatr)
#     ax.set_yscale("log")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Frequency (Hz)")
#     ax.set_title(title)
#     plt.colorbar(pcm, ax=ax)
#     return ax