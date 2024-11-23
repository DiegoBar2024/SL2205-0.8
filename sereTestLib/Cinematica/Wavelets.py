import matplotlib.pyplot as plt
import scaleogram as scg
import seaborn as sns
import numpy as np
import sys
import pywt
import obspy
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt

# Cantidad de muestras
N = 600

# Período de muestreo
T = 1.0 / 1500.0

## Vector de tiempos
x = np.linspace(0.0, N * T, N, endpoint = False)

## Función
y = np.cos(50.0 * 2.0 * np.pi * x) + 2 * np.cos(100.0 * 2.0 * np.pi * x)

## Ploteo de Escalograma
coef, scales_freq = pywt.cwt(data = y, scales = [1, 10], wavelet = 'cmor1.5-1', sampling_period = T)

coef = np.abs(coef[:-1, :-1])

fig, axs = plt.subplots(1, 1)
pcm = axs.pcolormesh(x, scales_freq, coef)
axs.set_yscale("log")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Frequency (Hz)")
axs.set_title("Continuous Wavelet Transform (Scaleogram)")
fig.colorbar(pcm, ax=axs)
plt.show()