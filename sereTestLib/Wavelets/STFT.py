import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt

# Cantidad de muestras
N = 600

# Período de muestreo
T = 1.0 / 200

## Vector de tiempos
x = np.linspace(0.0, N * T, N, endpoint = False)

## Función
y = 2 * np.cos(20 * 2.0 * np.pi * x) + 2 * np.cos(2.25 * 2.0 * np.pi * x) + 20 * np.cos(75 * 2.0 * np.pi * x)

## Hago el cálculo
f, t, Zxx = stft(y, fs = 1 / T, nperseg = 256)

plt.pcolormesh(t, f, np.abs(Zxx), shading = 'gouraud')
plt.ylabel('Hz')
plt.xlabel('Sec')
plt.show()

## Calculo la energía de la señal
E_x = sum(x ** 2) * T

# Calculo una STFT con escalamiento de PSD
f, t, Zxx = stft(x, 1 / T, nperseg = 1000, return_onesided = False, scaling = 'psd')

# Integro numéricamente para obtener la energía total de la señal
df, dt = f[1] - f[0], t[1] - t[0]
E_Zxx = sum(np.sum(Zxx.real**2 + Zxx.imag**2, axis=0) * df) * dt

print(E_x)
print(E_Zxx)