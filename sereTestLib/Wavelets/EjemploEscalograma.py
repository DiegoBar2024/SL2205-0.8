import sys
import pywt
from matplotlib import pyplot as plt
from scipy.signal import *
import numpy as np
import pathlib
sys.path.append(str(pathlib.Path().resolve()).replace('\\','/') + '/sereTestLib')
from parameters import *
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from matplotlib import cm, colors
import pywt        

## Período de muestreo
T = 1 / 400

## Vector de tiempos
t = np.arange(0, 10, T)

## Señal
x = chirp(t, 0, 1, 200)

## Escalas
s = np.arange(1, 201, 1)

## Creo una variable la cual almacene el ancho de banda de la wavelet
ancho_banda = 1.5

## Tipo de wavelet a utilizar. Wavelet de Morlet Compleja
## Parámetro B (Ancho de banda): 1.5 Hz (ajustable)
## Parámetro C (Frecuencia Central): 1 Hz
wavelet = 'cmor{}-1'.format(ancho_banda)

## Descomposición en CWT
coef, scales_freq = pywt.cwt(data = x, scales = s, wavelet = wavelet, sampling_period = T)

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))

axes[0].plot(t, x)
axes[0].set_xlabel("t(s)")
axes[0].set_ylabel("x(t)$")

data = np.abs(coef)
cmap = plt.get_cmap('jet', 256)
t = np.arange(coef.shape[1]) * T
axes[1].pcolormesh(t, scales_freq, data, cmap = cmap, vmin = data.min(), vmax = data.max(), shading = 'auto')
fig.colorbar(cm.ScalarMappable(norm = colors.Normalize(np.min(data), np.max(data)), cmap = cmap), ax = axes[1])
axes[1].set_xlabel("Tiempo (s)")
axes[1].set_ylabel("Frecuencia (Hz)")
axes[1].set_title("$|CWT_{x}(t,f)|$")

# Adjust layout to prevent titles and labels from overlapping
plt.tight_layout()

# Display the figure with both plots
plt.show()