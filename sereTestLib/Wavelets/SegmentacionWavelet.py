## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from Cinematica.Segmentacion import *

## ------------------------------------- INTEGRACIÓN ACELERACIÓN ---------------------------------------

## Defino la señal de aceleración vertical como la señal medida en el eje z menos la gravedad (aproximación)
acel_vert = acel[:,1] - constants.g

## Integro aceleración para obtener velocidad
vel_z = cumulative_trapezoid(acel_vert, tiempo, dx = periodoMuestreo, initial = 0)

## --------------------------------------- FILTRADO VELOCIDAD ------------------------------------------

## Filtro de Butterworth de orden 4, pasaaltos, frecuencia de corte 0.5Hz
## La idea es aplicar un filtro intermedio a la velocidad vertical antes de volver a integrar para obtener posición
sos = signal.butter(N = 4, Wn = 0.5, btype = 'highpass', fs = 1 / periodoMuestreo, output = 'sos')

## Velocidad vertical luego de haber aplicado la etapa de filtrado pasaaltos
vel_z_filtrada = signal.sosfiltfilt(sos, vel_z)

## ---------------------------- TRANSFORMADA DE WAVELET ACELERACIÓN VERTICAL ---------------------------

## Calculo la CWT de la señal de aceleración vertical
coefs, scales_freq = pywt.cwt(data = vel_z_filtrada, scales = [100], wavelet = 'cmor1.5-1', sampling_period = periodoMuestreo)

## Hago la diferenciación de los coeficientes de la CWT
diff_coefs = np.diff(np.abs(coefs))[0]

plt.plot(diff_coefs*10)
plt.plot(acel[:,1])
plt.show()

