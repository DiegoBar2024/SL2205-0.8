## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from scipy.signal import *
from LecturaDatos import *
from matplotlib import pyplot as plt
import pywt
from Fourier import *

## ---------------------------------------- DETECCIÓN DE GIROS -----------------------------------------

## Especifico el identificador del paciente cuyos datos van a ser procesados
id_persona = 302

## Hago la lectura de los datos del registro de marcha del paciente
data, acel, gyro, cant_muestras, periodoMuestreo, nombre_persona, nacimiento_persona, tiempo = LecturaDatos(id_persona, lectura_datos_propios = True)

## Obtengo el valor del giroscopio en el eje x
gyro_x = gyro[:, 0]

## Obtengo el valor del giroscopio en el eje y
gyro_y = gyro[:, 1]

## Obtengo el valor dek giroscopio en el eje z
gyro_z = gyro[:, 2]

## Grafico los datos. En mi caso las tres velocidades angulares
plt.plot(tiempo, gyro_x, color = 'r', label = '$w_x$')
plt.plot(tiempo, gyro_y, color = 'b', label = '$w_y$')
plt.plot(tiempo, gyro_z, color = 'g', label = '$w_z$')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la velocidad angular (rad/s)
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad angular (rad/s)")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()

## Etapa de filtrado pasabajos de Butterworth con frecuencia de corte 0.5Hz de orden 4
sos = butter(N = 4, Wn = 0.5, btype = 'lowpass', fs = 1 / periodoMuestreo, output = 'sos')

## La cantidad de tiempo que transcurre entre dos valles debe ser igual al tiempo de paso
gyro_y_filtrada = sosfiltfilt(sos, gyro_y)

## Grafico los datos. En mi caso las tres velocidades angulares
plt.plot(tiempo, gyro_y_filtrada, color = 'b', label = '$w_y$')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la velocidad angular (rad/s)
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad angular (rad/s)")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()

# ## Escalas de las wavelets a utilizar
# ## Se recuerda que la pseudo - frecuencia me queda f = f_muestreo / escala
# ## Ésto me implica que a escalas más pequeñas tengo frecuencias más grandes y viceversa
# escalas = np.arange(8, 200, 1)

# ## Creo una variable la cual almacene el ancho de banda de la wavelet
# ancho_banda = 1.5

# ## Tipo de wavelet a utilizar. Wavelet de Morlet Compleja
# ## Parámetro B (Ancho de banda): 1.5 Hz (ajustable)
# ## Parámetro C (Frecuencia Central): 1 Hz
# wavelet = 'cmor{}-1'.format(ancho_banda)

# ## Transformada de Wavelet de la aceleración en el eje x
# coef1, scales_freq = pywt.cwt(data = gyro_y, scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

# ## Graficación del escalograma en el plano tiempo frecuencia
# data = np.abs(coef1)
# cmap = plt.get_cmap('jet', 256)
# fig = plt.figure(figsize = (5,5))
# ax = fig.add_subplot(111)
# t = np.arange(coef1.shape[1]) * periodoMuestreo
# ax.pcolormesh(t, scales_freq, data, cmap = cmap, vmin = data.min(), vmax = data.max(), shading = 'auto')
# plt.xlabel("Tiempo (s)")
# plt.ylabel("Frecuencia (Hz)")
# plt.title("$|CWT_{ML}(t,f)|$")
# plt.show()
