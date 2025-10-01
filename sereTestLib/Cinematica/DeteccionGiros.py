## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from scipy.signal import *
from LecturaDatos import *
from matplotlib import pyplot as plt
from scipy.signal import *
import pywt
from scipy.integrate import cumulative_trapezoid
from Fourier import *
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/KielMAT-main/kielmat/modules/td')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/KielMAT-main/kielmat')
from _pham import *

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Ruta donde voy a abrir el archivo
ruta_registro = "C:/Yo/Tesis/sereData/sereData/Registros/MarchaEstandar_Rodrigo.txt"

## Lectura de los datos
data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = 299, lectura_datos_propios = True, ruta = ruta_registro)

## ---------------------------------------- DETECCIÓN DE GIROS -----------------------------------------

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

## Defino la variable que contiene la posición de la muestra en la que estoy parado, la cual se inicializa en 0
ubicacion_muestra = 0

## Defino un tamaño de muestras por ventana
muestras_ventana = 400

## Defino un tamaño de muestras de solapamiento entre ventanas
muestras_solapamiento = 0

## Integro aceleración angular para obtener el ángulo de giro
angulos_y = cumulative_trapezoid(gyro_y, dx = periodoMuestreo, initial = 0)

## Pregunto si ya terminó de recorrer todo el vector de muestras
while ubicacion_muestra < cant_muestras:

    ## En caso que la última ventana no pueda tener el tamaño predefinido, la seteo manualmente
    if (cant_muestras - ubicacion_muestra < muestras_ventana):    

        ## Me quedo con el segmento del giroscopio GY correspondiente
        segmento_GY = angulos_y[ubicacion_muestra :]

    ## En otro caso, digo que la ventana tenga el tamaño predefinido
    else:

        ## Me quedo con el segmento del giroscopio GY correspondiente
        segmento_GY = angulos_y[ubicacion_muestra : ubicacion_muestra + muestras_ventana]

    ## Actualizo el valor de la ubicación de la muestra para que me guarde la posición en la que debe comenzar la siguiente ventana
    ubicacion_muestra += muestras_ventana - muestras_solapamiento

## ---------------------------------------- MÉTODO KIELMAT -----------------------------------------

## Creo un objeto detector de giros
pham = PhamTurnDetection()

## Hago la detección de giros
pham.detect(
        accel_data = pd.DataFrame(acel),
        gyro_data = pd.DataFrame(gyro),
        gyro_vertical ="pelvis_GYRO_y",
        sampling_freq_Hz = 200,
        tracking_system = "imu",
        tracked_point ="LowerBack",
        plot_results = False
        )