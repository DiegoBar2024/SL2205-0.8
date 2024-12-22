## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')

from Magnitud import Magnitud
from Normalizacion import Normalizacion
from DeteccionPicos import *
from Filtros import FiltroMediana
from Fourier import TransformadaFourier
import numpy as np
import pandas as pd
from Muestreo import PeriodoMuestreo
import pandas as pd
from ValoresMagnetometro import ValoresMagnetometro
from matplotlib import pyplot as plt
from scipy import signal
from harm_analysis import *
from control import *
from skinematics.imus import analytical, IMU_Base
from scipy import *
from scipy.integrate import cumulative_trapezoid, simpson
import librosa
import pywt

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Identificación del paciente
numero_paciente = '221'

## Ruta del archivo
## {'Sentado':'1','Parado':'2','Caminando':'3','Escalera':'4'}
ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S{}/3S{}.csv".format(numero_paciente, numero_paciente)

## Lectura de datos
data = pd.read_csv(ruta)

## Hallo el período de muestreo de las señales
periodoMuestreo = PeriodoMuestreo(data)

## Armamos una matriz donde las columnas sean las aceleraciones
acel = np.array([np.array(data['AC_x']), np.array(data['AC_y']), np.array(data['AC_z'])]).transpose()

## Armamos una matriz donde las columnas sean los valores de los giros
gyro = np.array([np.array(data['GY_x']), np.array(data['GY_y']), np.array(data['GY_z'])]).transpose()

## Separo el vector de tiempos del dataframe
tiempo = np.array(data['Time'])

## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
tiempo = (tiempo - tiempo[0]) / 1000

## Cantidad de muestras de la señal
cant_muestras = len(tiempo)

## ---------------------------------------- FILTRADO DE MEDIANA ----------------------------------------

## Aplico filtrado de mediana con una ventana de tamaño 3 a la aceleración mediolateral
acc_ML = signal.medfilt(volume = acel[:,0], kernel_size = 3)

## Aplico filtrado de mediana con una ventana de tamaño 3 a la aceleración vertical
acc_VT = signal.medfilt(volume = acel[:,1], kernel_size = 3)

## Aplico filtrado de mediana con una ventana de tamaño 3 a la aceleración anteroposterior
acc_AP = signal.medfilt(volume = acel[:,2], kernel_size = 3)

## -------------------------------------- GRAFICACIÓN DE SEÑALES ---------------------------------------

## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
plt.plot(tiempo, acc_ML, color = 'r', label = 'Aceleración ML')
plt.plot(tiempo, acc_VT, color = 'b', label = 'Aceleración VT')
plt.plot(tiempo, acc_AP, color = 'g', label = 'Aceleración AP')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleracion $(m/s^2)$")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()

## ---------------------------------------- FILTRADO PASABAJOS -----------------------------------------

## Basado en la publicación "Implementation of a Real-Time Human Movement Classifier Using a Triaxial Accelerometer for Ambulatory Monitoring"
## Defino un filtro pasabajos IIR elíptico de tercer orden con frecuencia de corte 0.25Hz
## El filtro tiene 0.01dB ripple máximo en banda pasante y atenuación mínima de -100dB en banda supresora
sos_iir = signal.iirfilter(3, 0.25, btype = 'lowpass', analog = False, rp = 0.01, rs = 100, ftype = 'ellip', output = 'sos', fs = 1 / periodoMuestreo)

## Se seleccionan las componentes de GA (Gravity Acceleration) haciendo un filtrado pasabajos a las señales
## Filtrado de la señal de aceleración Mediolateral
acc_ML_GA = signal.sosfiltfilt(sos_iir, acc_ML)

## Filtrado de la señal de aceleración Vertical
acc_VT_GA = signal.sosfiltfilt(sos_iir, acc_VT)

## Filtrado de la señal de aceleración Anteroposterior
acc_AP_GA = signal.sosfiltfilt(sos_iir, acc_AP)

## -------------------------------------- GRAFICACIÓN DE SEÑALES ---------------------------------------

## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
plt.plot(tiempo, acc_ML_GA, color = 'r', label = 'Aceleración ML')
plt.plot(tiempo, acc_VT_GA, color = 'b', label = 'Aceleración VT')
plt.plot(tiempo, acc_AP_GA, color = 'g', label = 'Aceleración AP')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleracion $(m/s^2)$")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()

## --------------------------------- OBTENCIÓN DE COMPONENTES BA ---------------------------------------

## En base a las componentes de GA (Gravity Acceleration) en los tres ejes, se obtienen los componentes de BA (Body Acceleration) en los tres ejes
## Obtengo la componente de BA en dirección mediolateral
acc_ML_BA = acc_ML - acc_ML_GA

## Obtengo la componente de BA en dirección vertical
acc_VT_BA = acc_VT - acc_VT_GA 

## Obtengo la componente de BA en dirección anteroposterior
acc_AP_BA = acc_AP - acc_AP_GA

## -------------------------------------- GRAFICACIÓN DE SEÑALES ---------------------------------------

## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
plt.plot(tiempo, acc_ML_BA, color = 'r', label = 'Aceleración ML')
plt.plot(tiempo, acc_VT_BA, color = 'b', label = 'Aceleración VT')
plt.plot(tiempo, acc_AP_BA, color = 'g', label = 'Aceleración AP')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleracion $(m/s^2)$")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()

## ------------------------------------------- SEGMENTACIÓN --------------------------------------------

## Defino la cantidad de muestras que tiene cada ventana
## Éste acercamiento tiene como hipótesis el análisis con ventanas de tamaño fijo
## La cantidad de muestras por ventana y la cantidad de muestras del solapamiento son parámetros que se pueden variar
muestras_ventana = 200

## Defino la cantidad de muestras que tienen los solapamientos
muestras_solapamiento = 100

## Defino la variable que contiene la posición de la muestra en la que estoy parado, la cual se inicializa en 0
ubicacion_muestra = 0

## Genero un vector para guardar los valores de SMA computados
vector_SMA = []

print(periodoMuestreo * muestras_ventana)

## Pregunto si ya terminó de recorrer todo el vector de muestras
while ubicacion_muestra < cant_muestras:

    ## En caso que la última ventana no pueda tener el tamaño predefinido, la seteo manualmente
    if (cant_muestras - ubicacion_muestra < muestras_ventana):
    
        ## Me quedo con el segmento de la aceleración mediolateral (componente BA)
        segmento_ML = acc_ML_BA[ubicacion_muestra :]

        ## Me quedo con el segmento de la aceleración vertical (componente BA)
        segmento_VT = acc_VT_BA[ubicacion_muestra :]

        ## Me quedo con el segmento de la aceleración anteroposterior (componente BA)
        segmento_AP = acc_AP_BA[ubicacion_muestra :]
    
    ## En otro caso, digo que la ventana tenga el tamaño predefinido
    else:

        ## Me quedo con el segmento de la aceleración mediolateral (componente BA)
        segmento_ML = acc_ML_BA[ubicacion_muestra : ubicacion_muestra + muestras_ventana]

        ## Me quedo con el segmento de la aceleración vertical (componente BA)
        segmento_VT = acc_VT_BA[ubicacion_muestra : ubicacion_muestra + muestras_ventana]

        ## Me quedo con el segmento de la aceleración anteroposterior (componente BA)
        segmento_AP = acc_AP_BA[ubicacion_muestra : ubicacion_muestra + muestras_ventana]

    ## Hago el cálculo del SMA (Signal Magnitude Area) para dicha ventana siguiendo la definición
    SMA = (np.sum(np.abs(segmento_ML)) + np.sum(np.abs(segmento_VT)) + np.sum(np.abs(segmento_AP))) / (periodoMuestreo * muestras_ventana)

    ## Agrego el valor computado de SMA al vector correspondiente
    vector_SMA.append(SMA)

    ## Actualizo el valor de la ubicación de la muestra para que me guarde la posición en la que debe comenzar la siguiente ventana
    ubicacion_muestra += muestras_ventana - muestras_solapamiento

## --------------------------------------- GRAFICACIÓN VALORES SMA -------------------------------------

## Gráfica de Scatter
plt.scatter(x = np.arange(start = 0, stop = len(vector_SMA)), y = vector_SMA)
plt.show()

## Gráfica de Boxplot
plt.boxplot(vector_SMA)
plt.show()
