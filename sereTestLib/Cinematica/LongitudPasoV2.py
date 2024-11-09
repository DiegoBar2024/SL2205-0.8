import skinematics.quat
from skinematics.imus import analytical, IMU_Base
from skinematics.sensors import xsens
from skinematics.quat import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Integracion import CalcularVelocidades
from Fourier import TransformadaFourier
from ValoresMagnetometro import ValoresMagnetometro
from Muestreo import PeriodoMuestreo
from scipy import constants
from skinematics.sensors.xsens import XSens
import numpy as np
from scipy import integrate
from Filtros import FiltroMediana
from scipy.integrate import cumulative_trapezoid, simpson
from Magnitud import *
from Normalizacion import *
from DeteccionPicos import * 
from Segmentacion import picos_sucesivos, frec_fund

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S299/3S299.csv"

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

## ----------------------------------- FILTRADO ACELERACIÓN VERTICAL -----------------------------------

## Separo la señal de aceleración vertical
acc_vertical = acel[:,1] - constants.g

# # Aplico Filtro Butterworth Pasabajos de fc = 3Hz de orden 4 a la señal de aceleracion vertical
# sos = signal.butter(N = 4, Wn = 3, btype = 'lowpass', fs = 1 / periodoMuestreo, output = 'sos')

## Una posibilidad de que el algoritmo con el highpass no ande bien es por el drift que se acumula en la corrección de ejes por los datos del giroscopio
## Para mitigar ésto puedo hacer un filtrado pasabanda con el mismo orden pero tomando fc1 = 0.1Hz y fc2 = 3Hz como las frecuencias de corte
sos = signal.butter(N = 4, Wn = [0.1, 3], btype = 'bandpass', fs = 1 / periodoMuestreo, output = 'sos')

## Aplico el procesado para obtener la señal filtrada
acc_vertical_filtrada = signal.sosfiltfilt(sos, acc_vertical)

## --------------------------------------- SEGMENTACIÓN DE PASOS ----------------------------------------

## Creo una lista vacía en donde voy a guardar los pasos segmentados
segmentada = []

## Itero para cada uno de los pasos que tengo detectados
for i in range (len(picos_sucesivos) - 1):

    ## Hago la segmentación de la señal
    segmento = acc_vertical_filtrada[picos_sucesivos[i] : picos_sucesivos[i + 1]]

    ## Luego lo agrego a la señal segmentada
    segmentada.append(segmento)

## ------------------------------- VARIACIÓN DE LA ACELERACIÓN VERTICAL ---------------------------------

## Creo una lista donde voy a guardar las variaciones de las aceleraciones verticales
var_acc_vert = []

## Itero para cada uno de los segmentos de pasos detectados
for i in range (len(segmentada)):

    ## Calculo la variación máxima de la aceleración en dirección del eje vertical
    acc_max_min = abs(max(segmentada[i]) - min(segmentada[i]))

    ## Agrego el valor calculado a la lista
    var_acc_vert.append(acc_max_min)

## --------------------------------- CÁLCULO DE LA LONGITUD DEL PASO ------------------------------------

## Creo una lista donde voy a almacenar la longitud de los pasos de la persona
long_pasos = []

## Itero para cada uno de los segmentos de pasos detectados
for i in range (len(segmentada)):

    ## Aplico la fórmula de la raíz cuarta al valor acc_max_min en cada paso para obtener la estimacion
    long_paso = np.power(var_acc_vert[i], 1/4)

    ## Guardo el valor del paso en la lista
    long_pasos.append(long_paso)

## --------------------------------- CÁLCULO DE LA VELOCIDAD DE MARCHA -----------------------------------

## Se calcula la longitud de paso promedio
long_paso_promedio = np.mean(long_pasos)

## Se calcula la duración de paso promedio como el inverso de la cadencia
tiempo_paso_promedio = 1 / frec_fund

## Se calcula la velocidad de marcha como el cociente entre éstas cantidades
velocidad_marcha = long_paso_promedio / tiempo_paso_promedio

print("Longitud paso promedio (m): ", long_paso_promedio)
print("Duracion de paso promedio (s): ", tiempo_paso_promedio)
print("Velocidad de marcha (m/s): ", velocidad_marcha)

## ----------------------------------- GRAFICACIÓN LONGITUD DE PASOS -------------------------------------

plt.scatter(x = np.arange(start = 0, stop = len(long_pasos)), y = long_pasos)
plt.show()