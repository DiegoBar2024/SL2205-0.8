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
from Segmentacion import picos_sucesivos, frec_fund, pasos
import emd

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S274/3S274.csv"

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

## ------------------------------------- INTEGRACIÓN ACELERACIÓN ---------------------------------------

## Defino la señal de aceleración en dirección vertical
acc_vert = acel[:,1] - constants.g

## Integro aceleración para obtener velocidad
vel_vert = cumulative_trapezoid(acc_vert - np.mean(acc_vert), dx = periodoMuestreo, initial = 0)

## ----------------------------------- DESCOMPOSICIÓN EMD VELOCIDAD ------------------------------------

## Hago la descomposición EMD de la señal de velocidad
imfs_vel_vert = emd.sift.sift(vel_vert)

vel_vert_procesada = imfs_vel_vert[:,0] + imfs_vel_vert[:,1] + imfs_vel_vert[:,2] + imfs_vel_vert[:,3]

# ## Graficación de la señal
# plt.plot(vel_vert_procesada)
# plt.show()

## -------------------------------------- INTEGRACIÓN VELOCIDAD ----------------------------------------

## Integro velocidad para obtener posición
pos_vert = cumulative_trapezoid(vel_vert_procesada, dx = periodoMuestreo, initial = 0)

## ----------------------------------- DESCOMPOSICIÓN EMD POSICIÓN -------------------------------------

## Hago la descomposición EMD de la señal de posición
imfs_pos_vert = emd.sift.sift(pos_vert)

# ## Graficación de las IMFs de la señal de posición
# emd.plotting.plot_imfs(imfs_pos_x, scale_y = True, cmap = True)

pos_vert_procesada = imfs_pos_vert[:,0] + imfs_pos_vert[:,1] + imfs_pos_vert[:,2]

# ## Graficación de la señal
# plt.plot(pos_vert_procesada)
# plt.show()

## -------------------------------------- SEGMENTACIÓN DE PASOS ----------------------------------------

## Creo una lista vacía en donde voy a guardar los pasos segmentados
segmentada = []

## Itero para cada uno de los pasos que tengo detectados
for i in range (len(pasos) - 1):

    ## Hago la segmentación de la señal
    segmento = pos_vert_procesada[pasos[i][0] : pasos[i][1]]

    ## Luego lo agrego a la señal segmentada
    segmentada.append(segmento)

## -------------------------------- VARIACIÓN DE ALTURA CENTRO DE MASA ----------------------------------

## Creo una lista donde voy a almacenar las variaciones de altura del centro de masa en cada tramo
desp_vert_COM = []

## Itero para cada uno de los segmentos de pasos detectados
for i in range (len(segmentada)):

    ## Calculo la variación de altura del centro de masa en base a la diferencia entre el desplazamiento vertical máximo y mínimo
    d_step = abs(max(segmentada[i]) - min(segmentada[i]))

    ## Agrego el desplazamiento máximo del COM calculado a la lista
    desp_vert_COM.append(d_step)

## ---------------------------- GRAFICACIÓN VARIACIÓN ALTURA CENTRO DE MASA -----------------------------

# plt.scatter(x = np.arange(start = 0, stop = len(desp_vert_COM)), y = desp_vert_COM)
# plt.show()

## --------------------------------- CÁLCULO DE LA LONGITUD DEL PASO ------------------------------------

## Especifico la longitud de la pierna del individuo en metros
## Ésto debe considerarse como una entrada al sistema. Es un parámetro que puede medirse
## ¡IMPORTANTE: ÉSTE PARÁMETRO CAMBIA CON CADA PERSONA! SINO EL RESULTADO DA CUALQUIER COSA
long_pierna = 0.85

## Creo una lista donde voy a almacenar la longitud de los pasos de la persona
long_pasos = []

## Itero para cada uno de los segmentos de pasos detectados
for i in range (len(segmentada)):
    
    ## Calculo la longitud del paso con la fórmula sugerida por Zijlstra
    long_paso = 2 * np.sqrt(2 * long_pierna * desp_vert_COM[i] - desp_vert_COM[i] ** 2)

    ## Agrego el paso a la lista de longitud de pasos
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