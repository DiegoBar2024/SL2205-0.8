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
import emd

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

## ------------------------------------- INTEGRACIÓN ACELERACIÓN ---------------------------------------

## Defino la señal de aceleración anteroposterior como una aproximación directamente del sensor
acc_ap = acel[:,2]

## Elimino el valor medio de dicha señal
acc_ap_smedia = acc_ap - np.mean(acc_ap)

## Integro aceleración para obtener velocidad
vel_x = cumulative_trapezoid(acc_ap_smedia, dx = periodoMuestreo, initial = 0)

## ----------------------------------- DESCOMPOSICIÓN EMD VELOCIDAD ------------------------------------

## Hago la descomposición EMD de la señal de velocidad
imfs_vel_x = emd.sift.sift(vel_x)

# ## Graficación de las IMFs de la señal de velocidad
# emd.plotting.plot_imfs(imfs_vel_x, scale_y = True, cmap = True)

## Me quedo con los primeros 3 modos de IMF
vel_x_procesada = imfs_vel_x[:,0] + imfs_vel_x[:,1] + imfs_vel_x[:,2] + imfs_vel_x[:,3]

## Graficación de la señal
plt.plot(vel_x_procesada)
plt.show()

## -------------------------------------- INTEGRACIÓN VELOCIDAD ----------------------------------------

## Integro velocidad para obtener posición
pos_x = cumulative_trapezoid(vel_x_procesada, dx = periodoMuestreo, initial = 0)

## ----------------------------------- DESCOMPOSICIÓN EMD POSICIÓN -------------------------------------

## Hago la descomposición EMD de la señal de posición
imfs_pos_x = emd.sift.sift(pos_x)

# ## Graficación de las IMFs de la señal de posición
# emd.plotting.plot_imfs(imfs_pos_x, scale_y = True, cmap = True)

pos_x_procesada = imfs_pos_x[:,0] + imfs_pos_x[:,1] + imfs_pos_x[:,2]

## Graficación de la señal
plt.plot(pos_x_procesada)
plt.show()

## -------------------------------------- SEGMENTACIÓN DE PASOS ----------------------------------------

## Creo una lista vacía en donde voy a guardar los pasos segmentados para la aceleración
segmentada_acc = []

## Creo una lista donde voy a guardar los paso segmentados para la posición anteroposterior
segmentada_pos = []

## Itero para cada uno de los pasos que tengo detectados
for i in range (len(picos_sucesivos) - 1):

    ## Hago la segmentación de la señal de aceleración
    segmento_acc = acc_ap[picos_sucesivos[i] : picos_sucesivos[i + 1]]

    ## Luego lo agrego a la señal de aceleración segmentada
    segmentada_acc.append(segmento_acc)

    ## Hago la segmentación de la señal de posición
    segmento_pos = pos_x_procesada[picos_sucesivos[i] : picos_sucesivos[i + 1]]

    ## Luego lo agrego a la señal de posición segmentada
    segmentada_pos.append(segmento_pos)

## --------------------------------- CÁLCULO DE LA LONGITUD DEL PASO -----------------------------------

## Creo una lista donde voy a almacenar la longitud de los pasos de la persona
long_pasos = []

## Itero para cada uno de los segmentos de pasos detectados
for i in range (len(segmentada_acc)):

    ## Calculo el valor medio de la aceleración anteroposterior en el segmento del paso
    acc_ap_medio = np.mean(segmentada_acc[i])

    ## Calculo la diferencia de desplazamiento máxima anteroposterior en el segmento del paso
    dmax = abs(max(segmentada_pos[i]) - min(segmentada_pos[i]))

    print(dmax)

    ## Hago la estimación de la longitud del paso en base a la fórmula
    long_paso = acc_ap_medio * dmax

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