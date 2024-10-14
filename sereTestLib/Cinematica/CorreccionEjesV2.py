## Se referencia la carpeta donde se encuentra la librería a utilizar
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/IMU-Position-Tracking-master')

import numpy as np
import pandas as pd
from ValoresMagnetometro import ValoresMagnetometro
from matplotlib import pyplot as plt
from main import IMUTracker

## Ruta al fichero
ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S133/3S133.csv"

## Calculamos los valores del magnetometro
data_mag = ValoresMagnetometro("C:/Yo/Tesis/sereData/sereData/Dataset/raw_2_process", 133, ['Caminando'])

## Lectura de datos
data = pd.read_csv(ruta)

# Armamos una matriz donde las columnas sean las aceleraciones
acel = np.array([np.array(data['AC_x']), np.array(data['AC_y']), np.array(data['AC_z'])]).transpose()

## Armamos una matriz donde las columnas sean los valores de los giros
gyro = np.array([np.array(data['GY_x']), np.array(data['GY_y']), np.array(data['GY_z'])]).transpose()

## Armamos una matriz donde las columnas sean los valores del magnetometro
mag = np.array([np.array(data_mag['Mag_x']), np.array(data_mag['Mag_y']), np.array(data_mag['Mag_z'])], dtype = float).transpose()

## Hago la concatenación de las tres matrices anterior para armar la matriz total de datos
matriz_datos = np.concatenate((gyro, acel, mag), axis = 1)

## Creación de un objeto IMUTracker (especifico frecuencia de muestreo)
IMU_tracker = IMUTracker(sampling = 200)

## Inicializo el IMUTracker pasando como parámetro de entrada la matriz de datos
## Obtengo como salida los parámetros resultantes de la inicialización
init_list = IMU_tracker.initialize(data = matriz_datos, noise_coefficient={'w': 0, 'a': 0, 'm': 0})

## Se aplica la función <<track>> a la matriz de datos
## El objetivo de obtener la aceleración corregida sin la gravedad en el sistema inercial
(a_nav, orix, oriy, oriz) = IMU_tracker.attitudeTrack(data = matriz_datos, init_list = init_list)

## Se aplica la función de removido de error para la acleración
## a_nav = IMU_tracker.removeAccErr(a_nav)

## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
plt.plot(a_nav[:,0], color = 'r', label = '$a_x$')
plt.plot(a_nav[:,1], color = 'b', label = '$a_y$')
plt.plot(a_nav[:,2], color = 'g', label = '$a_z$')
## plt.plot(acel[:,2], color = 'r', label = '$a_z$ s/corregir')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleracion $(m/s^2)$")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()