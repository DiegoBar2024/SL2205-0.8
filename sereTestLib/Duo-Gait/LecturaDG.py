## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Especifico la ruta en la cual se encuentra el archivo
ruta = 'C:/Yo/Tesis/raw/raw/OG_dt_raw/sub_14/SA.csv'

## Selecciono las columnas deseadas
data = pd.read_csv(ruta, on_bad_lines = 'skip').drop([0])[["Time", "Gyro X", "Gyro Y", "Gyro Z", "Accel X", "Accel Y", "Accel Z"]]

## Armamos una matriz donde las columnas sean las aceleraciones
acel = np.array([np.array(data["Accel X"]), np.array(data["Accel Y"]), np.array(data["Accel Z"])]).transpose().astype(float)

## Armamos una matriz donde las columnas sean los valores de los giros
gyro = np.array([np.array(data["Gyro X"]), np.array(data["Gyro Y"]), np.array(data["Gyro Z"])]).transpose().astype(float)

## Separo el vector de tiempos del dataframe
tiempo = np.array(data['Time']).astype(float)

## Grafico los datos. En mi caso las tres velocidades angulares
plt.plot(tiempo, acel[:, 0], color = 'r', label = '$a_x$')
plt.plot(tiempo, acel[:, 1], color = 'b', label = '$a_y$')
plt.plot(tiempo, acel[:, 2], color = 'g', label = '$a_z$')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleracion $(m/s^2)$")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()