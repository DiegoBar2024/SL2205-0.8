from Magnitud import Magnitud
from Normalizacion import Normalizacion
from DeteccionPicos import *
import numpy as np
from Muestreo import PeriodoMuestreo
import pandas as pd
from ValoresMagnetometro import ValoresMagnetometro
from matplotlib import pyplot as plt
from scipy import signal

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S302/3S302.csv"

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

## Se calcula la magnitud de la señal de aceleración
magnitud = Magnitud(acel)

## Se hace la normalización en amplitud y offset de la señal de magnitud
mag_normalizada = Normalizacion(magnitud)

## Se hace el llamado a la función de detección de picos configurando un umbral de entrada
picos = DeteccionPicos(mag_normalizada, umbral = 0.3)

## Se hace la graficación de la señal marcando los picos
GraficacionPicos(mag_normalizada, picos)

## Obtengo el vector con las separaciones de los picos
separaciones = SeparacionesPicos(picos)

## Hago la traduccion de muestras a valores de tiempo usando la frecuencia de muestreo dada
sep_tiempos = separaciones * periodoMuestreo

## Valor medio de la separación de tiempos
tiempo_medio = np.mean(sep_tiempos)

## Desviación estándar de la separación de tiempos
desv_tiempos = np.std(sep_tiempos)

print(sep_tiempos, tiempo_medio, desv_tiempos)