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

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Ruta del archivo
ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S308/3S308.csv"

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

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Defino la señal de aceleración vertical traída directamente del sensor
acc_vert = acel[:,1] - constants.g

## Defino la aceleración anteroposterior
acc_AP = acel[:,2]

## Defino filtro de Butterworth pasabanda con banda pasante [0.5, 2.5] Hz para poder extraer el armónico principal
sos = signal.butter(N = 4, Wn = [0.5, 2.5], btype = 'bandpass', fs = 1 / periodoMuestreo, output = 'sos')

## Aplico el filtrado a la señal de acleracion vertical para obtener el armónico principal
acc_vert_filtrada = signal.sosfiltfilt(sos, acc_vert)

## Aplico el mismo filtro para obtener la aceleración anteroposterior
acc_AP_filtrada = signal.sosfiltfilt(sos, acel[:,2])

# TransformadaFourier(acel[:,2], periodoMuestreo)

#plt.plot(acc_vert_filtrada)
plt.plot(acc_AP_filtrada)
#plt.plot(acel[:,2] - np.mean(acel[:,2]))
plt.show()

GraficacionPicos(-acc_AP, picos_sucesivos)