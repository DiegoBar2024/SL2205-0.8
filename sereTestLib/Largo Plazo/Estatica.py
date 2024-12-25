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
from scipy import constants
from scipy.integrate import cumulative_trapezoid, simpson
import librosa
import pywt

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Identificación del paciente
numero_paciente = '252'

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

## -------------------------------------- MAGNITUD DE ACELERACIÓN --------------------------------------

## Calculo la magnitud de la señal de aceleración
mag_acc = np.sqrt(acel[:,0] ** 2 + acel[:,1] ** 2 + acel[:,2] ** 2)

## -------------------------------- ERROR ABSOLUTO MAGNITUD - GRAVEDAD ----------------------------------

## En caso de que la actividad sea estática (sentado, parado) la magnitud de la aceleración debe ser muy similar a la aceleración gravitatoria
## La magnitud se calcula para hacer el análisis más inmune a la orientación del sensor
## Las diferencias entre la magnitud de la aceleración y gravedad pueden deberse a la presencia de ruido en las mediciones
## Calculo el error absoluto entre la magnitud de la aceleración y la gravedad
err_abs = mag_acc - constants.g

## Hago la gráfica del error absoluto
plt.plot(tiempo, err_abs)
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleracion $(m/s^2)$")
plt.show()
