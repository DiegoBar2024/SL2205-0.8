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

## -------------------------------------- FILTRADO ACELERACIÓN ---------------------------------------

## Defino la señal de aceleración mediolateral como la aceleración medida por el sensor
acel_ML = acel[:,0]

## Hago un filtrado pasabanda para quedarme con la componente fundamental
sos = signal.butter(N = 4, Wn = [0.5, 1.1], btype = 'bandpass', fs = 1 / periodoMuestreo, output = 'sos')

## Aplico la etapa de filtrado a la señal
acel_ML_filtrada = signal.sosfiltfilt(sos, acel_ML)

## -------------------------------------- DISTINCIÓN DE PASOS ------------------------------------------

## Creo una lista en donde voy a guardar la pierna con la cual se dio el paso
## Asigno 1 para una pierna y 0 para otra pierna
## Una vez que sepamos la dirección de los ejes puedo asignar 1 o 0 como izquierda o derecha
piernas_pasos = []

## Itero para cada uno de los pasos que tengo detectados
for i in range (len(pasos) - 1):

    ## Hago la segmentación de la señal
    segmento = acel_ML_filtrada[pasos[i][0] : pasos[i][1]]

    ## Calculo la diferencia del segmento para ver una aproximación de la derivada en cada paso
    ## Sean x = a el extremo izquierdo y x = b el extremo derecho
    ## Si f'(a) - f'(b) > 0 --> Concluyo que tengo un máximo local en el segmento (concavidad positiva)
    ## Si f'(a) - f'(b) < 0 --> Concluyo que tengo un mínimo local en el segmento (concavidad negativa)
    diff_segmento = np.diff(segmento)

    ## En caso de tener concavidad positiva en el segmento (maximo local)
    if (diff_segmento[0] - diff_segmento[-1] > 0):

        ## Asigno la pierna 0 como la que dio el paso
        piernas_pasos.append(0)
    
    ## En caso de tener concavidad negativa en el segmento (minimo local)
    else:

        ## Asigno la pierna 0 como la que dio el paso
        piernas_pasos.append(1)

## -------------------------------------- ASOCIACIÓN DE PASOS ------------------------------------------

## Creo una lista donde guardo los pasos que tengo agregando la orientación de la pierna
pasos_orientacion = []

## Itero para cada uno de los pasos detectados
for i in range (len(pasos) - 1):

    ## Asocio cada paso con su orientación según lo que calculé antes
    pasos_orientacion.append((piernas_pasos[i], pasos[i][0], pasos[i][1]))