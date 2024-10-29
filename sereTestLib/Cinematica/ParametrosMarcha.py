from Magnitud import Magnitud
from Normalizacion import Normalizacion
from DeteccionPicos import *
from Filtros import FiltroMediana
from Fourier import TransformadaFourier
import numpy as np
from Muestreo import PeriodoMuestreo
import pandas as pd
from ValoresMagnetometro import ValoresMagnetometro
from matplotlib import pyplot as plt
from scipy import signal
from harm_analysis import *
from control import *

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S278/3S278.csv"

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

## ----------------------------------------- PREPROCESADO ------------------------------------------

## Se calcula la magnitud de la señal de aceleración
magnitud = Magnitud(acel)

## Se hace la normalización en amplitud y offset de la señal de magnitud
mag_normalizada = Normalizacion(magnitud)

## Se realiza un filtrado de medianas para eliminar algunos picos no relevantes de la señal
normal_filtrada = signal.medfilt(mag_normalizada, kernel_size = 11)

## ----------------------------------------- ANÁLISIS EN FRECUENCIA ---------------------------------------------------

## Defino la señal a procesar
señal = magnitud

## Se obtiene el espectro de toda la señal completa aplicando la transformada de Fourier
## Ya que es una señal real se cumple la simetría conjugada
(frecuencias, transformada) = TransformadaFourier(señal, periodoMuestreo)

## Obtengo información de los armónicos de la señal mediante el uso de <<harm_analysis>>
armonicos = harm_analysis(señal, FS = 1 / periodoMuestreo)

## Obtengo frecuencia fundamental de la señal
frec_fund = armonicos['fund_freq']

## Potencia de la componente fundamental de la señal
fund_db = armonicos['fund_db']

## Hago el pasaje del valor obtenido en decibeles a magnitud
fund_mag = db2mag(fund_db)

## Potencias de los armónicos no fundamentales
no_fund_db = np.array(armonicos['pot_armonicos'])

## Hago el pasaje de los valores obtenidos en decibeles a magnitud
no_fund_mag = db2mag(no_fund_db)

## ----------------------------------------- DETECCIÓN DE PICOS ------------------------------------------

## Cálculo de umbral óptimo
(T, stdT) = CalculoUmbral(señal = normal_filtrada)

## Se hace el llamado a la función de detección de picos configurando un umbral de entrada T
picos = DeteccionPicos(normal_filtrada, umbral = T)

## Cálculo del índice donde se encuentra ciclo potencial inicial
IPC = CicloPotencialInicial(picos)

## Se hace la graficación de la señal marcando los picos
GraficacionPicos(normal_filtrada, picos)

## Obtengo el vector con las separaciones de los picos
separaciones = SeparacionesPicos(picos)

## Hago la traduccion de muestras a valores de tiempo usando la frecuencia de muestreo dada
sep_tiempos = separaciones * periodoMuestreo

## Valor medio de la separación de tiempos
tiempo_medio = np.mean(sep_tiempos)

## Desviación estándar de la separación de tiempos
desv_tiempos = np.std(sep_tiempos)