## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

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
from scipy import constants
from scipy.integrate import cumulative_trapezoid, simpson
import librosa
import pywt

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Identificación del paciente
numero_paciente = '299'

## Ruta del archivo
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

## ------------------------------------- ANÁLISIS EN FRECUENCIA ----------------------------------------

## Señal de aceleración vertical (resto la gravedad que me da una buena aproximación)
acc_VT = acel[:,1] - constants.g

## Filtro de Butterworth pasabanda de rango [0.5, 2.5] Hz para que me detecte el armónico fundamental
## A partir del armónico fundamental obtengo la cadencia
filtro = signal.butter(N = 4, Wn = [0.5, 2.5], btype = 'bandpass', fs = 1 / periodoMuestreo, output = 'sos')

## Aplico el filtro anterior a la aceleración anteroposterior
acel_filtrada = signal.sosfiltfilt(filtro, acc_VT)

## Hago la normalizacion de la aceleración vertical filtrada
acc_AP_norm = Normalizacion(acel_filtrada)

## Lo mismo con el opuesto
acc_AP_norm_TO = Normalizacion(- acel_filtrada)

## Se obtiene el espectro de toda la señal completa aplicando la transformada de Fourier
## Ya que es una señal real se cumple la simetría conjugada
(frecuencias, transformada) = TransformadaFourier(acel_filtrada, periodoMuestreo, plot = False)

## Cálculo de los coeficientes de Fourier en semieje positivo
coefs = (2 / acel_filtrada.shape[0]) * np.abs(transformada)[:acel_filtrada.shape[0]//2]

## Calculo de las frecuencias en el semieje positivo
frecs = frecuencias[:acel_filtrada.shape[0]//2]

## Elimino la componente de contínua de la señal
coefs[0] = 0

## Determino la posición en la que se da el máximo. 
## ESTOY ASUMIENDO QUE EL MÁXIMO SE DA EN LA COMPONENTE FUNDAMNENTAL (no tiene porque ocurrir!)
## Ésta será considerada como la frecuencia fundamental de la señal
pos_maximo = np.argmax(coefs)

## Frecuencia fundamental de la señal
## La frecuencia fundamental de la señal en las aceleraciones CC (craneo-cervical) y AP (antero-posterior) son iguales.
## Se podría interpretar como la frecuencia fundamental de los pasos en la marcha de la persona
frec_fund = frecs[pos_maximo]

## Estimación del tiempo de paso en base al análisis en frecuencia
## La estimación se hace en base al armónico de mayor amplitud de la señal
tiempo_paso_frecuencia = 1 / frec_fund

## Calculo la cantidad promedio de muestras que tengo en mi señal por cada paso
muestras_paso = tiempo_paso_frecuencia / periodoMuestreo

## ------------------------------------ DETECCIÓN CORTES EN CERO ---------------------------------------

## Obtengo los índices booleanos en donde se producen los cruces en cero de la aceleración vertical
## Los elementos <<True>> son aquellos valores luego de que se produce el corte en 0
ceros = librosa.zero_crossings(acel[:,1] - constants.g)

## Hago la traducción de elementos booleanos a índices numéricos
indices_ceros = np.where(ceros == True)

## Filtrado pasabandas para quedarme solo con las componentes de aceleración que interesan
filtrada = signal.sosfiltfilt(signal.butter(N = 4, Wn = [0.5, 4], btype = 'bandpass', fs = 1 / periodoMuestreo, output = 'sos'), acel[:,1] - constants.g)

## Obtengo los índices booleanos en donde se producen los cruces en cero de la aceleración vertical
## Los elementos <<True>> son aquellos valores luego de que se produce el corte en 0
ceros_sin_interpolar = librosa.zero_crossings(filtrada)

## Hago la traducción de elementos booleanos a índices numéricos
## Me aseguro también de eliminar el primer elemento donde se detecta un cambio de signo. O sea el instante inicial donde no tiene sentido contabilizar un HS o un TO
indices_ceros = np.where(ceros_sin_interpolar == True)[0][1:]

## Los toe offs serán aquellos valores donde se cambia de negativo a positivo
toe_offs = np.take(indices_ceros, np.where(filtrada[indices_ceros] < 0))[0]

## Creo una lista donde guardo los ceros toe off luego de interpolar
ceros_to = []

## Itero para cada uno de los ceros TO que detecté
for i in range (len(toe_offs)):

    ## Hago una interpolación lineal entre ese punto y el anterior para obtener el verdadero cruce en 0 con el TO
    cero_to = np.interp(x = 0, xp = [filtrada[toe_offs[i]], filtrada[toe_offs[i] - 1]], fp = [toe_offs[i], toe_offs[i] - 1])

    ## Agrego el cero TO calculado a la lista
    ceros_to.append(cero_to)