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

from LecturaDatos import *

## ---------------------------------------- LECTURA DE DATOS -------------------------------------------

## Especifico el identificador del paciente para el cual voy a realizar la lectura de los datos
id_persona = 299

## Obtengo los datos leídos en el directorio correspondiente
data, acel, gyro, cant_muestras, periodoMuestreo, nombre_persona, nacimiento_persona = LecturaDatos(id_persona)

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

## Los heel strikes serán aquellos valores donde se cambia de positivo a negativo
heel_strikes = np.take(indices_ceros, np.where(filtrada[indices_ceros] > 0))[0]

## Creo una lista donde guardo los ceros heel strike luego de interpolar
ceros_hs = []

## Creo una lista donde guardo los ceros toe off luego de interpolar
ceros_to = []

## Itero para cada uno de los ceros HS que detecté
for i in range (len(heel_strikes)):

    ## Hago una interpolación lineal entre ese punto y el anterior para obtener el verdadero cruce en 0 con el HS
    cero_hs = np.interp(x = 0, xp = [filtrada[heel_strikes[i] - 1], filtrada[heel_strikes[i]]], fp = [heel_strikes[i] - 1, heel_strikes[i]])

    ## Agrego el cero HS calculado a la lista
    ceros_hs.append(cero_hs)