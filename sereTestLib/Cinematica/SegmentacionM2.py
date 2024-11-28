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

## Los toe offs serán aquellos valores donde se cambia de negativo a positivo
toe_offs = np.take(indices_ceros, np.where(filtrada[indices_ceros] < 0))[0]

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

## Itero para cada uno de los ceros TO que detecté
for i in range (len(toe_offs)):

    ## Hago una interpolación lineal entre ese punto y el anterior para obtener el verdadero cruce en 0 con el TO
    cero_to = np.interp(x = 0, xp = [filtrada[toe_offs[i]], filtrada[toe_offs[i] - 1]], fp = [toe_offs[i], toe_offs[i] - 1])

    ## Agrego el cero TO calculado a la lista
    ceros_to.append(cero_to)

## Graficación de Heel Strikes y Toe Offs tomando la aceleración vertical filtrada
plt.plot(ceros_hs, np.zeros(len(ceros_hs)), "x", label = 'Heel Strikes')
plt.plot(ceros_to, np.zeros(len(ceros_to)), "o", label = 'Toe Offs')
plt.plot(filtrada, label = "Aceleracion Vertical Filtrada")
plt.legend()
plt.show()

## Graficación de Heel Strikes y Toe Offs tomando la aceleración anteroposterior
plt.plot(ceros_hs, np.zeros(len(ceros_hs)), "x", label = 'Heel Strikes')
plt.plot(ceros_to, np.zeros(len(ceros_to)), "o", label = 'Toe Offs')
plt.plot(-acel[:,2], label = "Aceleracion Anteroposterior")
plt.legend()
plt.show()

print("Tiempo Paso Promedio: {}".format(np.mean(np.diff(ceros_hs)) * periodoMuestreo))
print("Tiempo Paso Desviación Estándar: {}".format(np.std(np.diff(ceros_hs)) * periodoMuestreo))
print("Tiempo Paso Mediana: {}".format(np.median(np.diff(ceros_hs)) * periodoMuestreo))

## ------------------------------------ DETECCIÓN PRIMER EVENTO ----------------------------------------

## En caso que el primer cero del HS se encuentre antes del primer cero del TO
if ceros_hs[0] < ceros_to[0]:

    ## Entonces el primer evento va a ser un Heel Strike
    primer_evento = 'hs'

## En caso que el primer cero del TO se encuentre antes del primer cero del HS
else:

    ## Entonces el primer evento va a ser un Toe Off
    primer_evento = 'to'

## ------------------------------- PROPORCIÓN ESTANCIA DOBLE Y SIMPLE ----------------------------------

## Creo una lista donde voy a guardar las proporciones de estancia doble y simple
proporciones_paso = []

## Itero para cada uno de los eventos que tengo detectados
for i in range (len(ceros_hs) - 1):

    ## En caso que el primer evento sea un HS
    if primer_evento == 'hs':

        ## Calculo la proporción del paso como DoubleStance/SingleStance = (TO[i]-HS[i])/(HS[i+1]-TO[i])
        proporcion = (ceros_to[i] - ceros_hs[i]) / (ceros_hs[i + 1] - ceros_to[i])

        ## Agrego la proporción calculada a la lista de proporciones de paso
        proporciones_paso.append(proporcion)
    
    ## En caso que el primer evento sea un TO
    else:

        ## Calculo la proporción del paso como DoubleStance/SingleStance = (TO[i+1]-HS[i])/(HS[i]-TO[i])
        proporcion = (ceros_to[i + 1] - ceros_hs[i]) / (ceros_hs[i] - ceros_to[i])

        ## Agrego la proporción calculada a la lista de proporciones de paso
        proporciones_paso.append(proporcion)

## ----------------------------------------- CONJUNTO DE PASOS -----------------------------------------

## Creo una lista donde voy a guardar los pasos detectados por Zero Crossing
pasos_zc = []

## Itero para cada uno de los eventos HS que tengo detectados
for i in range (len(ceros_hs) - 1):

    ## En caso de que el primer evento haya sido un Heel Strike
    if primer_evento == 'hs':

        ## Defino el paso como un diccionario donde tengo [HS[i], HS[i+1]] como los ICs y [TO[i]] como el TC
        ## Tengo que agregar los valores de los HS y TO para luego poder hacer la segmentación de la señal de posición al medir el paso
        paso_zc = {'IC': (heel_strikes[i], heel_strikes[i + 1]),'TC': toe_offs[i]}
    
    ## En caso que el primer evento haya sido un Toe Off
    else:

        ## Defino el paso como un diccionario donde tengo [HS[i], HS[i+1]] como los ICs y [TO[i+1]] como el TC
        ## Tengo que agregar los valores de los HS y TO para luego poder hacer la segmentación de la señal de posición al medir el paso
        paso_zc = {'IC': (heel_strikes[i], heel_strikes[i + 1]),'TC': toe_offs[i + 1]}
    
    ## Agrego el paso detectado a la lista de pasos
    pasos_zc.append(paso_zc)