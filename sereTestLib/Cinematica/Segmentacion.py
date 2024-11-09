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
from skinematics.imus import analytical, IMU_Base

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

## Cantidad de muestras de la señal
cant_muestras = len(tiempo)

## -------------------------------------- CORRECCIÓN DE EJES -------------------------------------------

## Armamos el diccionario con los datos a ingresar
dict_datos = {'acc': acel, 'omega' : gyro, 'rate' : 1 / periodoMuestreo}

## Matriz de rotación inicial
## Es importante tener en cuenta la orientación inicial de la persona
orient_inicial = np.array([np.array([1,0,0]), np.array([0,0,1]), np.array([0,1,0])])

## Creacion de instancia IMU_Base
imu_analytic = IMU_Base(in_data = dict_datos, q_type = 'analytical', R_init = orient_inicial, calculate_position = False)

## Accedo a los cuaterniones resultantes
q_analytic = imu_analytic.quat

## Accedo a la velocidad resultante
vel_analytic = imu_analytic.vel

## Accedo a la posición resultante
pos_analytic = imu_analytic.pos

## Accedo a la aceleración corregida
acc_analytic = imu_analytic.accCorr

## ------------------------------------------ PREPROCESADO ---------------------------------------------

## Se calcula la magnitud de la señal de aceleración
magnitud = Magnitud(acel)

## Se hace la normalización en amplitud y offset de la señal de magnitud
mag_normalizada = Normalizacion(magnitud)

## ------------------------------------- ANÁLISIS EN FRECUENCIA ----------------------------------------

## Filtro de Butterworth pasaaltos de orden 4, frecuencia de corte 0.1Hz
filtro = signal.butter(N = 4, Wn = 0.1, btype = 'highpass', fs = 1 / periodoMuestreo, output = 'sos')

## Aplico el Filtro anterior
magnitud_filtrada = signal.sosfilt(filtro, magnitud)

## Análisis de armónicos a la señal de magnitud luego de aplicar el filtrado pasaaltos
armonicos = harm_analysis(magnitud_filtrada, FS = 1 / periodoMuestreo)

## Obtengo frecuencia fundamental de la señal
frec_fund = armonicos['fund_freq']

## Estimación del tiempo de paso en base al análisis en frecuencia
## La estimación se hace en base al armónico de mayor amplitud de la señal
tiempo_paso_frecuencia = 1 / frec_fund

## Calculo la cantidad promedio de muestras que tengo en mi señal por cada paso
muestras_paso = tiempo_paso_frecuencia / periodoMuestreo

print("Muestras por paso: ", muestras_paso)
print("Cadencia: ", frec_fund)

## -------------------------------------- DETECCIÓN PRIMER PICO ----------------------------------------

## Especifico un umbral predefinido para la detección de picos
umbral = 0.2

## Hago la detección de picos para un umbral de T = 0.2
picos = DeteccionPicos(mag_normalizada, umbral = umbral)

## A partir de lo anterior genero un vector que me diga las posiciones temporales de los picos
picos_tiempo = picos * periodoMuestreo

## Se hace la graficación de la señal marcando los picos
GraficacionPicos(mag_normalizada, picos, tiempo, periodoMuestreo)

## Me tomo un entorno de 0.3 * P hacia delante centrado en el primer pico detectado
## Ésto lo hago porque puede pasar en algún caso que el primer pico detectado no sea el correcto
rango = picos[0] + np.array([0, 0.3 * muestras_paso])

## Rango de posibles primeros picos
picos_rango = picos[picos < 0.3 * muestras_paso + picos[0]]

## Obtengo el índice del valor correspondiente al pico máximo
ind_pico_maximo = np.argmax(mag_normalizada[picos_rango])

## Como están en el mismo orden puedo indexar el pico máximo en la lista de los picos detectados en el rango
pico_maximo_inicial = picos_rango[ind_pico_maximo]

## ------------------------------- GRAFICACIÓN DURACIÓN PASOS PRE MÉTODO -------------------------------

plt.scatter(x = np.arange(start = 0, stop = len(np.diff(picos))), y = np.diff(picos))
plt.show()

## ------------------------------------ DETECCIÓN PICOS SUCESIVOS --------------------------------------

## Creo una lista donde voy almacenando las posiciones de los picos sucesivos
## En ésta lista se va a almacenar el primer pico detectado por el algoritmo previo
picos_sucesivos = [pico_maximo_inicial]

## Creo una variable donde voy guardando el valor del indice de los picos sucesivos. Lo inicializo en 0
ind_picos_sucesivos = 0

## Mientras que el rango no supere la longitud de la señal, que siga iterando
while (picos_sucesivos[-1] + muestras_paso < cant_muestras):

    ## Se calcula el rango de separación donde se espera que esté el próximo pico.
    ## Empíricamente se escoge [0.7 * P, 1.3 * P] donde P sería la cantidad de muestras por pico (paper de Zhao)
    rango = picos_sucesivos[ind_picos_sucesivos] + np.array([0.7 * muestras_paso, 1.3 * muestras_paso])

    ## Aumento en una unidad el valor del índice
    ind_picos_sucesivos += 1

    ## Mientras que no haya picos detectados en el rango, sigo iterando éste sub bucle
    while True:

        ## Hago la segmentación de la señal de magnitud normalizada en éste rango. Hago la conversión de los umbrales a enteros
        segment_rango = mag_normalizada[int(rango[0]) : int(rango[1])]

        ## Hago la detección de picos en éste rango de señal de magnitud normalizada con el umbral preconfigurado
        ## Hago la detección de picos para un umbral de T = 0.2
        picos_rango = DeteccionPicos(segment_rango, umbral = umbral)

        ## En caso de que haya picos detectados, rompo el bucle para interpolar el pico que fue detectado
        if len(picos_rango) > 0:

                ## Ruptura de bucle
                break

        ## Seteo la referencia al pico previo sumado 0.7 * P en caso de que no haya ningún pico detectado en el rango actual
        rango = 0.7 * muestras_paso + rango

    ## Obtengo el valor se la señal de magnitud normalizada para éstos picos
    segment_picos = segment_rango[picos_rango]

    ## Obtengo el índice del valor correspondiente al pico máximo
    ind_pico_maximo = np.argmax(segment_picos)

    ## Como están en el mismo orden puedo indexar el pico máximo en la lista de los picos detectados en el rango
    pico_maximo = picos_rango[ind_pico_maximo]

    ## Agrego a la lista de picos sucesivos el pico detectado
    ## Recuerdo que debo sumarle al pico detectado el primer elemento del rango para llevarlo a la escala real
    picos_sucesivos.append(pico_maximo + int(rango[0]))

## Hago la traducción de array a vector numpy
picos_sucesivos = np.array(picos_sucesivos)

## Se hace la graficación de la señal marcando los picos
GraficacionPicos(mag_normalizada, picos_sucesivos, tiempo, periodoMuestreo)

## ----------------------------------------- CONJUNTO DE PASOS -----------------------------------------

## Creo una lista donde voy a almacenar todos los pasos
pasos = []

## Itero para cada uno de los picos detectados
for i in range (len(picos_sucesivos) - 1):
    
    ## En caso que la distancia entre picos esté en un rango aceptable, concluyo que ahí se habrá detectado un paso
    if (0.7 * muestras_paso < picos_sucesivos[i + 1] - picos_sucesivos[i] < 1.3 * muestras_paso):
        
        ## Entonces el par de picos me está diciendo que ahí hay un paso y entonces me lo guardo
        pasos.append((picos_sucesivos[i], picos_sucesivos[i + 1]))
    
## ----------------------------------------- DURACIÓN DE PASOS -----------------------------------------

## Creo una lista donde voy a almacenar las muestras entre todos los pasos
muestras_pasos = []

## Creo una lista en donde voy a almacenar las duraciones de todos los pasos
duraciones_pasos = []

## Itero para cada uno de los pasos detectados
for i in range (len(pasos)):
    
    ## Calculo la diferencia entre ambos valores de la tupla en términos temporales
    diff_pasos = pasos[i][1] - pasos[i][0]

    ## Almaceno la diferencia de muestras en la lista de muestras entre pasos
    muestras_pasos.append(diff_pasos)

    ## Almaceno la diferencia temporal entre los pasos en otra lista
    duraciones_pasos.append(diff_pasos * periodoMuestreo)

## ------------------------------- GRAFICACIÓN DURACIÓN PASOS POST MÉTODO ------------------------------

plt.scatter(x = np.arange(start = 0, stop = len(duraciones_pasos)), y = duraciones_pasos)
plt.show()