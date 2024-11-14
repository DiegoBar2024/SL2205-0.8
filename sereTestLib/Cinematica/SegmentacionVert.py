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

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Identificación del paciente
numero_paciente = '255'

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
## acc_analytic[:,2] --> Es la aceleración vertical
acc_analytic = imu_analytic.accCorr

## ------------------------------------------ PREPROCESADO ---------------------------------------------

## Se hace la normalización en amplitud y offset de la señal de aceleración vertical
acc_AP_norm = Normalizacion(acel[:,1] - constants.g)

## Hago el inverso también para detectar los toe offs
acc_AP_norm_TO = Normalizacion(- acel[:,1] + constants.g)

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

## -------------------------------------- DETECCIÓN PRIMER PICO ----------------------------------------

## Especifico un umbral predefinido para la detección de picos
umbral = 0.2

## Hago la detección de picos para un umbral predefinido
picos = DeteccionPicos(acc_AP_norm, umbral = umbral)

## Hago la detección de picos en el opuesto para calcular los TO
picosTO = DeteccionPicos(acc_AP_norm_TO, umbral = umbral)

## Se hace la graficación de la señal marcando los picos
GraficacionPicos(acc_AP_norm, picos)

## Me tomo un entorno de 0.3 * P hacia delante centrado en el primer pico detectado
## Ésto lo hago porque puede pasar en algún caso que el primer pico detectado no sea el correcto
rango = picos[0] + np.array([0, 0.3 * muestras_paso])

## Rango de posibles primeros picos
picos_rango = picos[picos < 0.3 * muestras_paso + picos[0]]

## Obtengo el índice del valor correspondiente al pico máximo
ind_pico_maximo = np.argmax(acc_AP_norm[picos_rango])

## Como están en el mismo orden puedo indexar el pico máximo en la lista de los picos detectados en el rango
pico_maximo_inicial = picos_rango[ind_pico_maximo]

## Mismo procedimiento para la señal opuesta

## Me tomo un entorno de 0.3 * P hacia delante centrado en el primer pico detectado
## Ésto lo hago porque puede pasar en algún caso que el primer pico detectado no sea el correcto
rangoTO = picosTO[0] + np.array([0, 0.3 * muestras_paso])

## Rango de posibles primeros picos
picos_rangoTO = picosTO[picosTO < 0.3 * muestras_paso + picosTO[0]]

## Obtengo el índice del valor correspondiente al pico máximo
ind_pico_maximoTO = np.argmax(acc_AP_norm_TO[picos_rangoTO])

## Como están en el mismo orden puedo indexar el pico máximo en la lista de los picos detectados en el rango
pico_maximo_inicialTO = picos_rangoTO[ind_pico_maximoTO]

## ------------------------------- GRAFICACIÓN DURACIÓN PASOS PRE MÉTODO -------------------------------

plt.scatter(x = np.arange(start = 0, stop = len(np.diff(picos))), y = np.diff(picos))
plt.show()

## ------------------------------------ DETECCIÓN PICOS SUCESIVOS --------------------------------------

## Creo una lista donde voy almacenando las posiciones de los picos sucesivos
## En ésta lista se va a almacenar el primer pico detectado por el algoritmo previo
picos_sucesivos = [pico_maximo_inicial]

## Creo la misma lista pero para la señal opuesta
picos_sucesivosTO = [pico_maximo_inicialTO]

## Creo una variable donde voy guardando el valor del indice de los picos sucesivos. Lo inicializo en 0
ind_picos_sucesivos = 0

## Defino la misma variable para la señal opuesta
ind_picos_sucesivosTO = 0

## Mientras que el rango no supere la longitud de la señal, que siga iterando
while (rango[0] < cant_muestras):

    ## Se calcula el rango de separación donde se espera que esté el próximo pico.
    ## Empíricamente se escoge [0.7 * P, 1.3 * P] donde P sería la cantidad de muestras por pico (paper de Zhao)
    rango = picos_sucesivos[ind_picos_sucesivos] + np.array([0.7 * muestras_paso, 1.3 * muestras_paso])

    ## Aumento en una unidad el valor del índice
    ind_picos_sucesivos += 1

    ## Mientras que no haya picos detectados en el rango, sigo iterando éste sub bucle
    while True:

        ## Hago la segmentación de la señal de aceleración AP en éste rango. Hago la conversión de los umbrales a enteros
        segment_rango = acc_AP_norm[int(rango[0]) : int(rango[1])]

        ## Hago la detección de picos en éste rango de señal de aceleración AP con el umbral preconfigurado
        ## Hago la detección de picos para un umbral predefinido
        picos_rango = DeteccionPicos(segment_rango, umbral = umbral)

        ## Debo ponerle dos condiciones para que el bucle se pare:
        # i) En caso de que haya picos detectados, rompo el bucle para interpolar el pico que fue detectado
        # ii) En caso de que el extremo izquierdo del rango sea mayor a la longitud de la señal
        if len(picos_rango) > 0  or rango[0] > cant_muestras:

                ## Ruptura de bucle
                break

        ## Seteo la referencia al pico previo sumado 0.7 * P en caso de que no haya ningún pico detectado en el rango actual
        rango = 0.7 * muestras_paso + rango

    ## En caso que el bucle haya salido porque se detectaron picos
    if len(picos_rango) > 0:

        ## Obtengo el valor se la señal de aceleración AP para éstos picos
        segment_picos = segment_rango[picos_rango]

        ## Obtengo el índice del valor correspondiente al pico máximo
        ind_pico_maximo = np.argmax(segment_picos)

        ## Como están en el mismo orden puedo indexar el pico máximo en la lista de los picos detectados en el rango
        pico_maximo = picos_rango[ind_pico_maximo]

        ## Agrego a la lista de picos sucesivos el pico detectado
        ## Recuerdo que debo sumarle al pico detectado el primer elemento del rango para llevarlo a la escala real
        picos_sucesivos.append(pico_maximo + int(rango[0]))
    
## Hago lo mismo para la señal opuesta
## Mientras que el rango no supere la longitud de la señal, que siga iterando
while (rangoTO[0] < cant_muestras):

    ## Se calcula el rango de separación donde se espera que esté el próximo pico.
    ## Empíricamente se escoge [0.7 * P, 1.3 * P] donde P sería la cantidad de muestras por pico (paper de Zhao)
    rangoTO = picos_sucesivosTO[ind_picos_sucesivosTO] + np.array([0.7 * muestras_paso, 1.3 * muestras_paso])

    ## Aumento en una unidad el valor del índice
    ind_picos_sucesivosTO += 1

    ## Mientras que no haya picos detectados en el rango, sigo iterando éste sub bucle
    while True:

        ## Hago la segmentación de la señal de aceleración AP en éste rango. Hago la conversión de los umbrales a enteros
        segment_rangoTO = acc_AP_norm_TO[int(rangoTO[0]) : int(rangoTO[1])]

        ## Hago la detección de picos en éste rango de señal de aceleración AP con el umbral preconfigurado
        ## Hago la detección de picos para un umbral predefinido
        picos_rangoTO = DeteccionPicos(segment_rangoTO, umbral = umbral)

        ## Debo ponerle dos condiciones para que el bucle se pare:
        # i) En caso de que haya picos detectados, rompo el bucle para interpolar el pico que fue detectado
        # ii) En caso de que el extremo izquierdo del rango sea mayor a la longitud de la señal
        if len(picos_rangoTO) > 0  or rangoTO[0] > cant_muestras:

                ## Ruptura de bucle
                break

        ## Seteo la referencia al pico previo sumado 0.7 * P en caso de que no haya ningún pico detectado en el rango actual
        rangoTO = 0.7 * muestras_paso + rangoTO
    
    ## En caso que el bucle haya salido porque se detectaron picos
    if len(picos_rangoTO) > 0:

        ## Obtengo el valor se la señal de aceleración AP para éstos picos
        segment_picosTO = segment_rangoTO[picos_rangoTO]

        ## Obtengo el índice del valor correspondiente al pico máximo
        ind_pico_maximoTO = np.argmax(segment_picosTO)

        ## Como están en el mismo orden puedo indexar el pico máximo en la lista de los picos detectados en el rango
        pico_maximoTO = picos_rangoTO[ind_pico_maximoTO]

        ## Agrego a la lista de picos sucesivos el pico detectado
        ## Recuerdo que debo sumarle al pico detectado el primer elemento del rango para llevarlo a la escala real
        picos_sucesivosTO.append(pico_maximoTO + int(rangoTO[0]))

## Hago la traducción de array a vector numpy
picos_sucesivos = np.array(picos_sucesivos)

## Hago lo mismo para la señal opuesta
picos_sucesivosTO = np.array(picos_sucesivosTO)

## Se hace la graficación de la señal marcando los picos
GraficacionPicos(acc_AP_norm, picos_sucesivos)
# GraficacionPicos(acel[:,1] - constants.g, picos_sucesivos)

## Grafico la señal con sus picos opuestos
GraficacionPicos(acc_AP_norm, picos_sucesivosTO)
# GraficacionPicos(acel[:,1] - constants.g, picos_sucesivosTO)

## ----------------------------------------- CONJUNTO DE PASOS -----------------------------------------

## Creo una lista donde voy a almacenar todos los pasos
pasos = []

## Itero para cada uno de los picos detectados
for i in range (len(picos_sucesivos) - 1):
    
    ## En caso que la distancia entre picos esté en un rango aceptable, concluyo que ahí se habrá detectado un paso
    if (0.7 * muestras_paso < picos_sucesivos[i + 1] - picos_sucesivos[i] < 1.3 * muestras_paso):

        ## Genero la variable donde guardo el Toe Off a incluír por defecto en 0
        toe_off = 0

        ## Busco el Toe Off que haya detectado para asociarlo al paso
        for picoTO in picos_sucesivosTO:
            
            ## En caso que el Toe Off esté entre los dos ICs
            if picos_sucesivos[i] < picoTO < picos_sucesivos[i + 1]:

                ## Me lo guardo
                toe_off = picoTO
        
        ## Entonces el par de picos me está diciendo que ahí hay un paso y entonces me lo guardo
        ## Me guardo también el Toe Off que haya detectado entre los dos pasos
        pasos.append({'IC': (picos_sucesivos[i], picos_sucesivos[i + 1]),'TC': toe_off})
    
## ----------------------------------------- DURACIÓN DE PASOS -----------------------------------------

## Creo una lista donde voy a almacenar las muestras entre todos los pasos
muestras_pasos = []

## Creo una lista en donde voy a almacenar las duraciones de todos los pasos
duraciones_pasos = []

## Itero para cada uno de los pasos detectados
for i in range (len(pasos)):
    
    ## Calculo la diferencia entre ambos valores de la tupla en términos temporales
    diff_pasos = pasos[i]['IC'][1] - pasos[i]['IC'][0]

    ## Almaceno la diferencia de muestras en la lista de muestras entre pasos
    muestras_pasos.append(diff_pasos)

    ## Almaceno la diferencia temporal entre los pasos en otra lista
    duraciones_pasos.append(diff_pasos * periodoMuestreo)

## ------------------------------- GRAFICACIÓN DURACIÓN PASOS POST MÉTODO ------------------------------

plt.scatter(x = np.arange(start = 0, stop = len(duraciones_pasos)), y = duraciones_pasos)
plt.show()

## --------------------------------------- TIEMPO ENTRE IC Y TC ----------------------------------------

## Creo una lista donde almaceno las distancias entre ICs y TCs expresado en muestras
dist_IC_TC = []

## Creo una lista donde almaceno las distancias entre ICs y TCs expresado en tiempo
dist_IC_TC_tiempo = []

## Itero para cada uno de los pasos detectados
for i in range (len(pasos)):
    
    ## Calculo la distancia entre IC y TC
    dist = pasos[i]['TC'] - pasos[i]['IC'][0]

    ## Agrego la distancia a la lista
    dist_IC_TC.append(dist)

## Genero la lista de tiempos
dist_IC_TC_tiempo = np.multiply(periodoMuestreo, dist_IC_TC)

## ---------------------------------- GRAFICACIÓN TIEMPO ENTRE IC Y TC ---------------------------------

# plt.scatter(x = np.arange(start = 0, stop = len(dist_IC_TC_tiempo)), y = dist_IC_TC_tiempo)
# plt.show()

## --------------------------------------- CALCULO DOBLE ESTANCIA --------------------------------------

## Genero una lista vacía donde voy a calcular las proporciones de doble estancia en un paso
doble_estancia = []

## Itero para cada uno de los pasos detectados
for i in range (len(pasos)):

    ## Calculo la proporcion de la doble estancia
    doble_estancia_paso = (pasos[i]['TC'] - pasos[i]['IC'][0]) / (pasos[i]['IC'][1] - pasos[i]['IC'][0])

    ## En caso que la doble estancia tenga un valor no permitido, que se saltee ésta parte
    if abs(doble_estancia_paso) > 1:

        ## Se saltea ésta iteración
        continue

    ## La agrego a la lista
    doble_estancia.append(doble_estancia_paso)

## ------------------------------- GRAFICACIÓN DOBLE ESTANCIA EN CADA PASO -----------------------------

# plt.scatter(x = np.arange(start = 0, stop = len(doble_estancia)), y = doble_estancia)
# plt.show()