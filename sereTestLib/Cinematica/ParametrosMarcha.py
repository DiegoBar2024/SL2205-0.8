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

## Ruta del archivo
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

## -------------------------------------- CORRECCIÓN DE EJES ----------------------------------

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

## ------------------------------------------- PREPROCESADO ------------------------------------------

## Se calcula la magnitud de la señal de aceleración
magnitud = Magnitud(acc_analytic)

## Se hace la normalización en amplitud y offset de la señal de magnitud
mag_normalizada = Normalizacion(magnitud)

## Se realiza un filtrado de medianas para eliminar algunos picos no relevantes de la señal
normal_filtrada = signal.medfilt(mag_normalizada, kernel_size = 11)

## -------------------------------------- PROCESADO ACELERACION AP --------------------------------------

# ## Aceleración en la dirección anteroposterior
# acel_x = acc_analytic[:,0]

# ## Filtrado Pasabajos Butterworth de Orden 4 con fc = 2Hz (paper de Zijlstra)
# sos_acel_x = signal.butter(N = 4, Wn = 2, btype = 'lowpass', fs = 1 / periodoMuestreo, output = 'sos')

# ## Aplico el Filtro anterior
# acel_x_filtrada = signal.sosfilt(sos_acel_x, acel_x)

# ## Graficación
# plt.plot(tiempo, acel_x, label = 'Sin filtrar')
# plt.plot(tiempo, acel_x_filtrada, label = 'Filtrada')
# plt.legend()
# plt.show()

## --------------------------------------- ANÁLISIS EN FRECUENCIA ----------------------------------------

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

print(tiempo_paso_frecuencia)

## ----------------------------------------- DETECCIÓN DE PICOS ------------------------------------------

## Cálculo de umbral óptimo
(T, stdT) = CalculoUmbral(señal = mag_normalizada, Tmin = 0.3, Tmax = 0.5)

## Se hace el llamado a la función de detección de picos configurando un umbral de entrada T
picos = DeteccionPicos(mag_normalizada, umbral = T)

## A partir de lo anterior genero un vector que me diga las posiciones temporales de los picos
picos_tiempo = picos * periodoMuestreo

## Se hace la graficación de la señal marcando los picos
GraficacionPicos(mag_normalizada, picos, tiempo, periodoMuestreo)

## Obtengo el vector con las separaciones de los picos
separaciones = SeparacionesPicos(picos)

## Hago la traduccion de muestras a valores de tiempo usando la frecuencia de muestreo dada
sep_tiempos = separaciones * periodoMuestreo

## Valor medio de la separación de tiempos
tiempo_medio = np.mean(sep_tiempos)

## Desviación estándar de la separación de tiempos
desv_tiempos = np.std(sep_tiempos)

## Graficación como valores de dispersión de las señales de separación de tiempos
plt.axhline(y = tiempo_medio + desv_tiempos, linestyle = '-', color = 'r')
plt.axhline(y = tiempo_medio, linestyle = '-', color = 'b')
plt.axhline(y = tiempo_medio - desv_tiempos, linestyle = '-', color = 'g')
plt.scatter(x = np.arange(start = 0, stop = len(sep_tiempos)), y = sep_tiempos)
plt.show()

condicion_filtrado_1 = sep_tiempos > tiempo_paso_frecuencia * 0.8

## Se hace el filtrado de aquellos tiempos que estén en el intervalo definido antes
filtrado_tiempos_1 = sep_tiempos[condicion_filtrado_1]

## Se explicita la condición de filtrado de que me quedo con aquellos tiempos que tengan una separación mayor a la media - 0.5 * desviacion estándar
condicion_filtrado_2 = filtrado_tiempos_1 < tiempo_paso_frecuencia * 1.2

## Se hace el filtrado de aquellos tiempos que estén en el intervalo definido antes
filtrado_tiempos = filtrado_tiempos_1[condicion_filtrado_2]

## Cantidad de pasos que me quedan
cant_pasos = len(filtrado_tiempos)

## Se calcula el valor medio luego de realizar el filtrado
media_filtrado = np.mean(filtrado_tiempos)

## Se calcula la desviación estándar luego de realizar el filtrado
desv_filtrado = np.std(filtrado_tiempos)

## Obtengo la proporción de separaciones que se encuentran dentro del rango (media - desv_estandar, media + desv_estandar)
## Cuanto mayor sea éste índice mejor ya que va a ser más constante el tiempo de separación de los pasos y es más consistente
## Se puede poner como valor aceptable para interpretar ésto un valor de índice mínimo de 0.75
muestras_interior = filtrado_tiempos[media_filtrado - desv_filtrado < filtrado_tiempos]

## Hago la segunda parte del filtrado para evitar que no se contabilicen datos repetidos
muestras_interior = muestras_interior[media_filtrado + desv_filtrado > muestras_interior]

## Calculo la proporción de muestras en el interior del rango comparado con el exterior
proporcion_int_ext = len(muestras_interior) / len(filtrado_tiempos)

## Calculo la segunda diferencia
sep_tiempos_seg = np.diff(sep_tiempos)

## Graficación de la dispersión de los puntos de la señal filtrada
plt.axhline(y = media_filtrado + desv_filtrado, linestyle = '-', color = 'r')
plt.axhline(y = media_filtrado, linestyle = '-', color = 'b')
plt.axhline(y = media_filtrado - desv_filtrado, linestyle = '-', color = 'g')
plt.scatter(x = np.arange(start = 0, stop = len(filtrado_tiempos)), y = filtrado_tiempos)
plt.show()

## Estimación del tiempo de paso en base a la detección de picos
tiempo_paso_picos = media_filtrado

print(cant_pasos)
print(media_filtrado)
print(desv_filtrado)

## ----------------------------- DETECCIÓN DE PICOS DEFECTUOSOS (MUY SEPARADOS) ---------------------------------

## Creo una lista donde guardo las separaciones mayores a lo razonable
sep_defect_ind = []

## Itero para cada una de las separaciones de picos que tengo
for i in range (len(sep_tiempos)):

    ## Recuerdo que la sep(i) = pico(i + 1) - pico(i) como el i-ésimo valor de la separación
    if (sep_tiempos[i] > 1.2 * tiempo_paso_frecuencia):
    
        ## La agrego a la lista de defectuosas
        sep_defect_ind.append(i)

## --------------------------- SEGMENTACIÓN DE PICOS DEFECTUOSOS (MUY SEPARADOS) --------------------------------

## Creo una lista vacía en donde voy a guardar los defectuosos segmentados
defectuosos = []

## Itero para cada uno de los segmentos defectuosos detectados
for i in sep_defect_ind:

    ## Hago la segmentación de la señal en los segmentos defectuosos
    segmento = mag_normalizada[picos[i] - 2 : picos[i + 1] + 2]

    ## Luego lo agrego a la señal segmentada
    defectuosos.append(segmento)

## ----------------------------- DETECCIÓN DE TRAMOS DEFECTUOSOS (MUY SEPARADOS) --------------------------------

## Inicializo un valor que me permita iterar en todos los índices del vector de defectuosos
indice_defect = 0

## Itero para cada uno de los índices que tengo
for i in sep_defect_ind:

    ## Cálculo de umbral óptimo
    (T, stdT) = CalculoUmbral(señal = defectuosos[indice_defect], Tmin = 0, Tmax = 0.3)

    ## Se hace el llamado a la función de detección de picos configurando un umbral de entrada T
    picos_def = DeteccionPicos(defectuosos[indice_defect], umbral = T)

    ## Actualizo el valor del índice
    indice_defect += 1

    ## GraficacionPicos(señal = defectuosos[indice_defect], picos = picos_def, dt = 1, tiempo = np.array([]))

    ## Agrego los picos interpolados a la lista de picos de la señal completa
    picos = np.concatenate((picos, np.add(picos_def, picos[i] - 2)), axis = None)

## Reordeno el vector con los picos interpolados. Elimino también picos duplicados
picos = np.unique(np.sort(picos))

## Se hace la graficación de la señal marcando los picos con picos interpolados
GraficacionPicos(mag_normalizada, picos, tiempo, periodoMuestreo)

## ----------------------------- DATOS DE PASOS LUEGO DE INTERPOLAR PICOS (MUY SEPARADOS) ------------------------

## Obtengo el vector con las separaciones de los picos
separaciones = SeparacionesPicos(picos)

## Hago la traduccion de muestras a valores de tiempo usando la frecuencia de muestreo dada
sep_tiempos = separaciones * periodoMuestreo

## Valor medio de la separación de tiempos
tiempo_medio = np.mean(sep_tiempos)

## Desviación estándar de la separación de tiempos
desv_tiempos = np.std(sep_tiempos)

## Graficación como valores de dispersión de las señales de separación de tiempos
plt.axhline(y = tiempo_medio + desv_tiempos, linestyle = '-', color = 'r')
plt.axhline(y = tiempo_medio, linestyle = '-', color = 'b')
plt.axhline(y = tiempo_medio - desv_tiempos, linestyle = '-', color = 'g')
plt.scatter(x = np.arange(start = 0, stop = len(sep_tiempos)), y = sep_tiempos)
plt.show()

print(tiempo_medio)
print(desv_tiempos)

## ----------------------------- DETECCIÓN DE PICOS DEFECTUOSOS (MUY JUNTOS) ----------------------------------

## Creo una lista donde guardo las separaciones menores a lo razonable
sep_defect_ind = []

## Itero para cada una de las separaciones de picos que tengo
for i in range (len(sep_tiempos)):

    ## Recuerdo que la sep(i) = pico(i + 1) - pico(i) como el i-ésimo valor de la separación
    if (sep_tiempos[i] < tiempo_paso_frecuencia * 0.8):
    
        ## La agrego a la lista de defectuosas
        sep_defect_ind.append(i)

## ----------------------------- DETECCIÓN DE PICOS MÁXIMOS EN CADA TRAMO ---------------------------------

## Inicializo una lista que va a contener los índices de los picos máximos que voy detectando en cada tramo
picos_maximos = []

## Itero para cada uno de los índices que tengo
## Recuerdo que la sep(i) = pico(i + 1) - pico(i) como el i-ésimo valor de la separación
for i in sep_defect_ind:

    ## Hago la segmentación de la señal en ese tramo
    segmento = mag_normalizada[picos[i] - 2 : picos[i + 1] + 2]

    ## Obtengo cual es el índice del pico máximo en cada tramo
    pico_máximo = np.argmax(segmento) + picos[i] - 2

    ## Agrego el pico máximo detectado a la lista de picos máximos
    picos_maximos.append(pico_máximo)

## Elimino los picos máximos que me quedan duplicados de la lista
picos_maximos = np.unique(picos_maximos)

## ----------------------------- DETECCIÓN DE PICOS MÍNIMOS EN CADA TRAMO ---------------------------------

## Creo una lista donde voy a guardar los picos mínimos de cada tramo
picos_minimos = []

## Itero para cada uno de los índices que tengo
## Recuerdo que la sep(i) = pico(i + 1) - pico(i) como el i-ésimo valor de la separación
for i in sep_defect_ind:

    ## Creo la lista conteniendo los índices de los picos que hay en ese tramo
    picos_tramo = np.array([picos[i], picos[i + 1]])

    ## En caso de que la diferencia simétrica sea no nula
    if len(np.setdiff1d(picos_tramo, picos_maximos)) > 0:
        
        ## Hago la diferencia simétrica entre la dupla de los dos picos en el tramo y el conjunto de picos máximos
        ## Hago la indexación en 0 para obtener el valor del pico mínimo.
        pico_minimo = np.setdiff1d(picos_tramo, picos_maximos)[0]

        ## Agrego el pico mínimo a la lista de picos mínimos
        picos_minimos.append(pico_minimo)

## Elimino duplicados y ordeno la lista de picos mínimos
picos_minimos = np.sort(np.unique(np.array(picos_minimos)))

## ----------------------------------ELIMINACIÓN DE PICOS MÍNIMOS ------------------------------------------

## Elimino los picos mínimos de la lista de picos
picos = np.setdiff1d(picos, picos_minimos)

# Se hace la graficación de la señal luego de procesar los picos
GraficacionPicos(mag_normalizada, picos, tiempo, periodoMuestreo)

## ----------------------------- DATOS DE PASOS LUEGO DE PROCESAR TODOS LOS PICOS --------------------------

## Obtengo el vector con las separaciones de los picos
separaciones = SeparacionesPicos(picos)

## Hago la traduccion de muestras a valores de tiempo usando la frecuencia de muestreo dada
sep_tiempos = separaciones * periodoMuestreo

## Valor medio de la separación de tiempos
tiempo_medio = np.mean(sep_tiempos)

## Desviación estándar de la separación de tiempos
desv_tiempos = np.std(sep_tiempos)

## Graficación como valores de dispersión de las señales de separación de tiempos
plt.axhline(y = tiempo_medio + desv_tiempos, linestyle = '-', color = 'r')
plt.axhline(y = tiempo_medio, linestyle = '-', color = 'b')
plt.axhline(y = tiempo_medio - desv_tiempos, linestyle = '-', color = 'g')
plt.scatter(x = np.arange(start = 0, stop = len(sep_tiempos)), y = sep_tiempos)
plt.show()

print(tiempo_medio)
print(desv_tiempos)

## ----------------------------- ELIMINADO DE VALORES QUE SE VAN DE LA TOLERANCIA --------------------------

condicion_filtrado_1 = sep_tiempos > tiempo_paso_frecuencia * 0.8

## Se hace el filtrado de aquellos tiempos que estén en el intervalo definido antes
filtrado_tiempos_1 = sep_tiempos[condicion_filtrado_1]

condicion_filtrado_2 = filtrado_tiempos_1 < tiempo_paso_frecuencia * 1.2

## Se hace el filtrado de aquellos tiempos que estén en el intervalo definido antes
filtrado_tiempos = filtrado_tiempos_1[condicion_filtrado_2]

## Cantidad de pasos que me quedan
cant_pasos = len(filtrado_tiempos)

## Se calcula el valor medio luego de realizar el filtrado
media_filtrado = np.mean(filtrado_tiempos)

## Se calcula la desviación estándar luego de realizar el filtrado
desv_filtrado = np.std(filtrado_tiempos)

## Graficación de la dispersión de los puntos de la señal filtrada
plt.axhline(y = media_filtrado + desv_filtrado, linestyle = '-', color = 'r')
plt.axhline(y = media_filtrado, linestyle = '-', color = 'b')
plt.axhline(y = media_filtrado - desv_filtrado, linestyle = '-', color = 'g')
plt.scatter(x = np.arange(start = 0, stop = len(filtrado_tiempos)), y = filtrado_tiempos)
plt.show()

print(media_filtrado)
print(desv_filtrado)
print(cant_pasos)

## ----------------------------------------- SEGMENTACIÓN --------------------------------------------------

## Creo una lista vacía en donde voy a guardar los pasos segmentados
segmentada = []

## Itero para cada uno de los pasos que tengo detectados
for i in range (len(picos) - 1):

    ## Hago la segmentación de la señal
    segmento = magnitud[picos[i] : picos[i + 1]]

    ## Luego lo agrego a la señal segmentada
    segmentada.append(segmento)

## ----------------------------------------- GRÁFICAS ------------------------------------------

# plt.plot(acc_analytic[:,0], color = 'r', label = '$a_x$')
# plt.plot(acc_analytic[:,1], color = 'b', label = '$a_y$')
# plt.plot(acc_analytic[:,2], color = 'g', label = '$a_z$')

# ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
# plt.xlabel("Tiempo (s)")
# plt.ylabel("Aceleracion $(m/s^2)$")

# ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
# plt.legend()

# ## Despliego la gráfica
# plt.show()

# plt.plot(defect_normalizados[0])
# plt.show()