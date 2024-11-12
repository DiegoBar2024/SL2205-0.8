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
from Segmentacion import picos_sucesivos, frec_fund, pasos, doble_estancia
import emd

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Ruta del archivo
ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S222/3S222.csv"

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

## ------------------------------------- INTEGRACIÓN ACELERACIÓN ---------------------------------------

## Defino la señal de aceleración en dirección vertical
acc_vert = acel[:,1] - constants.g

## Integro aceleración para obtener velocidad
vel_vert = cumulative_trapezoid(acc_vert - np.mean(acc_vert), dx = periodoMuestreo, initial = 0)

## ----------------------------------- DESCOMPOSICIÓN EMD VELOCIDAD ------------------------------------

## Hago la descomposición EMD de la señal de velocidad
imfs_vel_vert = emd.sift.sift(vel_vert)

## Me quedo con los primeros 4 IMFs de alta frecuencia
vel_vert_procesada = imfs_vel_vert[:,0] + imfs_vel_vert[:,1] + imfs_vel_vert[:,2] + imfs_vel_vert[:,3]

## -------------------------------------- INTEGRACIÓN VELOCIDAD ----------------------------------------

## Integro velocidad para obtener posición
pos_vert = cumulative_trapezoid(vel_vert_procesada, dx = periodoMuestreo, initial = 0)

## ----------------------------------- DESCOMPOSICIÓN EMD POSICIÓN -------------------------------------

## Hago la descomposición EMD de la señal de posición
imfs_pos_vert = emd.sift.sift(pos_vert)

## Me quedo con los primeros tres IMFs de alta frecuencia
pos_vert_procesada = imfs_pos_vert[:,0] + imfs_pos_vert[:,1] + imfs_pos_vert[:,2]

## -------------------------------------- SEGMENTACIÓN DE PASOS ----------------------------------------

## Creo una lista vacía en donde voy a guardar los pasos segmentados
segmentada = []

## Itero para cada uno de los pasos que tengo detectados
for i in range (len(pasos) - 1):

    ## Hago la segmentación de la señal
    segmento = pos_vert_procesada[pasos[i]['IC'][0] : pasos[i]['IC'][1]]

    ## Luego lo agrego a la señal segmentada
    segmentada.append(segmento)

## -------------------------------- VARIACIÓN DE ALTURA CENTRO DE MASA ---------------------------------

## Creo una lista donde voy a almacenar las variaciones de altura del centro de masa en cada tramo
desp_vert_COM = []

## Itero para cada uno de los segmentos de pasos detectados
for i in range (len(segmentada)):

    ## Calculo la variación de altura del centro de masa en base a la diferencia entre el desplazamiento vertical máximo y mínimo
    d_step = abs(max(segmentada[i]) - min(segmentada[i]))

    ## Agrego el desplazamiento máximo del COM calculado a la lista
    desp_vert_COM.append(d_step)

## ---------------------------- GRAFICACIÓN VARIACIÓN ALTURA CENTRO DE MASA ----------------------------

# plt.scatter(x = np.arange(start = 0, stop = len(desp_vert_COM)), y = desp_vert_COM)
# plt.show()

## ----------------------------- CÁLCULO DE LA LONGITUD DEL PASO (MÉTODO I) ----------------------------

## Especifico la longitud de la pierna del individuo en metros
## Ésto debe considerarse como una entrada al sistema. Es un parámetro que puede medirse
## ¡IMPORTANTE: ÉSTE PARÁMETRO CAMBIA CON CADA PERSONA! SINO EL RESULTADO DA CUALQUIER COSA
long_pierna = 0.9

## Creo una lista donde voy a almacenar la longitud de los pasos de la persona
long_pasos_m1 = []

## Itero para cada uno de los segmentos de pasos detectados (IC a IC)
for i in range (len(segmentada)):
    
    ## Calculo la longitud del paso con la fórmula sugerida por Zijlstra
    ## Aplico Factor de Corrección multiplicativo de 1.25 (recomendación para no subestimar longitud pasos)
    long_paso = 1.25 * 2 * np.sqrt(2 * long_pierna * desp_vert_COM[i] - desp_vert_COM[i] ** 2)

    ## Agrego el paso a la lista de longitud de pasos
    long_pasos_m1.append(long_paso)

## ----------------------------- CÁLCULO DE PARÁMETROS DE MARCHA (MÉTODO I) ----------------------------

## Se calcula la longitud de paso promedio
long_paso_promedio = np.mean(long_pasos_m1)

## Se calcula la duración de paso promedio como el inverso de la cadencia
tiempo_paso_promedio = 1 / frec_fund

## Se calcula la velocidad de marcha como el cociente entre éstas cantidades
velocidad_marcha = long_paso_promedio / tiempo_paso_promedio

## Calculo la proporción de la doble estancia por paso
doble_estancia_media = np.mean(doble_estancia)

print('MÉTODO I')
print("Longitud paso promedio (m): ", long_paso_promedio)
print("Duracion de paso promedio (s): ", tiempo_paso_promedio)
print("Velocidad de marcha (m/s): ", velocidad_marcha)
print("Proporcion de Doble Estancia / Paso: ", doble_estancia_media)
print("Cadencia (pasos/s): ", frec_fund)

## ----------------------------- CÁLCULO DE LA LONGITUD DEL PASO (MÉTODO II) ---------------------------

## Longitud del pie de la persona. Dato a medir y que puede variar el resultado
## Éste valor es necesario para estimar el desplazamiento en la fase de doble estancia
## Si no tengo un valor concreto tomo por defecto 30cm
long_pie = 0.3

## ----------------------------------- GRAFICACIÓN LONGITUD DE PASOS -----------------------------------

plt.scatter(x = np.arange(start = 0, stop = len(long_pasos_m1)), y = long_pasos_m1)
plt.show()

## ----------------------------- CÁLCULO DE LA LONGITUD DEL PASO (MÉTODO II) ---------------------------

## Longitud del pie de la persona. Dato a medir y que puede variar el resultado
## Éste valor es necesario para estimar el desplazamiento en la fase de doble estancia
## Si no tengo un valor concreto tomo por defecto 30cm
long_pie = 0.25

## Creo una lista donde voy a almacenar la longitud de los pasos de la persona
long_pasos_m2 = []

## Itero para cada uno de los segmentos de pasos detectados (IC a IC)
for i in range (len(pasos)):

    ## Hago la segmentación del paso en el tramo IC-TC
    tramo_IC_TC = pos_vert_procesada[pasos[i]['IC'][0] : pasos[i]['TC']]

    ## Hago la segmentación del paso en el tramo TC-IC (donde el IC es el contacto inicial opuesto)
    tramo_TC_IC = pos_vert_procesada[pasos[i]['TC'] : pasos[i]['IC'][1]]

    ## Calculo la variación de altura del centro de masa en base a la diferencia entre el desplazamiento vertical máximo y mínimo
    d_step = abs(max(tramo_TC_IC) - min(tramo_TC_IC))

    ## Calculo longitud del paso sumando la componente single stance y double stance
    long_paso = 2 * np.sqrt(2 * long_pierna * d_step - d_step ** 2) + 0.75 * long_pie

    ## Guargo la longitud de paso en la lista
    long_pasos_m2.append(long_paso)

## ----------------------------- CÁLCULO DE PARÁMETROS DE MARCHA (MÉTODO II) ---------------------------

## Se calcula la longitud de paso promedio
long_paso_promedio = np.mean(long_pasos_m2)

## Se calcula la duración de paso promedio como el inverso de la cadencia
tiempo_paso_promedio = 1 / frec_fund

## Se calcula la velocidad de marcha como el cociente entre éstas cantidades
velocidad_marcha = long_paso_promedio / tiempo_paso_promedio

## Calculo la proporción de la doble estancia por paso
doble_estancia_media = np.mean(doble_estancia)

print('MÉTODO II')
print("Longitud paso promedio (m): ", long_paso_promedio)
print("Duracion de paso promedio (s): ", tiempo_paso_promedio)
print("Velocidad de marcha (m/s): ", velocidad_marcha)
print("Proporcion de Doble Estancia / Paso: ", doble_estancia_media)
print("Cadencia (pasos/s): ", frec_fund)