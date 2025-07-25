## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from ContactosIniciales import *
from ContactosTerminales import *
from ParametrosGaitPy import *
from LongitudPasoM1 import LongitudPasoM1
from SegmentacionGaitPy import Segmentacion as SegmentacionGaitPy
from Segmentacion import Segmentacion
import sys
import pandas as pd
from LecturaDatosPacientes import *
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/gaitpy/gaitpy')
from ParametrosGaitPy import *

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## La idea de ésta parte consiste en poder hacer una discriminación entre reposo y actividad
## Especifico la ruta en la cual se encuentra el registro a leer
ruta_registro = 'C:/Yo/Tesis/sereData/sereData/Registros/MarchaLibre_Sabrina.txt'

##  Hago la lectura de los datos
data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = 299, lectura_datos_propios = False, ruta = ruta_registro)

## ------------------------------------- CÁLCULO DE ESTADÍSTICAS ---------------------------------------

## MÉTODO I: ALGORITMO PROPIO
## Cálculo de contactos iniciales
contactos_iniciales, muestras_paso, acc_AP_norm, frec_fund = ContactosIniciales(acel, cant_muestras, periodoMuestreo, graficar = True)

## Cálculo de contactos terminales
contactos_terminales = ContactosTerminales(acel, cant_muestras, periodoMuestreo, graficar = True)

## Hago la segmentación de la marcha
pasos, duraciones_pasos, giros = Segmentacion(contactos_iniciales, contactos_terminales, muestras_paso, periodoMuestreo, acc_AP_norm, gyro)

## Cálculo de parámetros de marcha
pasos_numerados, frecuencias, velocidades, long_pasos_m1, coeficientes_m1 = LongitudPasoM1(pasos, acel, tiempo, periodoMuestreo, frec_fund, duraciones_pasos, id_persona = 1)

# ## MÉTODO II: USANDO GAITPY
# ## Hago la lectura de los datos generales de los pacientes
# pacientes, ids_existentes = LecturaDatosPacientes()

# ## Hago el cálculo de los parámetros de marcha usando GaitPy
# features_gp, acc_VT = ParametrosMarcha(id_persona, data, periodoMuestreo, pacientes)

# ## Hago la segmentación usando GaitPy
# pasos_gp = SegmentacionGaitPy(features_gp, periodoMuestreo, acc_VT, plot = True)

# ## Cantidad de pasos
# print("Pasos GP: {}".format(len(pasos_gp)))
# print("Pasos metodo: {}".format(len(pasos)))

# ## Duración de pasos
# print("Duracion GP: {}".format(np.mean(np.array(features_gp['step_duration']))))
# print("Duración metodo: {}".format(np.mean(duraciones_pasos)))

# ## Cadencia de pasos
# print("Cadencia GP: {}".format(np.mean(np.array(features_gp['cadence'])) / 60))
# print("Cadencia metodo: {}".format(np.mean(frecuencias)))

# ## Longitud de pasos
# print("Longitud GP: {}".format(np.mean(np.array(features_gp['step_length']))))
# print("Longitud metodo: {}".format(np.mean(long_pasos_m1)))

print(giros)

plt.plot(gyro[:, 1])
plt.show()