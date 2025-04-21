## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import pandas as pd
import numpy as np
import sys
import pandas as pd
from scipy import signal
from harm_analysis import *
from control import *
from LecturaDatos import *

sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/gaitpy/gaitpy')
from gait import *

## -------------------------------------- PARÁMETROS DE MARCHA -----------------------------------------

## Expreso el vector de tiempos tomando como t = 0 el instante inicial de la sesión
## Se recuerda que los tiempos en dicho vector se encuentran expresados en milisegundos
tiempos = np.array(data['Time']) - data['Time'][0]

## Obtengo la aceleración vertical a efectos de ejecutar el algoritmo
acc_VT = np.array(data['AC_y'])

## Creo el dataframe cuyas columnas van a ser los instantes de tiempos y los valores de aceleración vertical respectivamente
## Por defecto el algoritmo toma el tiempo en milisegundos y las aceleraciones en metros por segundo cuadrado
dataframe_vert = pd.DataFrame(data = {'timestamps' : tiempos, 'y' : acc_VT})

## Creo un nuevo objeto de la clase GaitPy
gait_py = Gaitpy(data = dataframe_vert, sample_rate = 1 / periodoMuestreo)

## Obtengo las características de la marcha del sujeto. Debo pasar como parámetro la altura del sujeto en centímetros
features = gait_py.extract_features(subject_height = 190)

## Obtengo la lista de características que son extraídas de la señal de marcha
lista_features = features.columns

# lista_features = ['bout_number', 'bout_length_sec', 'bout_start_time', 'IC', 'FC',
#           'gait_cycles', 'steps', 'stride_duration', 'stride_duration_asymmetry',
#           'step_duration', 'step_duration_asymmetry', 'cadence',
#           'initial_double_support', 'initial_double_support_asymmetry',
#           'terminal_double_support', 'terminal_double_support_asymmetry',
#           'double_support', 'double_support_asymmetry', 'single_limb_support',
#           'single_limb_support_asymmetry', 'stance', 'stance_asymmetry', 'swing',
#           'swing_asymmetry', 'step_length', 'step_length_asymmetry',
#           'stride_length', 'stride_length_asymmetry', 'gait_speed']

## Obtengo un dataframe con los contactos iniciales (expresadas en milisegundos de sesión)
features_ICs = features['IC']

## Obtengo un dataframe aislando las cadencias (medida en pasos/minuto)
features_cadencia = features['cadence']

## Obtengo un dataframe aislando las longitudes de paso (medida en metros)
features_pasos = features['step_length']

## Obtengo un vector con los contactos iniciales expresados en términos de muestras
ICs_muestras = features_ICs / (1000 * periodoMuestreo)