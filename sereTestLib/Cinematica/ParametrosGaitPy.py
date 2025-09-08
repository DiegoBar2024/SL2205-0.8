## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import pandas as pd
import numpy as np
import sys
import pandas as pd
from LecturaDatosPacientes import *
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/gaitpy/gaitpy')
from gait import *

## ---------------------------------- CÁLCULO DE PARÁMETROS DE MARCHA ----------------------------------

def ParametrosMarcha(id_persona, data, periodoMuestreo, pacientes):

    ## --------------------------------------- DATOS DEL PACIENTE ------------------------------------------

    ## De todo el dataframe de pacientes me quedo únicamente con aquel paciente con el que estoy trabajando actualmente
    ## Debido al filtrado que se realiza previamente, ésto no producirá error ya que el paciente se encuentra en la base de datos
    datos_paciente = pacientes[pacientes['sampleid'] == id_persona]

    ## Obtengo la altura del paciente en cm la cual va a ser utilizada como entrada en el algoritmo
    #altura_paciente = int(datos_paciente.iloc[0]['Talla'])

    ## -------------------------------------- PARÁMETROS DE MARCHA -----------------------------------------

    ## En esta parte se calculan los parámetros de marcha mediante el uso del algoritmo implementado previamente en GaitPy
    ## Expreso el vector de tiempos tomando como t = 0 el instante inicial de la sesión (de manera relativa)
    ## Se recuerda que los tiempos en dicho vector se encuentran expresados en milisegundos
    tiempos = np.array(data['Time']) - data['Time'][0]

    ## Obtengo la aceleración vertical a efectos de ejecutar el algoritmo
    acc_VT = np.array(data['AC_y'])
    
    ## Comprobación de que la aceleración vertical se encuentre orientada correctamente
    ## En caso de que el valor medio de la aceleración vertical sea negativo, quiere decir que el versor está hacia abajo
    if np.mean(acc_VT) < 0:

        ## En dicho caso invierto el signo de la aceleración vertical para que quede consistente
        acc_VT = - acc_VT

    ## Creo el dataframe cuyas columnas van a ser los instantes de tiempos y los valores de aceleración vertical respectivamente
    ## Por defecto el algoritmo toma el tiempo en milisegundos y las aceleraciones en metros por segundo cuadrado
    dataframe_vert = pd.DataFrame(data = {'timestamps' : tiempos, 'y' : acc_VT})

    ## Creo un nuevo objeto de la clase GaitPy pasando como parámetro la frecuencia de muestreo y los datos
    gait_py = Gaitpy(data = dataframe_vert, sample_rate = 1 / periodoMuestreo)

    altura_paciente = 175

    ## Obtengo las características de la marcha del sujeto. Debo pasar como parámetro la altura del sujeto en centímetros
    features = gait_py.extract_features(subject_height = altura_paciente)
    
    ## Retorno la lista de características correspondientes al paciente
    return features, acc_VT

    # lista_features = ['bout_number', 'bout_length_sec', 'bout_start_time', 'IC', 'FC',
    #           'gait_cycles', 'steps', 'stride_duration', 'stride_duration_asymmetry',
    #           'step_duration', 'step_duration_asymmetry', 'cadence',
    #           'initial_double_support', 'initial_double_support_asymmetry',
    #           'terminal_double_support', 'terminal_double_support_asymmetry',
    #           'double_support', 'double_support_asymmetry', 'single_limb_support',
    #           'single_limb_support_asymmetry', 'stance', 'stance_asymmetry', 'swing',
    #           'swing_asymmetry', 'step_length', 'step_length_asymmetry',
    #           'stride_length', 'stride_length_asymmetry', 'gait_speed']