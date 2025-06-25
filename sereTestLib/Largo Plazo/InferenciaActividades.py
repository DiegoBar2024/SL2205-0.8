## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from LecturaDatos import *
from Muestreo import *
from LecturaDatosPacientes import *
import numpy as np
import pandas as pd
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from scipy import *
import json
import seaborn as sns
from scipy.stats import *
from sklearn.svm import SVC
from joblib import dump, load
from DeteccionActividades import DeteccionActividades

## ------------------------------- DISCRIMINACIÓN ACTIVIDAD - REPOSO -----------------------------------

## Especifico la ruta en la cual se encuentra el registro a leer
ruta_registro = 'C:/Yo/Tesis/sereData/sereData/Registros/Actividades_Sabrina.txt'

##  Hago la lectura de los datos
data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = None, lectura_datos_propios = True, ruta = ruta_registro)

## Defino la cantidad de muestras de la ventana que voy a tomar
muestras_ventana = 200

## Defino la cantidad de muestras de solapamiento entre ventanas
muestras_solapamiento = 100

## Hago el cálculo del vector de SMA para dicha persona
vector_SMA, features = DeteccionActividades(acel, tiempo, muestras_ventana, muestras_solapamiento, periodoMuestreo, cant_muestras, actividad = None)

## Cargo el modelo del clasificador ya entrenado según la ruta del clasificador
clf_entrenado = load("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/SVM.joblib")

## Determino la predicción del clasificador ante mi muestra de entrada
## Etiqueta 0: Reposo
## Etiqueta 1: Movimiento
pat_predictions = clf_entrenado.predict(np.array((vector_SMA)).reshape(-1, 1))

## Defino el espaciado de las ventanas
ventanas = np.arange(0, 200, 100)

## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
plt.plot(tiempo, acel[:,0], color = 'r', label = '$a_x$')
plt.plot(tiempo, acel[:,1], color = 'b', label = '$a_y$')
plt.plot(tiempo, acel[:,2], color = 'g', label = '$a_z$')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleracion $(m/s^2)$")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()

## ------------------------------- DISCRIMINACIÓN ACTIVIDADES -----------------------------------

## Cargo el modelo del clasificador entre actividades ya entrenado según la ruta del clasificador
clf_act_entrenado = load("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/LDA_Nuevo_Actividades.joblib")

## Determino la predicción del clasificador ante mi muestra de entrada
## Etiqueta 0: Escaleras
## Etiqueta 1: Parado
## Etiqueta 2: Sentado
## Etiqueta 3: Caminando
pat_act_predictions = clf_act_entrenado.predict(np.array(features))

print(pat_act_predictions)