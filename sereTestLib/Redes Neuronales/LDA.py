## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/clasificador')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
from parameters import *
from Modelo_AE import *
import os
from Modelo_AE import *
from Evaluar_AE import *
from Etiquetado import *

## ------------------------------------- CARGADO DEL CLASIFICADOR --------------------------------------

## Especifico la ruta del clasificador que voy a cargar
ruta_clasificador = "C:/Yo/Tesis/sereData/sereData/Modelos/clasificador/ModeloClasificador/ClasificadorUCU_lda.joblib"

## Cargo el clasificador previamente entrenado
lda = load(ruta_clasificador)

## Obtengo las probabilidades de ocurrencia de cada clase (probabilidades previas)
## El primer elemento va a ser la probabilidad de ocurrencia de estables (etiqueta 0)
## El segundo elemento va a ser la probabilidad de ocurrencia de inestables (etiqueta 1)
prob_clases = lda.priors_

## ------------------------------------- CARGADO DEL ETIQUETAS -----------------------------------------

## Especifico la ruta donde van a estar los escalogramas de entrenamiento
ruta_escalogramas = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo_gp'

## Obtengo una lista de todos los archivos presentes en la ruta donde se encuentran los escalogramas nuevos
archivos_escalogramas = [archivo for archivo in os.listdir(ruta_escalogramas + '/')]

## Importación de etiquetas (provenientes del fichero de ingesta_etiquetas())
(x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, x_ae_train, x_ae_val) = ingesta_etiquetas() 

## <<train>> va a tener el conjunto de IDs de pacientes que se van a usar para el entrenamiento
train = np.concatenate((x_inestables_train_clf, x_estables_train_clf), axis = None)

## <<val_test>> va a tener el conjunto de IDs de pacientes que se van a usar para la validación
val_test = np.concatenate((x_inestables_val_clf, x_estables_val_clf), axis = None)

## ------------------------------- DETERMINACIÓN ESTABLES - INESTABLES ---------------------------------

## ESTE PROCESO SE HACE PARA NO TENER QUE CALCULAR LOS ESCALOGRAMAS DE TODOS LOS PACIENTES PARA ENTRENAR LA RED
## Creo una lista con los identificadores de aquellos pacientes cuyos escalogramas están en la carpeta
lista_pacientes_existentes = np.zeros(len(archivos_escalogramas))

## Itero para cada uno de los ficheros de escalogramas existentes
for i in range (0, len(archivos_escalogramas)):

    ## Selecciono el identificador del paciente asociado al archivo
    id_paciente_archivo = int(archivos_escalogramas[i][1:])

    ## Agrego el paciente detectado a la lista de pacientes existentes
    lista_pacientes_existentes[i] = id_paciente_archivo

## Obtengo la lista de los pacientes inestables para el entrenamiento del clasificador
train_inestables = np.intersect1d(lista_pacientes_existentes, x_inestables_train_clf)

## Obtengo la lista de los pacientes estables para el entrenamiento del clasificador
train_estables = np.intersect1d(lista_pacientes_existentes, x_estables_train_clf)

## Obtengo la lista de los pacientes inestables para la validación del clasificador
val_inestables = np.intersect1d(lista_pacientes_existentes, x_inestables_val_clf)

## Obtengo la lista de los pacientes estables para la validación del clasificador
val_estables = np.intersect1d(lista_pacientes_existentes, x_estables_val_clf)

## ------------------------------------- DISTRIBUCIÓN POBLACIONAL --------------------------------------

## Creo un diccionario cuya clave va a ser la etiqueta y el valor va a ser la cantidad de segmentos que tenga asociadas a esa etiqueta
cantidad_segmentos = {'0' : 0, '1' : 0}

## Itero para cada uno de los pacientes estables de entrenamiento (etiqueta 0)
for estable in train_estables:

    ## Obtengo la ruta completa hacia los escalogramas del paciente
    ruta_estable = ruta_escalogramas + '/S{}'.format(int(estable))

    ## Obtengo el numero escalogramas que tengo del paciente
    cantidad_escalogramas = len([archivo for archivo in os.listdir(ruta_estable)])

    ## Agrego la cantidad de escalogramas del paciente estable al diccionario correspondiente
    cantidad_segmentos['0'] += cantidad_escalogramas

## Itero para cada uno de los pacientes inestables de entrenamiento (etiqueta 1)
for estable in train_inestables:

    ## Obtengo la ruta completa hacia los escalogramas del paciente
    ruta_estable = ruta_escalogramas + '/S{}'.format(int(estable))

    ## Obtengo el numero escalogramas que tengo del paciente
    cantidad_escalogramas = len([archivo for archivo in os.listdir(ruta_estable)])

    ## Agrego la cantidad de escalogramas del paciente estable al diccionario correspondiente
    cantidad_segmentos['1'] += cantidad_escalogramas