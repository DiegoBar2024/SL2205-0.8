## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/clasificador')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from parameters import *
from Modelo_AE import *
from Modelo_CLAS import *
from Etiquetado import *
from Modelo_CLAS import *
from LecturaDatosPacientes import *

## ------------------------------------------- ETIQUETADOS ---------------------------------------------

## Importación de etiquetas (provenientes del fichero de ingesta_etiquetas())
(x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, x_ae_train, x_ae_val) = ingesta_etiquetas() 

## <<train>> va a tener el conjunto de IDs de pacientes que se van a usar para el entrenamiento
train = np.concatenate((x_inestables_train_clf, x_estables_train_clf), axis = None)

## <<val_test>> va a tener el conjunto de IDs de pacientes que se van a usar para la validación
val_test = np.concatenate((x_inestables_val_clf, x_estables_val_clf), axis = None)

## ------------------------------------- SELECCIÓN DE MUESTRAS -----------------------------------------

## Obtengo la lista de todos los pacientes para los que se cuentan registros en la base de datos
pacientes, ids_existentes = LecturaDatosPacientes()

## Obtengo la lista de los pacientes inestables para el entrenamiento del clasificador
train_inestables = np.intersect1d(ids_existentes, x_inestables_train_clf)

## Obtengo la lista de los pacientes estables para el entrenamiento del clasificador
train_estables = np.intersect1d(ids_existentes, x_estables_train_clf)

## Obtengo la lista de los pacientes inestables para la validación del clasificador
val_inestables = np.intersect1d(ids_existentes, x_inestables_val_clf)

## Obtengo la lista de los pacientes estables para la validación del clasificador
val_estables = np.intersect1d(ids_existentes, x_estables_val_clf)

## Obtengo el conjunto de estables
id_estables = np.concatenate((train_estables, val_estables))

## Obtengo el conjunto de inestables
id_inestables = np.concatenate((train_inestables, val_inestables))

## ---------------------------------- CARGADO DE COMPRIMIDAS ESTABLES ----------------------------------

## Especifico la ruta en la cual se encuentran las muestras comprimidas
ruta_comprimidas = "C:/Yo/Tesis/sereData/sereData/Dataset/latente_ae"

## Construyo una lista donde voy a guardar las muestras etiquetadas como estables
estables = np.zeros((1, 256))

## Construyo una lista donde voy a guardar las muestras etiquetadas como inestables
inestables = np.zeros((1, 256))

## Itero para cada uno de los IDs considerados como estables
for estable in id_estables:

    ## Construyo un bloque try except para atrapar errores
    try:

        ## Abro el archivo donde tengo la muestra comprimida
        comprimida_estable = np.load(ruta_comprimidas + '/S{}/S{}_latente.npz'.format(estable, estable))

        ## Selecciono la representación en espacio latente de los segmentos de marcha
        latente_estable = comprimida_estable['X']

        ## Agrego dicho espacio latente a la lista de estables
        estables = np.concatenate((estables, latente_estable), axis = 0)

    ## En caso de que haya algún error
    except:

        ## Que continúe con la siguiente muestra
        continue

## -------------------------------- CARGADO DE COMPRIMIDAS INESTABLES ----------------------------------

## Itero para cada uno de los IDs considerados como inestables
for inestable in id_inestables:

    ## Construyo un bloque try except para atrapar errores
    try:

        ## Abro el archivo donde tengo la muestra comprimida
        comprimida_inestable = np.load(ruta_comprimidas + '/S{}/S{}_latente.npz'.format(inestable, inestable))

        ## Selecciono la representación en espacio latente de los segmentos de marcha
        latente_inestable = comprimida_inestable['X']

        ## Agrego dicho espacio latente a la lista de estables
        inestables = np.concatenate((inestables, latente_inestable), axis = 0)

    ## En caso de que haya algún error
    except:

        ## Que continúe con la siguiente muestra
        continue

## Selecciono únicamente aquellas muestras no nulas (elimino el dummy vector) para estables
estables = estables[1:,:]

## Selecciono únicamente aquellas muestras no nulas (elimino el dummy vector) para inestables
inestables = inestables[1:,:]