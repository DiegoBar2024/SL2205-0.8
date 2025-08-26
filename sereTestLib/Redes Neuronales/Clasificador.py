## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/clasificador')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
from parameters import *
from Modelo_AE import *
from Modelo_CLAS import *
from Etiquetado import *
from Modelo_CLAS import *

## ------------------------------------- SEPARACIÓN DE MUESTRAS ----------------------------------------

## Obtengo una lista de todos los archivos presentes en la ruta donde se encuentran los escalogramas nuevos
archivos_escalogramas = [archivo for archivo in os.listdir(ruta_escalogramas + '/')]

## Importación de etiquetas (provenientes del fichero de ingesta_etiquetas())
(x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, x_ae_train, x_ae_val) = ingesta_etiquetas() 

## <<train>> va a tener el conjunto de IDs de pacientes que se van a usar para el entrenamiento
train = np.concatenate((x_inestables_train_clf, x_estables_train_clf), axis = None)

## <<val_test>> va a tener el conjunto de IDs de pacientes que se van a usar para la validación
val_test = np.concatenate((x_inestables_val_clf, x_estables_val_clf), axis = None)

## ------------------------------------- SELECCIÓN DE MUESTRAS -----------------------------------------

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

## ----------------------------------- ENTRENAMIENTO Y VALIDACIÓN --------------------------------------

## Especifico el nombre del modelo del autoencoder
nombre_autoencoder = 'AutoencoderUCU_nuevo'

## Especifico el tipo de clasificador que voy a entrenar
clasificador = 'lda_nuevo'

## Especifico el nombre que le voy a dar al modelo del clasificador
clasificador_name = 'ClasificadorUCU_{}'.format(clasificador)

## Cargo el modelo del autoencoder a partir de la dirección determinada
modelo_autoencoder = ae_load_model(ruta_ae, nombre_autoencoder)

## Llamo a la función del entrenamiento del clasificador
entrenamiento_clasificador(clasificador_name, train_inestables, train_estables, val_inestables, val_estables, modelo_autoencoder, clasificador, ruta_escalogramas, act_clf)