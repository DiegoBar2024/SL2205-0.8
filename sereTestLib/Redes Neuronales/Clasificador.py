## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/clasificador')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
from parameters import *
from entrenamiento_clasificador import *
from ingesta_etiquetas import *

## ------------------------------------- SEPARACIÓN DE MUESTRAS ----------------------------------------

## Importación de etiquetas haciendo el llamado
(x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, x_ae_train, x_ae_val, x_estables_ae_train, x_inestables_ae_train, x_estables_ae_val, x_inestables_ae_val) = ingesta_etiquetas()

## <<train>> va a tener el conjunto de IDs de pacientes que se van a usar para el entrenamiento
train = np.concatenate((x_inestables_train_clf, x_estables_train_clf), axis = None)

## <<val_test>> va a tener el conjunto de IDs de pacientes que se van a usar para la validación
val_test = np.concatenate((x_inestables_val_clf, x_estables_val_clf), axis = None)

## ------------------------------------------ ENTRENAMIENTO --------------------------------------------

# ## <<config>> va a ser un diccionario que tiene parámetros necesarios para configurar el clasficador
# config = {"giro x": girox, "giro z": giroz, "Escalado": escalado, "Clasificador": clasificador, "Actividad": act_clf, "AE": autoencoder_name, "Preprocesamiento": preprocesamiento}

# # ## Hago el llamado a la función del entrenamiento del clasificador
# # entrenamiento_clasificador(x_inestables_train, x_estables_train, x_inestables_val, x_estables_val, modelo_autoencoder, clasificador, dir_preprocessed_data_train, dir_preprocessed_data_test, act_clf)