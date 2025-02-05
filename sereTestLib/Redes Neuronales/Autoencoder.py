## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/autoencoder')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
from ae_train_save_model import *
from parameters import *
from ingesta_etiquetas import *

## ------------------------------------- SEPARACIÓN DE MUESTRAS ----------------------------------------

## Importación de etiquetas
(x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, x_ae_train, x_ae_val, x_estables_ae_train, x_inestables_ae_train, x_estables_ae_val, x_inestables_ae_val) = ingesta_etiquetas()

## <<train>> va a tener el conjunto de IDs de pacientes que se van a usar para el entrenamiento
train = np.concatenate((x_inestables_ae_train, x_estables_ae_train), axis = None)

## <<val_test>> va a tener el conjunto de IDs de pacientes que se van a usar para la validación
val_test = np.concatenate((x_inestables_ae_val, x_estables_ae_val), axis = None)

## ------------------------------------------ ENTRENAMIENTO --------------------------------------------

## <<config>> va a ser un diccionario que tiene parámetros necesarios para configurar el autoencoder
config = {"giro x": girox, "giro z": giroz, "Escalado": escalado, "Loss": loss_name, "Actividad": act_ae, "Preprocesamiento": preprocesamiento, "lr": base_learning_rate}

## Hago el llamado a la función del entrenamiento del autoencoder
ae_train_save_model(dir_preprocessed_data_train, dir_preprocessed_data_test, model_path_ae, inDim, train, val_test, num_epochs, act_ae, batch_size, debug = True)

## ------------------------------------------- VALIDACIÓN ----------------------------------------------
