## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
import wandb
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/clasificador')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
from parameters import *
from entrenamiento_clasificador import entrenamiento_clasificador
from ingesta_etiquetas import *

## ------------------------------------- SEPARACIÓN DE MUESTRAS ----------------------------------------

## Importación de etiquetas haciendo el llamado
(x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, x_ae_train, x_ae_val, x_estables_ae_train, x_inestables_ae_train, x_estables_ae_val, x_inestables_ae_val) = ingesta_etiquetas()

## <<train>> va a tener el conjunto de IDs de pacientes que se van a usar para el entrenamiento
train = np.concatenate((x_inestables_train_clf, x_estables_train_clf), axis = None)

## <<val_test>> va a tener el conjunto de IDs de pacientes que se van a usar para la validación
val_test = np.concatenate((x_inestables_val_clf, x_estables_val_clf), axis = None)

## ------------------------------------------ ENTRENAMIENTO --------------------------------------------

## Inicio una sesión Wandb
run = wandb.init(project = "Autoencoder", reinit = True, job_type = "load ae")

try:
    modelo_artifact = run.use_artifact(autoencoder_name + ':best')

except:
    modelo_artifact = run.use_artifact(autoencoder_name + ':latest')

## Cargo el modelo del autoencoder
modelo_dir = modelo_artifact.download(model_path_ae)

run.finish()

## Cargo el modelo del autoencoder a partir de la dirección determinada
modelo_autoencoder = ae_load_model(modelo_dir)

## Especifico configuración para el entrenamiento del clasificacor
config = {"giro x": girox, "giro z": giroz, "Escalado": escalado, "Clasificador": clasificador, "Actividad": act_clf, "AE": autoencoder_name, "Preprocesamiento": preprocesamiento}

## Inicio una sesión <<wandb>> para el entrenamiento del clasificador
run = wandb.init(project = "SereTest-clasificador", reinit = True, config = config, job_type = "train clf", name = clasificador_name)

## Llamo a la función del entrenamiento del clasificador
entrenamiento_clasificador(x_inestables_train, x_estables_train, x_inestables_val, x_estables_val, modelo_autoencoder, clasificador, dir_preprocessed_data_train, dir_preprocessed_data_test, act_clf)

## Configuración restante de guardado en <<wandb>>
trained_model_artifact = wandb.Artifact(clasificador_name, type = "model")
trained_model_artifact.add_dir(model_path_clf)
run.log_artifact(trained_model_artifact)
run.finish()