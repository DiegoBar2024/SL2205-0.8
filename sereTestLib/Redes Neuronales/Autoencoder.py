## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/autoencoder')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Wavelets')
from ae_train_save_model import *
from parameters import *
from ingesta_etiquetas import *
import wandb
import keras
from Modelo_AE import *
from Etiquetado import *

## ------------------------------------- SEPARACIÓN DE MUESTRAS ----------------------------------------

## Especifico la ruta donde se encuentran los archivos
ruta_escalogramas =  'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo/train/'

## Obtengo una lista de todos los archivos presentes en la ruta donde se encuentran los escalogramas nuevos
archivos_escalogramas = [archivo for archivo in os.listdir(ruta_escalogramas)]

## Importación de etiquetas (provenientes del fichero de ingesta_etiquetas())
(x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, x_ae_train, x_ae_val, x_estables_ae_train, x_inestables_ae_train, x_estables_ae_val, x_inestables_ae_val) = ingesta_etiquetas() 

## <<train>> va a tener el conjunto de IDs de pacientes que se van a usar para el entrenamiento
train = np.concatenate((x_inestables_ae_train, x_estables_ae_train), axis = None)

## <<val_test>> va a tener el conjunto de IDs de pacientes que se van a usar para la validación
val_test = np.concatenate((x_inestables_ae_val, x_estables_ae_val), axis = None)

## ------------------------------------- SELECCIÓN DE MUESTRAS -----------------------------------------

## ESTE PROCESO SE HACE PARA NO TENER QUE CALCULAR LOS ESCALOGRAMAS DE TODOS LOS PACIENTES PARA ENTRENAR LA RED
## Creo una lista con los identificadores de aquellos pacientes cuyos escalogramas están en la carpeta
lista_pacientes_existentes = np.zeros(len(archivos_escalogramas))

## Itero para cada uno de los ficheros de escalogramas existentes
for i in range (1, len(archivos_escalogramas)):

    ## Selecciono el identificador del paciente asociado al archivo
    id_paciente_archivo = int(archivos_escalogramas[i][1:])

    ## Agrego el paciente detectado a la lista de pacientes existentes
    lista_pacientes_existentes[i] = id_paciente_archivo

## Hago que los pacientes de entrenamiento sea la intersección entre los pacientes cuyo escalograma existe y los pacientes de entrenamiento originales
train = np.intersect1d(lista_pacientes_existentes, train)

## Hago que los pacientes de validación sea la intersección entre los pacientes cuyo escalograma existe y los pacientes de validación originales
val_test = np.intersect1d(lista_pacientes_existentes, val_test)

## ----------------------------------- ENTRENAMIENTO Y VALIDACIÓN --------------------------------------

## <<config>> va a ser un diccionario que tiene parámetros necesarios para configurar el autoencoder
config = {"giro x": girox, "giro z": giroz, "Escalado": escalado, "Loss": loss_name, "Actividad": act_ae, "Preprocesamiento": preprocesamiento, "lr": base_learning_rate}

## Comienza la sesión wandb cuyo objetivo va a ser guardar los parámetros del autoencoder luego del entrenamiento
run = wandb.init(project = "Autoencoder", reinit = True, config = config, job_type = "train ae", name = autoencoder_name)

## Se hace un reset de los estados        
keras.backend.clear_session()

## Hago el llamado a la función del entrenamiento del autoencoder
ae_train_save_model(dir_escalogramas_nuevo_train, dir_escalogramas_nuevo_test, model_path_ae, inDim, train, val_test, num_epochs, act_ae, batch_size, debug = True)

## Terminación del autoencoder
trained_model_artifact = wandb.Artifact(autoencoder_name, type = "model")
trained_model_artifact.add_dir(model_path_ae)
run.log_artifact(trained_model_artifact)
wandb.run.finish()