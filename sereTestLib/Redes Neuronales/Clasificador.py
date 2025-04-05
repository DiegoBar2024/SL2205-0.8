## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
import wandb
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/clasificador')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
from parameters import *
from Modelo_AE import *
from entrenamiento_clasificador import entrenamiento_clasificador
from Etiquetado import *
from Modelo_CLAS import *

## ------------------------------------- SEPARACIÓN DE MUESTRAS ----------------------------------------

## Especifico la ruta donde se encuentran los archivos
ruta_escalogramas =  'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo/train/'

## Obtengo una lista de todos los archivos presentes en la ruta donde se encuentran los escalogramas nuevos
archivos_escalogramas = [archivo for archivo in os.listdir(ruta_escalogramas)]

## Importación de etiquetas (provenientes del fichero de ingesta_etiquetas())
(x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, x_ae_train, x_ae_val, x_estables_ae_train, x_inestables_ae_train, x_estables_ae_val, x_inestables_ae_val) = ingesta_etiquetas() 

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

## Inicio una sesión Wandb que me permita cargar el modelo del autoencoder
run = wandb.init(project = "Autoencoder", reinit = True, job_type = "load ae")

try:
    modelo_artifact = run.use_artifact(autoencoder_name + ':best')

except:
    modelo_artifact = run.use_artifact(autoencoder_name + ':latest')

## Cargo la dirección en la cual se encuentra almacenado el modelo del autoencoder
modelo_dir = modelo_artifact.download(model_path_ae)

run.finish()

## Cargo el modelo del autoencoder a partir de la dirección determinada
modelo_autoencoder = ae_load_model(modelo_dir)

## Especifico configuración para el entrenamiento del clasificacor
config = {"giro x": girox, "giro z": giroz, "Escalado": escalado, "Clasificador": clasificador, "Actividad": act_clf, "AE": autoencoder_name, "Preprocesamiento": preprocesamiento}

## Inicio una sesión <<wandb>> para el entrenamiento del clasificador
run = wandb.init(project = "Clasificador", reinit = True, config = config, job_type = "train clf", name = clasificador_name)

## Llamo a la función del entrenamiento del clasificador
entrenamiento_clasificador(train_inestables, train_estables, val_inestables , val_estables, modelo_autoencoder, clasificador, dir_escalogramas_nuevo_train, dir_escalogramas_nuevo_test, act_clf)

## Configuración restante de guardado en <<wandb>>
trained_model_artifact = wandb.Artifact(clasificador_name, type = "model")
trained_model_artifact.add_dir(model_path_clf)
run.log_artifact(trained_model_artifact)
run.finish()