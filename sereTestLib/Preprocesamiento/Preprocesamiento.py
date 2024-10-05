####### IMPORTO LA CARPETA DONDE ESTÁN LOS PARÁMETROS ########
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Preprocesamiento')
##############################################################

from parameters  import *
from preprocesamiento_train import preprocesamiento_train
from preprocesamiento_train2 import preprocesamiento_train2
from preprocesamiento_val_test import preprocesamiento_val_test
from preprocesamiento_val_test2 import preprocesamiento_val_test2
from utils.ingesta_etiquetas import ingesta_etiquetas
import wandb

## Función que se encarga de preprocesar los datos de los pacientes cuyas IDs se especifican en las listas <<train>> y <<val_test>> y las guarda usando wandb
## <<train>> va a ser una lista de IDs de pacientes que se van a preprocesar para luego usarse como entrenamiento (ya sea del autoencoder y clasificador)
## <<val_test>> va a ser una lista de IDs de pacientes que se van a preprocesar para luego usarse como validación (ya sea del autoencoder y clasificador)
## Se abre primero una sesión wandb, se crea un artefacto, se agrega una referencia a la carpeta de Dataset y luego se loguea
def Preprocesamiento(train, val_test, upload_wandb = True):

    ## En caso de que <<upload_wandb>> sea True. Por defecto se tiene que <<upload_wandb>> = True
    ################################## DEJO COMENTADO LO DEL WANDB #################################################
    # if upload_wandb:

    #     ## <<project_wandb>> me da el nombre del proyecto el cual se extrae de <<parameters.py>> y es "SereTest-data"
    #     ## <<job_type>> especifica el tipo de tarea a realizar. En éste caso se está cargando el dataset de modo que se tiene <<job_type>> = "load dataset"
    #     run = wandb.init(project = project_wandb, job_type = "load dataset")
        
    #     print("PASO 1 -  preprocesar datos")
    ################################## DEJO COMENTADO LO DEL WANDB #################################################
    
    ## En caso de que el prepocesamiento tiene valor 1. Ésta tiene una menor cantidad de etapas de preprocesamiento Ésto se extrae de <<parameters.py>>
    if preprocesamiento == 1:

        ## Itero para cada uno de los pacientes en el conjunto de entrenamiento
        ## En particular, la iteración se hace para cada uno de los IDs de pacientes que se van a usar para entrenar y procesar
        for i in train:

            ## Se llama a la función <<preprocesamiento_train>> pasando como argumento el ID del paciente
            ## De ésta manera se hace el preprocesamiento (reduccion/correccion ejes, segmentacion, transformada en escalogramas, augmentado) correspondiente a los segmentos del paciente cuya ID se pasa como parámetro
            preprocesamiento_train(i)

        ## Itero para cada uno de los pacientes en el conjunto de validación
        ## En particular, la iteración se hace para cada uno de los IDs de pacientes que se van a usar para entrenar y procesar
        for j in val_test:

            ## Se llama a la función <<preprocesamiento_val_test>> pasando como argumento el ID del paciente
            ## De ésta manera se hace el preprocesamiento (reduccion/correccion ejes, segmentacion, transformada en escalogramas, augmentado) correspondiente a los segmentos del paciente cuya ID se pasa como parámetro
            preprocesamiento_val_test(j)

    ## En caso de que el preprocesamiento tenga valor 2. Ésta tiene una mayor cantidad de etapas de preprocesamiento que la 1. Se extrae a partir de <<parameters.py>>
    elif preprocesamiento == 2:

        ## Itero para cada uno de los pacientes en el conjunto de entrenamiento
        ## En particular, la iteración se hace para cada uno de los IDs de pacientes que se van a usar para entrenar y procesar
        for i in train:

            ## Se llama a la función <<preprocesamiento_train2>> pasando como argumento el ID del paciente
            ## De ésta manera se hace el preprocesamiento (reduccion/correccion ejes, segmentacion, filtrado, clasificacion de estado, renombrado de segmentos, transformada en escalogramas, augmentado) correspondiente a los segmentos del paciente cuya ID se pasa como parámetro
            preprocesamiento_train2(i)

        ## Itero para cada uno de los pacientes en el conjunto de validación
        ## En particular, la iteración se hace para cada uno de los IDs de pacientes que se van a usar para entrenar y procesar
        for j in val_test:

            ## Se llama a la función <<preprocesamiento_val_test2>> pasando como argumento el ID del paciente
            ## De ésta manera se hace el preprocesamiento (reduccion/correccion ejes, segmentacion, filtrado, clasificacion de estado, renombrado de segmentos, transformada en escalogramas, augmentado) correspondiente a los segmentos del paciente cuya ID se pasa como parámetro
            preprocesamiento_val_test2(j)

    ################################## DEJO COMENTADO LO DEL WANDB #################################################
    ## En caso de que <<upload_wandb>> sea True. Por defecto se tiene que <<upload_wandb>> = True
    # if upload_wandb:

    #     ## Creacion de un Artefacto Wandb
    #     ## <<extra>> me da el extra en el nombre del archivo. Se extrae del fichero <<parameters.py>>
    #     ## <<preprocesamiento>> me da el tipo de preprocesamiento que hice. Puede ser '1' o '2'. Se extrae del fichero <<parameters.py>>
    #     ## Se asigna un nombre al Artifact en base al <<extra>> y al preprocesamiento que tengo
    #     my_data = wandb.Artifact(extra + str(preprocesamiento), type = "raw_data")

    #     ## <<data_dir>> va a ser la ruta a la carpeta Dataset del proyecto
    #     ## Para el caso de MI computadora es: <<data_dir>> = 'C:/Yo/Tesis/sereData/sereData/Dataset'
    #     ## Especifico una cantidad máxima de objetos de 90000 con <<max_objects>>
    #     my_data.add_reference(data_dir, max_objects = 90000)
    
    #     ## Se logea el artefacto al run wandb
    #     run.log_artifact(my_data)

    #     ## Se termina la sesión wandb
    #     run.finish()
    ################################## DEJO COMENTADO LO DEL WANDB #################################################
    

if __name__== '__main__':
    
    ## Llamo a la función <<ingesta_etiquetas>> la cual me trae todas las listas de pacientes que van a usarse para entrenamiento y validación de clasificador y autoencoder
    x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, x_ae_train, x_ae_val, x_estables_ae_train, x_inestables_ae_train, x_estables_ae_val, x_inestables_ae_val = ingesta_etiquetas()
    
    ## <<train>> me va a guardar la lista de IDs de los pacientes cuyas muestras van a ser usadas para ENTRENAR tanto el clasificador como el autoencoder
    train = np.concatenate((x_inestables_train_clf, x_estables_train_clf, x_ae_train), axis = None)

    ## <<val_test>> me va a guardar una lista de IDs de los pacientes cuyas muestras van a ser usadas para VALIDAR y TESTEAR del autoencoder y el clasificador
    val_test = np.concatenate((x_ae_val, x_inestables_val_clf, x_estables_val_clf), axis = None)

    ## Se hace el llamado a la función de preprocesamiento pasando como argumento las listas <<train>> y <<val_test>> para entrenamiento y validación
    Preprocesamiento(train, val_test)