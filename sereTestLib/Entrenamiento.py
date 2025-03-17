####### IMPORTO LA CARPETA DONDE ESTÁN LOS PARÁMETROS ########
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
##############################################################

from Preprocesamiento.Preprocesamiento import Preprocesamiento
from utils.ingesta_etiquetas import ingesta_etiquetas_concat
from parameters  import *
from autoencoder.ae_train_save_model import ae_train_save_model, ae_load_model
from clasificador.entrenamiento_clasificador import entrenamiento_clasificador
from utils.detectar_errores import tiene_carpeta_vacia

from tensorflow import keras
import numpy as np
import os
import wandb

## Ésta función se encarga de hacer el entrenamiento del autoencoder y el clasificador
## El parámetro <<entrenar_ae>> me dice si quiero entrenar o no el autoencoder
## El parámetro <<preprocesar>> me dice si quiero preprocesar o no los datos
def Entrenamiento(entrenar_ae = True, preprocesar = False):
    """
    Function that trains the autoencoder and the classifier

    Parameters
    ----------
    entrenar_ae: bool
        If its False it does not train the autoencoder and uses an older version. Defaults to True
    """
    ## Hago el llamado a <<ingesta_etiquetas_concat()>> obteniendo así los vectores de IDs de pacientes estables/inestables que voy a usar para entrenar/validar
    x_estables_train, x_inestables_train, x_estables_val, x_inestables_val = ingesta_etiquetas_concat()

    ## <<train>> va a tener el conjunto de IDs de pacientes que se van a usar para el entrenamiento
    train = np.concatenate((x_inestables_train, x_estables_train), axis = None)

    ## <<val_test>> va a tener el conjunto de IDs de pacientes que se van a usar para la validación
    val_test = np.concatenate((x_inestables_val, x_estables_val), axis=None)

    ## En caso que <<preprocesar>> sea True, llamo a la función de <<Preprocesamiento>> para que haga el preprocesado de los conjuntos de validación y entrenamiento
    if preprocesar:
        Preprocesamiento(train, val_test, upload_wandb = True)

    ## En caso que <<preprocesar>> sea False, obtengo los datos a partir del artefacto de wandb. O sea cargo los que ya existen y no los genero en el momento
    else:
        run = wandb.init(project = project_wandb, job_type = "load dataset")
        data_artifact = run.use_artifact(extra+str(preprocesamiento) + ':latest')
        data_artifact.checkout(path_dataset)
        run.finish()

    # se fija en cada muestra si tiene carpetas vacías en el preprocesado y
    # en ese caso manda a preprocesar denuevo dichas muestras

    ## Creo una lista <<train_reprocesar>> inicialmente vacía
    train_reprocesar = []

    ## Creo una lista <<valtest_reprocesar>> inicialmente vacía
    valtest_reprocesar = []

    ## Seteo la variable <<volver_a_preprocesar>> a False
    ## En caso de que <<volver_a_preprocesar>> sea True, tengo al menos una muestra para volver a preprocesar
    ## <<volver_a_preprocesar>> va a ser False sii no tengo ninguna muestra a reprocesar
    volver_a_preprocesar = False

    ## Iterar ID por ID (muestra por muestra) en conjunto de entrenamiento
    for muestra in train:

        ## En caso de que dicha muestra tenga la carpeta vacía, lo reproceso
        if tiene_carpeta_vacia(muestra, train, val_test, check_only_if_preprocessed_data_is_empty):

            ## Agrego dicha muestra (ID) a la lista de <<train_reprocesar>> que contiene las muestras de entrenamiento a reprocesar
            train_reprocesar.append(muestra)

            ## Seteo la bandera a true
            volver_a_preprocesar = True

    ## Iterar ID por ID (muestra por muestra) en conjunto de validación
    for muestra in val_test:
        
        ## En caso de que dicha muestra tenga la carpeta vacía, lo reproceso
        if tiene_carpeta_vacia(muestra, train, val_test, check_only_if_preprocessed_data_is_empty):

            ## Agrego dicha muestra (ID) a la lista de <<valtest_reprocesar>> que contiene las muestras de validación a reprocesar
            valtest_reprocesar.append(muestra)

            ## Seteo la bandera a true
            volver_a_preprocesar = True

    ## En caso de que tenga al menos alguna muestra que deba volver a preprocesar
    if volver_a_preprocesar:

        ## Hago el preprocesamiento correspondiente para aquellas muestras que separé que necesitan volver a preprocesarse
        Preprocesamiento(train_reprocesar, valtest_reprocesar, upload_wandb = True)

    ## En caso de que <<entrenar_ae>> esté en True de modo que quiero entrenar el autoencoder
    if entrenar_ae:
        print("Paso 2 - Entrenar autoencoder")

        ## <<config>> va a ser un diccionario que tiene parámetros necesarios para configurar el autoencoder
        config = {"giro x": girox, "giro z": giroz, "Escalado": escalado, "Loss": loss_name, "Actividad": act_ae, "Preprocesamiento": preprocesamiento, "lr": base_learning_rate}
        
        ## Comienza la sesión wandb cuyo objetivo va a ser guardar los parámetros del autoencoder luego del entrenamiento
        run = wandb.init(project = "SereTest-autoencoder", reinit = True, config = config, job_type = "train ae", name = autoencoder_name)

        ## Se hace un reset de los estados        
        keras.backend.clear_session()

        ae_train_save_model(dir_preprocessed_data_train, dir_preprocessed_data_test, model_path_ae, inDim, train, val_test, num_epochs, act_ae, batch_size, debug = True)
        trained_model_artifact=wandb.Artifact(autoencoder_name, type="model")
        trained_model_artifact.add_dir(model_path_ae)
        run.log_artifact(trained_model_artifact)
        wandb.run.finish()

    print( "Paso 3 - Entrenar clasificador")

    run = wandb.init(project="SereTest-autoencoder", reinit = True, job_type = "load ae")
    try:
        modelo_artifact = run.use_artifact(autoencoder_name + ':best')
    except:
        modelo_artifact = run.use_artifact(autoencoder_name + ':latest')

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

## Ejecución del programa principal
if __name__== '__main__':
    Entrenamiento(entrenar_ae = True, preprocesar = False)