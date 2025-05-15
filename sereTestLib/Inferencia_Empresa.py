## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/clasificador')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Redes Neuronales')
from parameters import *
from Modelo_AE import *
import os
from Modelo_AE import *
from Evaluar_AE import *

## ------------------------------------ INFERENCIA DE LAS MUESTRAS -------------------------------------

def Inferencia(sample_id, result: results = results(), plot=False):
    """
    Function that classify a sample as stable or unstable.
    Parameters:
    sample id: int
    plot: bool
        If its true it plots original and reconstructed scalograms. Defaults to False
    """

    ## Recuerdo que result es una instancia de la clase results() la cual se encuentra especificada en los parámetros
    ## Asigno como atributo del objeto resultado el identificador del paciente correspondiente
    result.sample_id = sample_id

    ## Asigno como atributo del objeto resultado la característica de si tengo una muestra larga o corta
    result.long_sample = long_sample

    ## Especifico la ruta de donde yo voy a cargar el modelo del autoencoder
    ruta_ae = 'C:/Yo/Tesis/sereData/sereData/Modelos/autoencoder/Augmented_Caminando_100_mse_dense_base_classificator/'

    ## Especifico la ruta de donde yo voy a guardar el modelo del clasificador
    ruta_clf = 'C:/Yo/Tesis/sereData/sereData/Modelos/clasificador/lda_Augmented/'

    ## Especifico el nombre del modelo del autoencoder
    nombre_autoencoder = 'Augmented_Caminando_100_mse_dense_base_classificator'

    ## Cargo el modelo del autoencoder especificado en la ruta anterior
    modelo_autoencoder = ae_load_model(ruta_ae, nombre_autoencoder)

    ## En caso de que la ruta <<results_path_day>> no exista
    if not os.path.exists(results_path_day):

        ## Hago la creación de dicha ruta
        os.makedirs(results_path_day)
    
    ## En caso de que el directorio de salida no exista
    if not os.path.exists(results_path):

        # Creo el directorio de salida de datos
        os.makedirs(results_path)

    ## Obtengo el nombre del autoencoder
    result.ae_name = nombre_autoencoder

    ## Especifico el tipo de clasificador
    clasificador = "lda"

    ## Especifico el nombre del clasificador
    clasificador_name = 'lda_Augmented'

    ## Obtengo el nombre del clasificador con su correspondiente extensión .joblib
    clf_name = clasificador_name + '.joblib'

    ## Obtengo el nombre del clasificador
    result.clf_name = clasificador_name

    ## Obtengo la representación latente de las características asociadas a cada escalograma tridimensional
    ## Luego de hacer eso hago la inferencia correspondiente a la muestra con el clasificador entrenado
    evaluar_aelda(clasificador_name, nombre_autoencoder, ruta_clf, clasificador, modelo_autoencoder, sample_id, result, clf_model_file = clf_name)

    ## Retorno el resultado de la inferencia como un objeto
    return result

## Rutina principal del programa
if __name__== '__main__':

    ## Especifico el identificador numérico del paciente
    i = 216

    ## Ejecuto la inferencia del clasificador
    result = Inferencia(i)