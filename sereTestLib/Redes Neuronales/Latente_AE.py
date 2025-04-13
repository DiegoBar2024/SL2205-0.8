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
from Extras_CLAS import *

## ------------------------------------- IMPORTACIÓN DEL MODELO ----------------------------------------

## Especifico la ruta en la cual yo voy a guardar el modelo de autoencoder
ruta_ae = 'C:\\Users\\diego/Dropbox/PROJECTS/SL2205/sereData/Modelos/autoencoder/ModeloAutoencoder/'

## Especifico el nombre del autoencoder
nombre_autoencoder = 'AutoencoderUCU'

## Especifico la ruta en la cual yo voy a guardar las muestras comprimidas
ruta_comprimidas = 'C:/Yo/Tesis/sereData/sereData/Dataset/latente_ae/S114/'

## En caso de que el directorio no exista
if not os.path.exists(ruta_comprimidas):
    
    ## Creo el directorio correspondiente
    os.makedirs(ruta_comprimidas)

## Cargo el modelo entrenado del autoencoder
modelo_autoencoder = ae_load_model(ruta_ae, nombre_autoencoder)

## ------------------------------------- COMPRESIÓN Y GUARDADO  ----------------------------------------

## Parámetros del autoencoder
params = {'data_dir' :  dir_escalogramas_nuevo_test,
                        'dim' : inDim,
                        'batch_size' : batch_size,
                        'shuffle' : False,
                        'activities' : ['Caminando']}

## Lista de identificadores de los pacientes para el cual se van a computar las imagenes
lista_IDs = np.array([114])

## Obtengo el espacio latente de todas las muestras correspondientes al paciente según mi modelo de autoencoder
## La variable <<espacio_latente>> va a ser una matriz de dos dimensiones dadas por (m, c) donde
##  m: Me da la cantidad de muestras que estoy comprimiendo
##  c: Me da la cantidad de características (feature) correspondientes al espacio latente
espacio_latente = patient_group_aelda(lista_IDs, modelo_autoencoder, layer_name = layer_name, **params)

## Guardo el espacio latente como una matriz bidimensional
## Las filas van a ser cada una de las muestras observadas mientras que las columnas corresponden a los features (formato tidy)
np.savez_compressed(ruta_comprimidas + 'S{}_latente'.format(id_persona), y = 0, X = espacio_latente)