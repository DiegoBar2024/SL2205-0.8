## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
import wandb
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/clasificador')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
from parameters import *
from Modelo_AE import *
from Modelo_CLAS import *
from Etiquetado import *
from Modelo_CLAS import *

## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

## Especifico la ruta en la cual yo voy a guardar el modelo de autoencoder
ruta_ae = 'C:\\Users\\diego/Dropbox/PROJECTS/SL2205/sereData/Modelos/autoencoder/ModeloAutoencoder/'

## Especifico el nombre del autoencoder
nombre_autoencoder = 'AutoencoderUCU'

## Especifico la ruta en la cual yo voy a guardar las muestras reconstruidas
ruta_reconstruidas = 'C:/Yo/Tesis/sereData/sereData/Dataset/reconstrucciones_ae/S114/'

## En caso de que el directorio no exista
if not os.path.exists(ruta_reconstruidas):
    
    ## Creo el directorio correspondiente
    os.makedirs(ruta_reconstruidas)

## Cargo el modelo entrenado del autoencoder
modelo_autoencoder = ae_load_model(ruta_ae, nombre_autoencoder)

## Parámetros del autoencoder
params = {'data_dir' :  dir_escalogramas_nuevo_test,
                        'dim' : inDim,
                        'batch_size' : batch_size,
                        'shuffle' : False,
                        'activities' : ['Caminando']}

## Lista de identificadores de los pacientes para el cual se van a computar las imagenes
lista_IDs = np.array([114])

## Generación de datos de entrenamiento (va a ser un objeto de la clase DataGeneratorAuto)
## Le paso como argumento los parámetros de entrenamiento
generador_datos = DataGeneratorAuto(list_IDs = lista_IDs, **params)

## Obtengo las imagenes reconstruidas
## La variable <<img_reconstruida>> me va a almacenar un tensor tetradimensional de dimensiones (m, c, f, t) dpnde
##  m: Me da la cantidad de segmentos (escalogramas) que se están analizando por el autoencoder
##  c: Me da la cantidad de canales del escalograma (en este caso c = 6)
##  f: Me da la cantidad de frecuencias sobre las cuales se encuentra calculado el escalograma
##  t: Me da la cantidad de muestras temporales sobre las cuales se calcula el escalogarma
img_reconstruida = modelo_autoencoder.predict(generador_datos)

## Itero para cada uno de los segmentos que tengo detectados
for i in range (img_reconstruida.shape[0]):

    ## Obtengo el escalograma asociado al segmento actual
    escalograma = img_reconstruida[i,:,:,:]

    ## Guardado de los escalogramas reconstruídos en el directorio correspondiente
    np.savez_compressed(ruta_reconstruidas + 'Segmento_{}'.format(i), y = 0, X = escalograma)