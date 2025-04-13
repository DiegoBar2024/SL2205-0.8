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

## ------------------------------------- IMPORTACIÓN DEL MODELO ----------------------------------------

## Especifico la ruta en la cual yo voy a guardar el modelo de autoencoder
ruta_ae = 'C:\\Users\\diego/Dropbox/PROJECTS/SL2205/sereData/Modelos/autoencoder/ModeloAutoencoder/'

## Especifico el nombre del autoencoder
nombre_autoencoder = 'AutoencoderUCU_v2'

## Especifico la ruta en la cual yo voy a guardar las muestras reconstruidas
ruta_reconstruidas = 'C:/Yo/Tesis/sereData/sereData/Dataset/reconstrucciones_ae/S114_v2/'

## Cargo el modelo entrenado del autoencoder
modelo_autoencoder = ae_load_model(ruta_ae, nombre_autoencoder)

## En caso de que el directorio no exista
if not os.path.exists(ruta_reconstruidas):
    
    ## Creo el directorio correspondiente
    os.makedirs(ruta_reconstruidas)

## Obtengo el identificador del paciente para el cual voy a tener las muestras comprimidas
id_persona = 299

## ------------------------------------ RECONSTRUCCIÓN Y GUARDADO  -------------------------------------

## Parámetros del autoencoder
params = {'data_dir' :  dir_escalogramas_nuevo_test,
                        'dim' : inDim,
                        'batch_size' : batch_size,
                        'shuffle' : False,
                        'activities' : ['Caminando']}

## Lista de identificadores de los pacientes para el cual se van a computar las imagenes
lista_IDs = np.array([id_persona])

## Generación de datos de entrenamiento (va a ser un objeto de la clase DataGeneratorAuto)
## Le paso como argumento los parámetros de entrenamiento
generador_datos = DataGeneratorAuto(list_IDs = lista_IDs, **params)

## Obtengo el error del modelo sobre los datos provistos
modelo_autoencoder.evaluate(generador_datos)

## Obtengo las imagenes reconstruidas
## La variable <<img_reconstruida>> me va a almacenar un tensor tetradimensional de dimensiones (m, c, f, t) donde
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