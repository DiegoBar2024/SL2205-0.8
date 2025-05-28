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
from skimage.metrics import structural_similarity as ssim
from PIL import Image as im 

## ------------------------------------- IMPORTACIÓN DEL MODELO ----------------------------------------

## Obtengo el identificador del paciente para el cual voy a tener las muestras comprimidas
id_persona = 299

## Especifico la ruta donde están los escalogramas
ruta_escalogramas = "C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo"

## Especifico la ruta de donde voy a leer los escalogramas del paciente
ruta_escalogramas_paciente = "C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo/S{}/".format(id_persona)

## Especifico la ruta de la cual yo voy a importar el modelo de autoencoder
ruta_ae = 'C:/Yo/Tesis/sereData/sereData/Modelos/autoencoder/ModeloAutoencoder/'

## Especifico el nombre del autoencoder
nombre_autoencoder = 'AutoencoderUCU_nuevo'

## Especifico la ruta en la cual yo voy a guardar las muestras reconstruidas
ruta_reconstruidas = 'C:/Yo/Tesis/sereData/sereData/Dataset/reconstrucciones_ae/S{}/'.format(id_persona)

## Cargo el modelo entrenado del autoencoder
modelo_autoencoder = ae_load_model(ruta_ae, nombre_autoencoder)

## En caso de que el directorio no exista
if not os.path.exists(ruta_reconstruidas):
    
    ## Creo el directorio correspondiente
    os.makedirs(ruta_reconstruidas)

## Recuerdo que cada uno de los archivos se corresponde con un escalograma (tensor tridimensional)
archivos = [archivo for archivo in os.listdir(ruta_escalogramas_paciente) if archivo.endswith("npz")]

## ------------------------------------ RECONSTRUCCIÓN Y GUARDADO  -------------------------------------

## Inicializo una variable booleana diciendo si quiero hacer guardado o no
hacer_guardado = False

## Parámetros del autoencoder
## IMPORTANTE EL HECHO DE QUE LOS SEGMENTOS NO ESTÉN DESORDENADOS
params = {'data_dir' :  ruta_escalogramas,
                        'dim' : inDim,
                        'batch_size' : batch_size,
                        'shuffle' : False,
                        'activities' : ['Caminando']}

## Lista de identificadores de los pacientes para el cual se van a computar las imagenes
lista_IDs = np.array([id_persona])

## Generación de datos de entrenamiento (va a ser un objeto de la clase DataGeneratorAuto)
## Le paso como argumento los parámetros de entrenamiento
generador_datos = DataGeneratorAuto(list_IDs = lista_IDs, **params)

## Obtengo las imagenes reconstruidas
## La variable <<img_reconstruida>> me va a almacenar un tensor tetradimensional de dimensiones (m, c, f, t) donde
##  m: Me da la cantidad de segmentos (escalogramas) que se están analizando por el autoencoder
##  c: Me da la cantidad de canales del escalograma (en este caso c = 6)
##  f: Me da la cantidad de frecuencias sobre las cuales se encuentra calculado el escalograma
##  t: Me da la cantidad de muestras temporales sobre las cuales se calcula el escalogarma
img_reconstruida = modelo_autoencoder.predict(generador_datos)

## Construyo una lista donde voy a almacenar los errores cuadráticos medios de las reconstrucciones respecto a la entrada
errores_mse = []

## Construyo una lista donde voy a almacenar los indicadores SSIM correspondientes a las reconstrucciones respecto a la entrada
indicadores_ssim = []

## Itero para cada uno de los segmentos que tengo detectados
for i in range (img_reconstruida.shape[0]):

    ## Abro el archivo con la extensión .npz donde se encuentra el escalograma
    escalograma_original = np.load(ruta_escalogramas_paciente + archivos[i])['X']

    ## Obtengo el escalograma asociado al segmento actual
    escalograma_reconstruido = img_reconstruida[i,:,:,:]

    ## Obtengo el MSE (Error Cuadrático Medio) asociado a dicho escalograma
    error_mse = sklearn.metrics.mean_absolute_error(np.ndarray.flatten(escalograma_original), np.ndarray.flatten(escalograma_reconstruido))

    ## Agrego el valor calculado de MSE a la lista correspondiente
    errores_mse.append(error_mse)

    ## Obtengo el SSIM (Structural Similarity Index Measure) asociado a dicho escalograma
    ## Inicializo primero la variable en cero donde voy a almacenar el valor de dicho indicador
    indicador_ssim = 0

    ## Itero para cada uno de los canales correspondientes a la imagen (3 acelerómetros, 3 giroscopios)
    for j in range (img_reconstruida.shape[1]):

        ## Obtengo el escalograma original asociado al canal dado (imagen en blanco y negro)
        escalograma_orig_canal = im.fromarray(escalograma_original[j, :, :]).convert("L")

        ## Obtengo el escalograma reconstruído asociado al canal dado (imagen en blanco y negro)
        escalograma_reconstr_canal =  im.fromarray(escalograma_reconstruido[j, :, :]).convert("L")

        ## Hago la expansión de dimensiones al escalograma original para poder calcular el SSIM
        escalograma_orig_canal = tf.expand_dims(escalograma_orig_canal, axis = 2)

        ## Hago la expansión de dimensiones al escalograma reconstruído para poder calcular el SSIM
        escalograma_reconstr_canal = tf.expand_dims(escalograma_reconstr_canal, axis = 2)

        ## Obtengo el indicador SSIM comparando la imagen original con la reconstrucción
        ssim_canal = tf.image.ssim(escalograma_orig_canal, escalograma_reconstr_canal, max_val = 255, filter_size = 1)

        ## Sumo el valor de SSIM calculado a la variable donde se almacena
        indicador_ssim += float(ssim_canal)
    
    ## Obtengo el valor medio de los indicadores SSIM para todos los canales asociados al escalograma
    indicador_ssim /= img_reconstruida.shape[1]

    ## Almaceno el valor del indicador SSIM en la lista correspondiente
    indicadores_ssim.append(indicador_ssim)

    ## En caso de que quiera hacer un guardado
    if hacer_guardado:

        ## Guardado de los escalogramas reconstruídos en el directorio correspondiente
        np.savez_compressed(ruta_reconstruidas + 'Segmento_{}'.format(i), y = 0, X = escalograma_reconstruido)

## Creo una matriz bidimensional donde:
## La i-ésima fila hace referencia al i-ésimo segmento
## La primer columna indica el error MSE en la reconstrucción, la segunda columna indica el SSIM en la imagen reconstruída
indicadores = np.transpose(np.array([errores_mse, indicadores_ssim]))

## Impresión de estadísticas relacionadas al MSE
print("Estadísticas MSE\n   - Valor Medio MSE: {}\n   - Mediana MSE: {}\n   - Desviación estándar MSE: {}".format(np.mean(indicadores[:,0]), np.median(indicadores[:,0]), np.std(indicadores[:,0])))

## Impresión de estadísticas relacionadas al SSIM
print("Estadísticas SSIM\n   - Valor Medio SSIM: {}\n   - Mediana SSIM: {}\n   - Desviación estándar SSIM: {}".format(np.mean(indicadores[:,1]), np.median(indicadores[:,1]), np.std(indicadores[:,1])))