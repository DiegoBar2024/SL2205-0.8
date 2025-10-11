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

## Función que toma como entrada los escalogramas originales de un paciente y a la salida retorna las reconstrucciones del autoencoder
def Reconstruccion(id_persona, nombre_autoencoder):

    ## Especifico la ruta de donde voy a leer los escalogramas del paciente
    ruta_escalogramas_paciente = ruta_escalogramas + "/S{}/".format(id_persona)

    ## Cargo el modelo entrenado del autoencoder
    modelo_autoencoder = ae_load_model(ruta_ae, nombre_autoencoder)

    ## Recuerdo que cada uno de los archivos se corresponde con un escalograma (tensor tridimensional)
    archivos = [archivo for archivo in os.listdir(ruta_escalogramas_paciente) if archivo.endswith("npz")]

    ## Creo una lista donde voy a almacenar todos los escalogramas originales
    escalogramas_originales = []

    ## Itero para cada uno de los segmentos presentes en el archivo
    for i in range (len(archivos)):

        ## Abro el archivo con la extensión .npz donde se encuentra el escalograma
        escalograma_original = np.load(ruta_escalogramas_paciente + archivos[i])['X']

        ## Agrego el escalograma a la lista de escalogramas
        escalogramas_originales.append(escalograma_original)
    
    ## Hago la traducción de array a vector numpy
    escalogramas_originales = np.array(escalogramas_originales)

    ## Obtengo las imagenes reconstruidas
    ## La variable <<img_reconstruida>> me va a almacenar un tensor tetradimensional de dimensiones (m, c, f, t) donde
    ##  m: Me da la cantidad de segmentos (escalogramas) que se están analizando por el autoencoder
    ##  c: Me da la cantidad de canales del escalograma (en este caso c = 6)
    ##  f: Me da la cantidad de frecuencias sobre las cuales se encuentra calculado el escalograma
    ##  t: Me da la cantidad de muestras temporales sobre las cuales se calcula el escalogarma
    escalogramas_reconstruidos = modelo_autoencoder.predict(escalogramas_originales)

    ## Retorno la imagen reconstruída, la lista de archivos y las rutas correspondientes
    return escalogramas_reconstruidos, escalogramas_originales, ruta_escalogramas_paciente, archivos

## Defino una función que realice el guardado de los escalogramas como matrices
def Guardado(img_reconstruida, id_persona):

    ## Especifico la ruta en la cual yo voy a guardar las muestras reconstruidas
    ruta_reconstruidas_paciente = ruta_reconstruidas + '/S{}/'.format(id_persona)

    ## En caso de que el directorio no exista
    if not os.path.exists(ruta_reconstruidas_paciente):
        
        ## Creo el directorio correspondiente
        os.makedirs(ruta_reconstruidas_paciente)

    ## Itero para cada uno de los segmentos que tengo detectados
    for i in range (img_reconstruida.shape[0]):

        ## Obtengo el escalograma asociado al segmento actual
        escalograma_reconstruido = img_reconstruida[i,:,:,:]
        
        ## Guardado de los escalogramas reconstruídos en el directorio correspondiente
        np.savez_compressed(ruta_reconstruidas_paciente + 'Segmento_{}'.format(i), y = 0, X = escalograma_reconstruido)

## Construyo una función que me evalúe las métricas de error entre la entrada y la salida
def ErrorReconstruccion(img_reconstruida, ruta_escalogramas_paciente, archivos):

    ## Construyo una función que me haga el cálculo de errores de aproximación entre una entrada y la salida
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

    ## Creo una matriz bidimensional donde:
    ## La i-ésima fila hace referencia al i-ésimo segmento
    ## La primer columna indica el error MSE en la reconstrucción, la segunda columna indica el SSIM en la imagen reconstruída
    indicadores = np.transpose(np.array([errores_mse, indicadores_ssim]))

    ## Retorno los indicadores de error calculados
    return indicadores

## Creo una función que cargue los tensores reconstruídos a la salida del autoencoder y los guarde como imágenes
def ConversionImagenes(id_persona):

    ## La idea es poder guardar todos los escalogramas generados como imágenes con extensión .png en una carpeta aparte
    ## Especifico la ruta de donde voy a leer los escalogramas
    ruta_lectura = ruta_reconstruidas + '/S{}/'.format(id_persona)

    ## Especifico la ruta de donde voy a guardar los escalogramas
    ruta_guardado = ruta_img_reconstruidas + "/S{}/".format(id_persona)

    ## En caso de que el directorio no exista
    if not os.path.exists(ruta_guardado):

        ## Creo el directorio correspondiente
        os.makedirs(ruta_guardado)

    ## Obtengo todos los archivos presentes en la ruta anterior
    ## Recuerdo que cada uno de los archivos se corresponde con un escalograma (tensor tridimensional)
    archivos = [archivo for archivo in os.listdir(ruta_lectura) if archivo.endswith("npz")]

    ## Itero para todos los archivos presentes en el directorio
    for i in range (len(archivos)):

        ## Abro el archivo con la extensión .npz donde se encuentra el escalograma
        archivo_escalograma = np.load(ruta_lectura + archivos[i])

        ## Almaceno el escalograma correspondiente en la variable <<escalograma>>
        escalograma = archivo_escalograma['X']

        ## Hago la traducción de los escalogramas a imágenes
        imagen_AC_x = im.fromarray(escalograma[0]).convert("L")
        imagen_AC_y = im.fromarray(escalograma[1]).convert("L")
        imagen_AC_z = im.fromarray(escalograma[2]).convert("L")
        imagen_GY_x = im.fromarray(escalograma[3]).convert("L")
        imagen_GY_y = im.fromarray(escalograma[4]).convert("L")
        imagen_GY_z = im.fromarray(escalograma[5]).convert("L")

        ## Hago el guardado de los escalogramas como imagenes en la ruta correspondiente
        imagen_AC_x.save(ruta_guardado + "3S{}s{}_ACx.png".format(id_persona, i))
        imagen_AC_y.save(ruta_guardado + "3S{}s{}_ACy.png".format(id_persona, i))
        imagen_AC_z.save(ruta_guardado + "3S{}s{}_ACz.png".format(id_persona, i))
        imagen_GY_x.save(ruta_guardado + "3S{}s{}_GYx.png".format(id_persona, i))
        imagen_GY_y.save(ruta_guardado + "3S{}s{}_GYy.png".format(id_persona, i))
        imagen_GY_z.save(ruta_guardado + "3S{}s{}_GYz.png".format(id_persona, i))

## Ejecución principal del programa
if __name__== '__main__':

    ## Ejecución de la rutina de reconstruccion
    escalogramas_reconstruidos, escalogramas_originales, ruta_escalogramas_paciente, archivos = Reconstruccion(302, "AutoencoderUCU_nuevo")