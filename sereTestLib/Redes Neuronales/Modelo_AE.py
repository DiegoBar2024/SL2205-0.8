import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/autoencoder')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
from ae_train_save_model import *
from parameters import *

from tabnanny import verbose
from numpy.core.fromnumeric import size
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
import os
from GeneradorDatos import *

from skimage.metrics import structural_similarity as ssim

## SSIM LOSS
def ssim_loss(y_true, y_pred):
    loss = tf.reduce_mean(1-tf.image.ssim(y_true, y_pred,max_val=255))
    return loss

## MS SSIM LOSS
def ms_ssim(y_true, y_pred):

    ## Calculo la función de error correspondiente
    loss = tf.reduce_mean(1 - tf.image.ssim_multiscale(y_true, y_pred,max_val=255,filter_size=3))

    return loss

## SSIM METRIC
def ssim_metric(y_true, y_pred):

    ## Calculo la función de error correspondiente
    loss = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val = 255))
    ##loss = ssim(y_pred, y_true, channel_axis = 1, data_range = 255)

    return loss

## Función que genera el modelo del autoencoder y lo entrena según los parámetros de entrada
def ae_train_save_model(autoencoder_name, path_to_scalograms_train = dir_escalogramas_nuevo_train, path_to_scalograms_val = dir_escalogramas_nuevo_test , model_path = model_path_ae, input_dimension = inDim, list_of_samples_number_train = [], 
                        list_of_samples_number_validation = [], number_epochs = num_epochs, activities = act_ae, batch_size = batch_size, latent_dimension = latent_dimension, debug = False):
    """
        Function that generates the autoencoder model base on training datagroup. Validates it and saves it.
        NOTE: If more than one activity is given, are trated together
    Parameters:
    ----------
    autoencoder_name: str
        Autoencoder name
    path_to_scalograms_train: str
        Path to training segments
    path_to_scalograms_val: str
        Path to validation segments
    model_path: str
        Path to save the model
    input_dimension: list 
        Scalogram dimention
    list_of_samples_number_train: list
    list_of_samples_number_validation: list
    number_epochs: int
    activities list: 
        List of activities to train and validate the model.
    batch_size: int
    latent_dimension: int
    debug: bool
        Defaults to False.
    """

    # Especifico la extensión del archivo del modelo
    model_extension = '.h5'

    ## En caso de que el nombre de la función de costo no sea "MS_SSIM" (por defecto está en MSE) 
    if loss_name != 'ms_ssim':

        ## Construyo el modelo de autoencoder correspondiente
        ## <<dim_input>> contiene las dimensiones del tensor de entrada al autoencoder (largo, ancho, profundidad)
        ## <<latentDim>> contiene las dimensiones del espacio codificado del autoencoder (en principio 256)
        autoencoder = ae_model_builder(dim_input = input_dimension, latentDim = latent_dimension)
    
    ## En caso de que la función de costo sea "MS_SSIM" ejecuto lo siguiente
    else:

        basic_path = modo_ae + str(latent_dimension) + "".join(act_ae) + str(num_epochs)
        autoencoder = keras.models.load_model(model_path + autoencoder_name+'.h5')

        #autoencoder=keras.models.load_model(model_path + autoencoder_name, custom_objects={"ssim_loss":ssim_loss})
        autoencoder.trainable = True
        autoencoder.compile(optimizer = tf.keras.optimizers.Adam(lr= 0.0001), loss = ms_ssim)

    ## En caso de que la bandera <<debug>> se encuentre seteada en True
    if debug:

        ## Imprimo un resumen del autoencoder correspondiente
        autoencoder.summary()

    ## En caso de que el directorio de salida no exista
    if not os.path.exists(model_path):

        ## Creo dicho directorio
        os.makedirs(model_path)

    ## Parámetros de entrenamiento del autoencoder
    ## <<data_dir>> me va a decir la ruta donde están los escalogramas que van a ser usados para la validación del autoencoder
    ## <<dim>> va a contener una tupla de 3 elementos con la dimensión de los escalogramas de entrada
    ## <<batch_size>> va a contener el tamaño del batch (cantidad de muestras de entrenamiento) que voy a usar para entrenar
    ## <<activities>> va a ser una lista de strings donde cada cadena es la actividad que voy a analizar
    paramsT = {'data_dir' : path_to_scalograms_train,
                            'dim' : input_dimension,
                            'batch_size' : batch_size,
                            'shuffle' : True,
                            'activities' : activities}

    ## Parámetros de validación del autoencoder
    ## <<data_dir>> me va a decir la ruta donde están los escalogramas que van a ser usados para la validación del autoencoder
    ## <<dim>> va a contener una tupla de 3 elementos con la dimensión de los escalogramas de entrada
    ## <<batch_size>> va a contener el tamaño del batch (cantidad de muestras de entrenamiento) que voy a usar para entrenar
    ## <<activities>> va a ser una lista de strings donde cada cadena es la actividad que voy a analizar
    paramsV = {'data_dir' : path_to_scalograms_val,
                            'dim' : input_dimension,
                            'batch_size' : batch_size,
                            'shuffle' : False,
                            'activities' : activities}

    ## Generación de datos de entrenamiento (va a ser un objeto de la clase DataGeneratorAuto)
    ## Le paso como argumento los parámetros de entrenamiento
    training_generator = DataGeneratorAuto(list_IDs = list_of_samples_number_train, **paramsT)
    
    ## Generación de datos de validación (va a ser un objeto de la clase DataGeneratorAuto)
    ## Le paso como argumento los parámetros de validación
    validation_generator = DataGeneratorAuto(list_IDs = list_of_samples_number_validation, **paramsV)

    # Entrenamiento del modelo de autoencoder especificando todos los parámetros correspondientes
    history = autoencoder.fit(x = training_generator,
                    validation_data = validation_generator,
                    epochs = number_epochs,
                    verbose = 1)

    # Genero el nombre de archivo del autoencoder y lo guardo
    # Ésto es lo que debo usar como entrada al clasificador
    autoencoder.save(model_path + autoencoder_name + model_extension)

    ## Hago la limpieza de la sesión de Keras para poder reducir la memoria y problemas de bucle
    del autoencoder
    keras.backend.clear_session()

## Función la cual me construye el modelo del autoencoder
## <<dim_input>> me da las dimensiones de la entrada del autoencoder. Por defecto tengo dim_input = (128, 600, 6) lo cual quiere decir que mi entrada es un tensor tridimensional de 128 x 600 y profundidad 6. Es decir un tensor de dimensiones 128 x 600 x 6
## <<filters>> es una tupla cuyos elementos son la cantidad de filtros que tienen las distintas capas del encoder. El decoder tendrá los mismos filtros pero en orden inverso
## <<latentDim>> me va a dar la dimensión del espacio latente, es decir me va a dar la representación comprimida de la entrada como un vector de 256 elementos o 256 características
def ae_model_builder(dim_input = inDim, filters = (32, 16, 8), latentDim = 256):
    """
    Function that builds the autoencoder model
    Parameters:
    ----------
    dim_input: list
    filters: list
        the number of output filters in each layer.
    latentDim: int
        Dimension of the latent vector

    Returns:
    --------
    model:
        Autoencoder model

    """

    ## Instrucción de inicialización
    init = keras.initializers.he_normal()

    ## Defino cuál va a ser la entrada del autoencoder con sus respectivas dimensiones
    IN = keras.Input(shape = dim_input, name = 'scalograms')
    x = IN

    ## ENCODER
    ## La función es tomar el tensor tridimensional de entrada y alcanzar una representación comprimida de 256 características en el espacio latente
    ## El encoder va a estar compuesto por 3 pares de capas convolucional y pooling en serie

    ## Itero sobre la tupla donde tengo los números de filtros
    for f in filters:
        
        ## Se define una CAPA CONVOLUCIONAL
        ## El parámetro <<filters>> toma la cantidad de filtros que tiene ésta capac convolucional
        ## El parámetro <<kernel_size>> me da las dimensiones espaciales del kernel que se va a utilizar en dicha capa.
        ##    Algo muy importante es que al colocar <<kernel_size = (n,n)>> yo especifico las DIMENSIONES ESPACIALES (largo y ancho) del kernel, pero debo recordar que la PROFUNDIDAD DE CADA KERNEL DEBE SER IGUAL AL NÚMERO DE FEATURE MAPS que tiene la capa convolucional anterior (muy importante!)
        ## El parámetro <<strides>> me dice "de a cuanto va saltando" el kernel a medida que se desliza  través de la entrada. Al colocar <<strides=1>> ésto implica que va de uno en uno lo cual es lo intutivo y no hay compresión
        ## El parámetro <<activation>> me especifica el tipo de función de activación no lineal que se le aplica al bloque de salida. Algo importante es que la aplicación de la función de activación no lineal NO ME CAMBIA LAS DIMENSIONES DE LA SALIDA DE LA CAPA.
        ## El parámetro <<padding>> me da la opción de hacer inserción de ceros en la entrada para conservar las dimensiones ESPACIALES (no profundidad!) de la salida.
        ##    Entonces al colocar <<padding='same'>> yo estoy diciendo que las dimensiones espaciales del tensor tridimensional de salida son IGUALES A LAS DE LA ENTRADA, SIN EMBARGO LA PROFUNDIDAD DEL TENSOR TRIDIMENSIONAL DE SALIDA VA A SER IGUAL AL NÚMERO DE FEATURE MAPS QUE TENGA ESA CAPA (o sea al numero de filtros)
        x = layers.Conv2D(filters = f, kernel_size = (3, 3), strides = 1, activation = "elu", padding = 'same', data_format = 'channels_first')(x)

        ## Se define una CAPA POOLING
        ## El parámetro <<pool_size=(2,2)>> me va a dar la VENTANA sobre la cual se va a deslizar la ventana que tomará el máximo (por defecto se asume striding unitario)
        ## Algo importante es que la capa pooling SÍ ME VA A CAMBIAR LAS DIMENSIONES ESPACIALES DEL TENSOR TRIDIMENSIONAL DE ENTRADA, PERO NO ME CAMBIA LA PROFUNDIDAD (el número de feature maps sigue siendo el mismo a la salida que a la entrada)
        ## Esencialmente lo que estoy haciendo acá es COMPRIMIR el tensor tridimensional de entrada a lo largo de sus dimensiones espaciales quedandome con el máximo al deslizar una ventana de 2 x 2
        x = layers.MaxPooling2D(pool_size=(2, 2), padding = 'same', data_format = 'channels_first')(x)

    ## En <<volumeSize>> almaceno la CANTIDAD DE ELEMENTOS que tiene el tensor tridimensional x
    volumeSize = int_shape(x)

    ## Aplicando el metodo <<Flatten>> lo que hago es traducir x de un tensor tridimensional a un vector unidimensional
    x = layers.Flatten()(x)

    ## Se crea una dense layer la cual tenga una cantidad <<latentDim>> de neuronas la cual vendría a ser la versión comprimida de x
    latent = layers.Dense(latentDim, name = 'Dense_encoder')(x)

    ## DECODER
    ## Creo un decoder el cual va a aceptar la salida del encoder como sus entradas
    ## De éste modo lo que voy a hacer es armar el INVERSO del encoder

    ## Primero creo una dense layer la cual tenga una cantidad de neuronas igual al numero de elementos de x antes de comprimirlo a las 256 características
    x = layers.Dense(np.prod(volumeSize[1:]), name = 'Dense_decoder')(latent)

    ## Modifico la forma de x para convertirlo en un tensor tridimensional con las mismas dimensiones que antes
    x = layers.Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

    ## Itero para todos los filtros en la lista de filtros sólo que en orden INVERSO
    for f in filters[::-1]:

        ## Se define el inverso de la capa CONVOLUCIONAL
        ## En lugar de aplicar una convolución como en el encoder, aquí se aplica la operación inversa a la convolución siendo ésta la convolución con el kernel transpuesto
        ## El resto de los parámetros son los mismos que los definidos en la correspondiente capa convolucional
        x = layers.Conv2DTranspose(filters = f, kernel_size = (3, 3), strides = 1, activation = "elu", padding = 'same', data_format = 'channels_first')(x)
        
        ## Se define la operación inversa al submuestreo realizado en la capa POOLING del encoder
        ## Se usa la función UpSampling2D para duplicar las dimensiones espaciales del tensor tridimensional de entrada, aunque la profundidad dependerá del número de feature maps que tenga
        x = layers.UpSampling2D((2, 2), data_format = 'channels_first')(x)

    ## Se aplica una única convolución con la transpuesta para volver a obtener la profundidad original
    x = layers.Conv2DTranspose(depth, (3, 3), padding = "same", kernel_initializer = init, data_format = 'channels_first')(x)

    ## Especifico función de activación lineal
    decoded = layers.Activation("linear")(x)
    
    ## Se declara el modelo donde la <<input>> va a ser el tensor tridimensional de entrada IN y la <<output>> va a ser la salida decodificada
    model = keras.Model(inputs = IN, outputs = decoded)

    ## En caso de que la función de pérdida sea la "SSIM_LOSS"
    if loss_name == "ssim_loss":

        ## Hago la compilación del modelo
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = base_learning_rate), loss = ssim_loss, metrics = [ssim_metric])
    
    ## En caso de que la función de pérdida no sea la "SSIM LOSS"
    else:

        ## Hago la compilación del modelo
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = base_learning_rate), loss = loss_name)
    
    ## Retorno el modelo de autoencoder resultante
    return model

## Traigo la función <<int_shape>> proveniente del GitHub de Keras
## Actualmente está deprecado, así que la tengo que escribir manualmente
def int_shape(x):
    try:
        shape = x.shape
        if not isinstance(shape, tuple):
            shape = tuple(shape.as_list())
        return shape
    except ValueError:
        return None

## Función que me permite cargar el modelo del autoencoder
def ae_load_model(modelo_dir, autoencoder_name):

    ## En caso de que la función de costo sea SSIM
    if loss_name == "ssim_loss":

        ## Cargo el modelo del autoencoder correspondiente
        modelo_autoencoder = keras.models.load_model(modelo_dir + autoencoder_name +'.h5', custom_objects = {"ssim_loss": ssim_loss})
    
    ## En caso de que la función de costo sea MS SSIM
    elif loss_name == "ms_ssim":

        ## Cargo el modelo del autoencoder correspondiente
        modelo_autoencoder = keras.models.load_model(modelo_dir + autoencoder_name +'.h5', custom_objects = {"ms_ssim": ms_ssim})
    
    ## En caso que tenga otra función de costo
    else:

        ## Cargo el modelo de autoencoder correspondiente
        modelo_autoencoder = keras.models.load_model(modelo_dir + autoencoder_name +'.h5', custom_objects = {"ssim_metric": ssim_metric, 'mse' : 'mse'})
    
    ## Retorno el modelo del autoencoder
    return(modelo_autoencoder)