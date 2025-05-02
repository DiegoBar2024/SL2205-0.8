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
ruta_ae = 'C:/Yo/Tesis/sereData/sereData/Modelos/autoencoder/ModeloAutoencoder/'

## Especifico el nombre del autoencoder que voy a cargar
nombre_autoencoder = 'AutoencoderUCU_gp'

## Especifico el nombre con el que voy a guardar el nuevo autoencoder guardado
autoencoder_nuevo = 'AutoencoderUCU_gp'

## Cargo el modelo entrenado del autoencoder
modelo_autoencoder = ae_load_model(ruta_ae, nombre_autoencoder)

## ------------------------------------ ENTRENAMIENTO Y GUARDADO  --------------------------------------

## Especifico la ruta en la cual van a estar los escalogramas
ruta_escalogramas = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo_gp'

## EL OBJETIVO ES PODER ENTRENAR UN MODELO DE AUTOENCODER PREVIAMENTE EXISTENTE PARA NO TENER QUE ENTRENARLO DE CERO
## Parámetros de entrenamiento de autoencoder
params = {'data_dir' :  ruta_escalogramas,
                        'dim' : inDim,
                        'batch_size' : batch_size,
                        'shuffle' : False,
                        'activities' : ['Caminando']}

## Lista de identificadores de los pacientes para el cual se van a computar las imagenes
lista_IDs = np.array([120, 122, 126, 129, 143, 151, 163])

## Generación de datos de entrenamiento (va a ser un objeto de la clase DataGeneratorAuto)
generador_entrenamiento = DataGeneratorAuto(list_IDs = lista_IDs, **params)

## Hago la compilación del modelo para poder volverlo a entrenar especificando el learning rate y la función de costo que va a utilizar el optimizador
modelo_autoencoder.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = base_learning_rate), loss = loss_name)

# Entrenamiento del modelo de autoencoder especificando todos los parámetros correspondientes
history = modelo_autoencoder.fit(x = generador_entrenamiento, epochs = 1, verbose = 1)

# Genero el nombre de archivo del autoencoder y lo guardo
# Ésto es lo que debo usar como entrada al clasificador
modelo_autoencoder.save(ruta_ae + autoencoder_nuevo + '.h5')