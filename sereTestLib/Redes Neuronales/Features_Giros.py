## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from parameters import *
from Etiquetado import *
from Modelo_AE import *
from LecturaDatosPacientes import *

## ------------------------------------- IMPORTACIÓN DEL MODELO ----------------------------------------

## Especifico la ruta de la cual yo voy a importar el modelo de autoencoder
ruta_ae = 'C:/Yo/Tesis/sereData/sereData/Modelos/autoencoder/ModeloAutoencoder/'

## Especifico el nombre del autoencoder
nombre_autoencoder = 'AutoencoderUCU_nuevo'

## Cargo el modelo entrenado del autoencoder
modelo_autoencoder = ae_load_model(ruta_ae, nombre_autoencoder)

## ------------------------------------- COMPRESIÓN Y GUARDADO  ----------------------------------------

## Hago la lectura de los datos generales de los pacientes presentes en la base de datos
pacientes, ids_existentes = LecturaDatosPacientes()

## Especifico la ruta en la cual se encuentran los escalogramas que voy a comprimir
ruta_escalogramas = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_sin_giros'

## Itero para cada uno de los identificadores de los pacientes. Hago la generación de escalogramas para cada paciente
for id_persona in ids_existentes:

    ## Creo un bloque try catch en caso de que ocurra algún error en el procesamiento
    try:
        
        ## Obtengo el conjunto de tramos de marcha sin giros detectados para el paciente
        tramos = os.listdir(ruta_escalogramas + '/S{}'.format(id_persona))

        ## Itero para cada uno de los tramos listados anteriormente
        for i in range (len(tramos)):
    
            ## Especifico la ruta en donde voy a guardar las muestras comprimidas
            ruta_comprimidas = "C:/Yo/Tesis/sereData/sereData/Dataset/latente_sin_giros/S{}/{}/".format(id_persona, tramos[i])

            ## En caso de que el directorio no exista
            if not os.path.exists(ruta_comprimidas):
                
                ## Creo el directorio correspondiente
                os.makedirs(ruta_comprimidas)

            ## Obtengo el conjunto de segmentos correspondientes a ese paciente a ese tramo
            segmentos = sorted(os.listdir('C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_sin_giros/S{}/{}'.format(id_persona, tramos[i])), key = len)
            
            ## Construyo un vector vacío en el cual voy a almacenar los espacios latentes como matriz de ceros
            espacio_latente = np.zeros((len(segmentos), 256))

            ## Itero para cada uno de los escalogramas del tramo
            for j in range (len(segmentos)):

                ## Abro el archivo .npz correspondiente
                escalograma_segmento = np.load(ruta_escalogramas + '/S{}/Tramo{}/{}'.format(id_persona, i, segmentos[j]))['X']

                ## Se crea un modelo en base al autoencoder que toma como entrada la entrada al autoencoder y que toma como la salida las 256 características del autoencoder con los parámetros que ya tiene
                intermediate_layer_model = keras.Model(inputs = modelo_autoencoder.input,
                                        outputs = modelo_autoencoder.get_layer('Dense_encoder').output)

                ## Se realiza entonces la predicción del autoencoder a los datos del generador dando como resultado para cada muestra las 256 características a que correspondan a la salida
                intermediate = intermediate_layer_model.predict(np.reshape(escalograma_segmento, (1, 6, 128, 800)))

                ## Lo agrego a la matriz de los espacios latentes
                espacio_latente[j, :] = intermediate

            ## Guardo el espacio latente como una matriz bidimensional
            ## Las filas van a ser cada una de las muestras observadas mientras que las columnas corresponden a los features (formato tidy)
            np.savez_compressed(ruta_comprimidas + 'S{}_latente'.format(id_persona), y = 0, X = espacio_latente)

        ## Impresión en pantalla avisando el identificador del paciente que está siendo procesado
        print("ID del paciente que está siendo procesado: {}".format(id_persona))
    
    ## En caso de que ocurra un error en el procesamiento
    except:

        ## Sigo con el paciente que sigue
        continue