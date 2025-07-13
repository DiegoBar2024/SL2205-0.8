## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/clasificador')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from parameters import *
from Modelo_AE import *
from Modelo_CLAS import *
from Etiquetado import *
from Modelo_CLAS import *
from Extras_CLAS import *
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
ruta_escalogramas = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_giros_tot'

## Itero para cada uno de los identificadores de los pacientes. Hago la generación de escalogramas para cada paciente
for id_persona in ids_existentes[ids_existentes > 73]:

    ## Creo un bloque try catch en caso de que ocurra algún error en el procesamiento
    try:

        ## Especifico la ruta en donde voy a guardar las muestras comprimidas
        ruta_comprimidas = "C:/Yo/Tesis/sereData/sereData/Dataset/latente_ae_giros_tot/S{}/".format(id_persona)

        ## En caso de que el directorio no exista
        if not os.path.exists(ruta_comprimidas):
            
            ## Creo el directorio correspondiente
            os.makedirs(ruta_comprimidas)

        ## Parámetros de compresión del autoencoder
        params = {'data_dir' :  ruta_escalogramas,
                                'dim' : inDim,
                                'batch_size' : batch_size,
                                'shuffle' : False,
                                'activities' : ['Caminando']}

        ## Lista de identificadores de los pacientes para el cual se van a computar las imagenes
        lista_IDs = np.array([id_persona])

        ## Obtengo el espacio latente de todas las muestras correspondientes al paciente según mi modelo de autoencoder
        ## La variable <<espacio_latente>> va a ser una matriz de dos dimensiones dadas por (m, c) donde
        ##  m: Me da la cantidad de muestras que estoy comprimiendo
        ##  c: Me da la cantidad de características (feature) correspondientes al espacio latente
        espacio_latente = patient_group_aelda(lista_IDs, modelo_autoencoder, layer_name = layer_name, **params)

        ## Guardo el espacio latente como una matriz bidimensional
        ## Las filas van a ser cada una de las muestras observadas mientras que las columnas corresponden a los features (formato tidy)
        np.savez_compressed(ruta_comprimidas + 'S{}_latente'.format(id_persona), y = 0, X = espacio_latente)

        ## Impresión en pantalla avisando el identificador del paciente que está siendo procesado
        print("ID del paciente que está siendo procesado: {}".format(id_persona))
    
    ## En caso de que ocurra un error en el procesamiento
    except:

        ## Sigo con el paciente que sigue
        continue