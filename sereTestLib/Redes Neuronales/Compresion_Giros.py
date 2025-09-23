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
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler

## Creo una función que me haga la compresión y guardado de los giros usando un autoencoder
def CompresionAutoencoder():

    ## ------------------------------------- IMPORTACIÓN DEL MODELO ----------------------------------------

    ## Especifico el nombre del autoencoder
    nombre_autoencoder = 'AutoencoderUCU_nuevo'

    ## Cargo el modelo entrenado del autoencoder
    modelo_autoencoder = ae_load_model(ruta_ae, nombre_autoencoder)

    ## ------------------------------------- COMPRESIÓN Y GUARDADO  ----------------------------------------

    ## Hago la lectura de los datos generales de los pacientes presentes en la base de datos
    pacientes, ids_existentes = LecturaDatosPacientes()

    ## Itero para cada uno de los identificadores de los pacientes. Hago la generación de escalogramas para cada paciente
    for id_persona in ids_existentes:

        ## Creo un bloque try catch en caso de que ocurra algún error en el procesamiento
        try:

            ## Especifico la ruta en donde voy a guardar las muestras comprimidas
            ruta_comprimidas = ruta_comprimidas_giros + "/S{}/".format(id_persona)

            ## En caso de que el directorio no exista
            if not os.path.exists(ruta_comprimidas):
                
                ## Creo el directorio correspondiente
                os.makedirs(ruta_comprimidas)

            ## Obtengo el conjunto de segmentos correspondientes a ese paciente a ese tramo
            segmentos = sorted(os.listdir(ruta_escalogramas_giros + '/S{}'.format(id_persona)), key = len)

            ## Construyo un vector vacío en el cual voy a almacenar los espacios latentes como matriz de ceros
            espacio_latente = np.zeros((len(segmentos), 256))

            ## Itero para cada uno de los escalogramas del tramo
            for j in range (len(segmentos)):

                ## Abro el archivo .npz correspondiente
                escalograma_giro = np.load(ruta_escalogramas + '/S{}/{}'.format(id_persona, segmentos[j]))['X']

                ## Se crea un modelo en base al autoencoder que toma como entrada la entrada al autoencoder y que toma como la salida las 256 características del autoencoder con los parámetros que ya tiene
                intermediate_layer_model = keras.Model(inputs = modelo_autoencoder.input,
                                        outputs = modelo_autoencoder.get_layer(layer_name).output)

                ## Se realiza entonces la predicción del autoencoder a los datos del generador dando como resultado para cada muestra las 256 características a que correspondan a la salida
                intermediate = intermediate_layer_model.predict(np.reshape(escalograma_giro, (1, 6, 128, 800)))

                ## Lo agrego a la matriz de los espacios latentes
                espacio_latente[j, :] = intermediate

            ## Guardo el espacio latente como una matriz bidimensional
            ## Las filas van a ser cada una de las muestras observadas mientras que las columnas corresponden a los escalograma_giros_scomprimir (formato tidy)
            np.savez_compressed(ruta_comprimidas + 'S{}_latente'.format(id_persona), y = 0, X = espacio_latente)

            ## Impresión en pantalla avisando el identificador del paciente que está siendo procesado
            print("ID del paciente que está siendo procesado: {}".format(id_persona))
        
        ## En caso de que ocurra un error en el procesamiento
        except:

            ## Sigo con el paciente que sigue
            continue

## Construyo una función la cual me genere y guarde de una matriz con los escalogramas DE GIROS SIN COMPRIMIR
## TODOS LOS ESCALOGRAMAS DE GIROS DE TODOS LOS PACIENTES VAN A ESTAR JUNTOS
def GenerarVectoresEscalogramas():

    ## Hago la lectura de los datos generales de los pacientes presentes en la base de datos
    pacientes, ids_existentes = LecturaDatosPacientes()

    ## Construyo un vector vacío en el cual voy a almacenar los escalogramas de giros aplanados
    ## Voy a almacenar los escalogramas de giros de todos los pacientes
    vectores_escalogramas = np.zeros((1, 6 * 128 * 800))

    ## Itero para cada uno de los identificadores de los pacientes. Hago la generación de escalogramas para cada paciente
    for id_persona in ids_existentes:

        ## Creo un bloque try catch en caso de que ocurra algún error en el procesamiento
        try:

            ## Obtengo el conjunto de giros correspondientes a ese paciente a ese tramo
            segmentos = sorted(os.listdir(ruta_escalogramas_giros + '/S{}'.format(id_persona)), key = len)

            ## Itero para cada uno de los escalogramas del tramo
            for j in range (len(segmentos)):

                ## Abro el archivo .npz correspondiente
                escalograma_giro = np.load(ruta_escalogramas_giros + '/S{}/{}'.format(id_persona, segmentos[j]))['X']

                ## Construyo un vector aplanado del escalograma del giro
                vector_escalograma = np.ndarray.flatten(escalograma_giro)

                ## Agrego el vector del giro a la matriz de vectores de giro
                vectores_escalogramas = np.concatenate((vectores_escalogramas, np.reshape(vector_escalograma, (1, vector_escalograma.shape[0]))), axis = 0)
    
            ## Imprimo mostrando el ID del paciente que fue procesado
            print("ID del paciente que está siendo procesado: {}".format(id_persona))

        ## En caso de que ocurra un error en el procesamiento
        except:

            ## Sigo con el paciente que sigue
            continue
    
    ## Elimino el dummy vector de la matriz de vectores de escalogramas de giros
    vectores_escalogramas = vectores_escalogramas[1:,:]

    ## Guardo todos los escalogramas juntas en un único archivo
    np.savez_compressed(ruta_giros_juntos_sin_comprimir + "/GirosJuntos", X = vectores_escalogramas)

## Ejecución principal del programa
if __name__== '__main__':

    ## Creo una bandera que me diga si quiero o no guardar giros
    guardarGiros = False

    ## En caso de que quiera guardar giros
    if guardarGiros:

        ## Obtengo los vectores de escalogramas de giros sin comprimir y los guardo
        GenerarVectoresEscalogramas()

    ## En caso de que quiera procesar giros sin comprimir
    else:

        ## Abro el archivo .npz correspondiente
        escalograma_giros_scomprimir = np.load(ruta_giros_juntos_sin_comprimir + "/GirosJuntos.npz")['X']

        ## ---------------------------------------- NORMALIZACIÓN ----------------------------------------------

        ## Creo un objeto que me permita hacer el escalado de los datos
        escalador = StandardScaler()

        ## Hago la estandarización de los datos
        ## Cada columna me queda entonces de valor medio 1 y desviación estándar 0
        escalograma_giros_scomprimir_norm = escalador.fit_transform(escalograma_giros_scomprimir)

        ## --------------------------------------------- PCA ---------------------------------------------------

        ## Genero un objeto el cual realizará el PCA
        pca_init = PCA()

        ## Ajusto el modelo de PCA a los datos
        pca_init.fit(escalograma_giros_scomprimir_norm)

        ## Genero una variable donde guardo la suma de la explicación de la varianza del dataset por las escalograma_giros_scomprimir
        suma_exp_varianza = pca_init.explained_variance_ratio_.cumsum()

        ## Grafico la varianza explicada por los componentes
        plt.figure(figsize = (10, 8))
        plt.plot(range(1, suma_exp_varianza.shape[0] + 1), suma_exp_varianza, marker = 'o', linestyle = '--')
        plt.title('Varianza Explicada por los componentes')
        plt.xlabel('Número de componentes')
        plt.ylabel('Varianza Explicada Acumulativa')
        plt.show()

        ## Construyo una variable que me va a decir la cantidad de componentes que tengo
        componentes = 0

        ## Itero para cada una de las proporciones explicativas de la varianza
        for i in range (len(suma_exp_varianza)):
            
            ## Aumento en una unidad la cantidad de componentes
            componentes += 1

            ## En caso de que la suma total de la varianza explicada sea mayor a 0.9
            if suma_exp_varianza[i] > 0.9:

                ## Termino el bucle
                break

        ## Ahora aplico PCA con la cantidad de componentes que seleccioné con el criterio anterior
        pca = PCA(n_components = componentes)

        ## Ajusto el modelo a mis datos con la cantidad seleccionada de componentes
        pca.fit(escalograma_giros_scomprimir_norm)

        ## Hago la selección de los escalograma_giros_scomprimir transformando según las componentes principales halladas (que explican una varianza del 90% aproximadamente)
        escalograma_giros_scomprimir_norm = pca.transform(escalograma_giros_scomprimir_norm)