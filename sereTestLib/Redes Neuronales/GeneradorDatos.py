from natsort.natsort import natsorted
import numpy as np
from numpy.core.numeric import full
from tensorflow import keras
import os
from parameters import *
import pandas as pd

class DataGeneratorAuto(keras.utils.Sequence):
    '''
    Generates data for Keras.
    Be careful. Check if the n_samples is greater than zero before use the output in Keras models.
    '''
    def __init__(self,  activities = [], list_IDs = [], data_dir = './', batch_size = 4, dim = inDim, numClases = 2, shuffle = True, long_sample = long_sample):
        '''Initialization'''

        ## En caso de que la muestra sea larga, inicializo el atributo <<character>> con el valor 'L'
        if long_sample:
            self.character = 'L'

        ## En caso de que la muestra sea corta, inicializo el atributo <<character>> con el valor 'S'
        else:
            self.character = 'S'
        
        ## El atributo <<dim>> va a almacenar las dimensiones de la entrada al autoencoder
        ## Yo tengo dimensiones (128,600,6) lo que quiere decir que tengo como entrada un tensor tridimensional con dimensiones espaciales de 128 x 600 y una profundidad de 6
        self.dim = dim

        ## El tamaño del batch, es decir el atributo <<batch_size>>, me va a determinar TAMAÑO DEL LOTE DE ENTRENAMIENTO. A cada lote de entrenamiento se denomina epoch
        ## Con "tamaño del lote de entrenamiento" me refiero al NÚMERO DE SEGMENTOS que compone cada lote de entrenamiento que se va a usar para entrenar la red.
        self.batch_size = batch_size

        ## El atributo <<data_dir>> me va a almacenar el path a la carpeta global en donde se encuentran todas las muestras a procesar
        self.data_dir = data_dir

        self.shuffle = shuffle

        ## El atributo <<list_IDs>> va a almacenar la lista de IDs de los pacientes que corresponden a las muestras de entrada
        self.list_IDs = list_IDs

        ## El atributo <<activities>> me va a guardar una lista de cadenas cuyos elementos sean los nombres de las actividades que quiero examinar
        self.activities = activities

        ## El atributo <<IDs_len>> me va a contener una lista con el número de archivos .csv a procesar por cada paciente
        self.IDs_len = []

        ## El atributo <<clases>> me va a dar el número de clases que tengo en la clasificación que estoy haciendo
        ## En mi caso tengo dos clases, que van a ser ESTABLE e INESTABLE
        self.clases = numClases

        ## El atributo <<n_samples>> me va a contener la CANTIDAD TOTAL de segmentos a procesar de todos mis pacientes
        self.n_samples = 0

        self.pow = []

        ## Itero para cada una de las IDs de los pacientes de la lista de IDs
        for pat_id in self.list_IDs:

            ## Genero el valor <<id_folder>> como la concatenación de <<character>> el cual era 'S' si se trata de una muestra corta y 'L' si se trata de una muestra larga, y la ID del paciente que corresponda
            ## Por ejemplo, si tengo una muestra corta es decir <<self.character>> = 'S' y tengo un paciente de ID 223 entonces id_folder = 'S223' concatenando ambas cosas
            id_folder = self.character + str(int(pat_id))

            ## Llamo a <<__get_files_lists>> para que me agarre de dicho paciente aquellos ficheros .csv los cuales contengan las actividades que se quieren procesar
            files = self.__get_files_lists(id_folder)

            ## Recuerdo que <<files>> es una lista de cadenas que va a contener la totalidad de archivos a procesar asociados a un paciente con las actividades correspondientes
            ## La instrucción <<np.shape(files)[0]>> me va a devolver la cantidad de archivos a procesar por parte de dicho paciente, o dicho de otro modo me da la cantidad de elementos de <<files>>
            self.IDs_len.append(np.shape(files)[0])

        ## <<IDs_len>> me guardará entonces la cantidad de segmentos a procesar por parte de cada paciente
        ## Ésto es, el i-ésimo elemento de <<IDs_len>> me dará la cantidad de segmentos .csv que tengo que procesar correspondientes al i-ésimo paciente, respetando el orden de la lista de IDs
        ## <<sum(self.IDs_len)>> me dará la suma de todos los elementos de <<IDs_len>>
        ## De éste modo el atributo <<self.n_samples>> contendrá la CANTIDAD TOTAL de segmentos a procesar de TODOS MIS PACIENTES 
        self.n_samples = sum(self.IDs_len)

        ## El atributo <<n_batches>> me va a almacenar la cantidad de batches que se pueden armar con el número total de segmentos que voy a usar para entrenar
        ## Recuerdo que en caso que la cantidad de segmentos no sea un múltiplo entero del tamaño del batch, <<n_batches>> será el siguiente número entero más próximo al que me de el cociente
        self.n_batches = self.__len__()

        self.on_epoch_end()

    ## Dado el <<id_folder>> de un determinado paciente, se obtiene la lista de csv asociado al paciente correspondiente a las actividades que quiero analizar. Los nombres de los archivos están ordenados de manera NATURAL
    def __get_files_lists(self, id_folder):

        ## Concateno primero los paths <<self.data_dir>> con <<id_folder>>
        fullD = self.data_dir + '/' + id_folder

        try:

            ## Guardo en <<files>> la lista de archivos que se encuentran dentro del path dado por fullD
            files = os.listdir(fullD)
        
        except:

            return []
            
        ## Recuerdo que <<dict_actividades = {'Sentado':'1','Parado':'2','Caminando':'3','Escalera':'4'}>> era aquel diccionario cuyas claves son los nombres de las actividades y los valores son los números asociados
        ## <<actividades>> me va a quedar como resultado una lista cuyos elementos van a ser los valores asociados a las actividades que pasé como lista de entrada en activities
        ## Por ejemplo, suponiendo que <<activities = ['Parado', 'Caminando']>> se me almacenará en <<actividades>> la lista <<actividades = ['2','3']>>
        actividades = [dict_actividades.get(activity) for activity in self.activities]

        ## Primero hago un filtrado de aquellos segmentos que me describan las actividades que estoy queriendo procesar
        ## Luego hago un ordenamiento natural usando <<natsorted>> de dichos archivos filtrados
        return natsorted([file for file in files if file.startswith(tuple(actividades))])

    def __get_file(self,id_folder,file_index):
        # print(id_folder)
        files = self.__get_files_lists(id_folder)
        return files[file_index]

    ## Función que me devuelve la cantidad de batches que puedo armar con mis segmentos
    def __len__(self):
        '''Denotes the number of batches per epoch'''
        
        ## Recuerdo que <<n_samples>> me da la cantidad total de segmentos que tengo para hacer el entrenamiento
        ## Por otro lado <<batch_size>> me va a dar la cantidad de muestras que voy a usar para cada lote de entrenamiento
        ## La función np.ceil(<<num>>) lo que hace es redondearme para arriba <<num>>. Por ejemplo: np.ceil(2.9) --> 3
        ## Lo que hago entonces acá es, para mis muestras, CUÁNTOS BATCHES EN TOTAL PUEDO ARMAR.
        ## Si la división <<self.n_samples / self.batch_size>> me da un número entero, entonces se me retorna LA CANTIDAD DE BATCHES DE TAMAÑO <<BATCH_SIZE>> que puedo armar con mis segmentos
        ## Si la división <<self.n_samples / self.batch_size>> no me da un número entero, entonces se me retorna LA CANTIDAD DE BATCHES DE TAMAÑO <<BATCH_SIZE>> que puedo armar con mis segmentos + 1 (debido al redondeo hacia arriba que tengo)
        ## A modo de ejemplo, teniendo n_samples = 12 y batch_size = 4 ésta función me retorna 3, que es la cantidad de batches llenos que puedo armar con mis 12 segmentos
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''

        # Generación de los índices del batch
        # Si es el ultimo batch, que sea del tamano que los datos permiten.
        if self.n_samples > 0:

            ## <<batch_s>> va a tener el tamaño del batch que voy a usar
            batch_s = self.batch_size

            ## Recuerdo que self.batch_size me da la cantidad de muestras de un batch
            ## Recuerdo que self.n_batches me da la cantidad total de escalogramas (muestras) que tengo disponibles
            ## self.n_batches me va a decir CUANTOS BATCHES DE MUESTRAS SE USAN EN UNA EPOCH
            if (index < self.n_batches - 1) :

                ## Genero los índices de los datos correspondientes
                indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

            else:
                batch_size_resto = np.mod(self.n_samples, self.batch_size)
                if batch_size_resto:
                    batch_s = batch_size_resto
                    indexes = self.indexes[index * self.batch_size:index * self.batch_size + batch_size_resto]
                else:
                    indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

            # Generación de los datos para el entrenamiento y validación
            X, y = self.__data_generation(indexes, batch_s)

            ## Retorno los datos correspondiente
            return X, y

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes, batch_s):
        '''Generates data containing batch_size samples'''

        ## Primero lo que hago es usar el metodo <<np.empty>> para crear un tensor X vacío de dimensiones (batch_s, self.dim[0], self.dim[1], self.dim[2])
        ## Los elementos de X van a ser numéricos de tipo punto flotante
        X = np.empty((batch_s, *self.dim), dtype = np.float32)

        ## Generación de datos
        ## Itero para cada i entre 0 y batch_s - 1. Es decir, i = 0, 1, 2 ... batch_s - 1
        for i in range(batch_s):

            ## Asigno la variable j a 0
            ## El rango de valores que podrá tomar la variable j debe ser menor al número de pacientes que tengo
            j = 0

            ## <<samples>> me va a almacenar la cantidad de segmentos a procesar asociados al j-ésimo paciente
            samples = self.IDs_len[j]

            if np.shape(self.indexes)[0] > i:
                num_sample = indexes[i] + 1
                while (samples <  num_sample):
                    j += 1
                    samples += self.IDs_len[j]

                ## <<pat_id>> va a contener el número identificador del paciente correspodniente
                pat_id = self.list_IDs[j]

                ## <<id_folder>> va a contener el identificador de la carpeta donde se encuentran los segmentos asociado al paciente
                ## De este modo, si <<pat_id>> = 299 y tengo una muestra corta por ejemplo, obtengo <<id_folder>> = 'S299'
                id_folder = self.character + str(int(pat_id))

                ## <<file_index>> me va a contener el índice del archivo actual
                file_index = indexes[i] - samples + self.IDs_len[j]

                ## Obtengo el archivo donde se encuentra el segmento (escalograma) individual a procesar
                file = self.__get_file(id_folder, file_index)
                
                ## <<dirIn>> me va a contener información de si tengo una muestra corta o larga (S o L) y el identificador del paciente correspondiente
                dirIn = self.character + str(int(pat_id))

                # <<fullPath>> me va a contener toda la ruta del escalograma que yo voy a procesar
                fullPath = self.data_dir + '/' + dirIn + '/' + file
                
                ## En caso de que exista la ruta que yo le estoy especificando
                if os.path.exists(fullPath):

                    ## Le digo que me cargue el archivo correspondiente
                    arch = np.load(fullPath)
                
                ## En caso de que no exista la ruta que yo le estoy especificando
                else:

                    ## Le digo a la iteración que prosiga con la siguiente muestra
                    continue

                ## <<auxX>> va a ser la variable que me contenga el escalograma tridimensional que voy a usar
                auxX = arch['X']

                ## La variable X va a ser un tensor de dimensión cuatro la cual va a contener todos los escalogramas tridimensionales asociados al batch
                ## La forma de dicho tensor va a estar dada por (tamaño_batch, c, f, t) donde:
                ## <<c>> es la cantidad de canales (tengo 6)
                ## <<f>> es la cantidad de frecuencias
                ## <<t>> es la cantidad de muestras temporales
                X[i, : ] = auxX

        return X, X