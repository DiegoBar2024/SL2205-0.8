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
from LecturaDatosPacientes import *
from mlxtend.preprocessing import MeanCenterer
from dtwParallel import dtw_functions
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import normalize

## ------------------------------------------- ETIQUETADOS ---------------------------------------------

## Importación de etiquetas (provenientes del fichero de ingesta_etiquetas())
(x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, x_ae_train, x_ae_val) = ingesta_etiquetas() 

## <<train>> va a tener el conjunto de IDs de pacientes que se van a usar para el entrenamiento
train = np.concatenate((x_inestables_train_clf, x_estables_train_clf), axis = None)

## <<val_test>> va a tener el conjunto de IDs de pacientes que se van a usar para la validación
val_test = np.concatenate((x_inestables_val_clf, x_estables_val_clf), axis = None)

## ------------------------------------- SELECCIÓN DE MUESTRAS -----------------------------------------

## Obtengo la lista de todos los pacientes para los que se cuentan registros en la base de datos
pacientes, ids_existentes = LecturaDatosPacientes()

## Obtengo la lista de los pacientes inestables para el entrenamiento del clasificador
train_inestables = np.intersect1d(ids_existentes, x_inestables_train_clf)

## Obtengo la lista de los pacientes estables para el entrenamiento del clasificador
train_estables = np.intersect1d(ids_existentes, x_estables_train_clf)

## Obtengo la lista de los pacientes inestables para la validación del clasificador
val_inestables = np.intersect1d(ids_existentes, x_inestables_val_clf)

## Obtengo la lista de los pacientes estables para la validación del clasificador
val_estables = np.intersect1d(ids_existentes, x_estables_val_clf)

## Obtengo el conjunto de estables
id_estables = np.concatenate((train_estables, val_estables))

## Obtengo el conjunto de inestables
id_inestables = np.concatenate((train_inestables, val_inestables))

## Obtengo el conjunto de los IDs de todos los pacientes
ids_pacientes = np.concatenate((id_estables, id_inestables))

## ---------------------------------- CARGADO DE COMPRIMIDAS ESTABLES ----------------------------------

## Especifico la ruta en la cual se encuentran las muestras comprimidas
ruta_comprimidas = "C:/Yo/Tesis/sereData/sereData/Dataset/latente_ae"

## Construyo una lista donde voy a guardar las muestras etiquetadas como estables
estables = np.zeros((1, 256))

## Construyo una lista donde voy a guardar las muestras etiquetadas como inestables
inestables = np.zeros((1, 256))

## Inicializo un diccionario cuya clave va a ser el ID del paciente y el valor una tupla con las posiciones de sus objetos en la matriz en estables
ids_posiciones_estables = {}

## Inicializo un diccionario cuya clave va a ser el ID del paciente y el valor una tupla con las posiciones de sus objetos en la matriz en inestables
ids_posiciones_inestables = {}

## Itero para cada uno de los IDs considerados como estables
for estable in id_estables:

    ## Construyo un bloque try except para atrapar errores
    try:

        ## Abro el archivo donde tengo la muestra comprimida
        comprimida_estable = np.load(ruta_comprimidas + '/S{}/S{}_latente.npz'.format(estable, estable))

        ## Selecciono la representación en espacio latente de los segmentos de marcha
        latente_estable = comprimida_estable['X']

        ## Guardo en el diccionario la ID del paciente con sus posiciones
        ids_posiciones_estables[str(estable)] = (len(estables) - 1, len(estables) + len(latente_estable) - 2)

        ## Agrego dicho espacio latente a la lista de estables
        estables = np.concatenate((estables, latente_estable), axis = 0)

    ## En caso de que haya algún error
    except:

        ## Que continúe con la siguiente muestra
        continue

## -------------------------------- CARGADO DE COMPRIMIDAS INESTABLES ----------------------------------

## Itero para cada uno de los IDs considerados como inestables
for inestable in id_inestables:

    ## Construyo un bloque try except para atrapar errores
    try:

        ## Abro el archivo donde tengo la muestra comprimida
        comprimida_inestable = np.load(ruta_comprimidas + '/S{}/S{}_latente.npz'.format(inestable, inestable))

        ## Selecciono la representación en espacio latente de los segmentos de marcha
        latente_inestable = comprimida_inestable['X']

        ## Guardo en el diccionario la ID del paciente con sus posiciones
        ids_posiciones_inestables[str(inestable)] = (len(inestables) - 1, len(inestables) + len(latente_inestable) - 2)

        ## Agrego dicho espacio latente a la lista de estables
        inestables = np.concatenate((inestables, latente_inestable), axis = 0)

    ## En caso de que haya algún error
    except:

        ## Que continúe con la siguiente muestra
        continue

## Selecciono únicamente aquellas muestras no nulas (elimino el dummy vector) para estables
estables = estables[1:,:]

## Selecciono únicamente aquellas muestras no nulas (elimino el dummy vector) para inestables
inestables = inestables[1:,:]

## Construyo una matriz con todos los segmentos comprimidos
comprimidos_total = np.concatenate((estables, inestables))

## Construyo una matriz con todas las etiquetas
vector_etiquetas =  np.concatenate((np.zeros(len(estables)), np.ones(len(inestables))))

## Hago la normalización por columna es decir por feature
#comprimidos_total = normalize(comprimidos_total, norm = "l2", axis = 0)

## -------------------------------- VALIDACIÓN CRUZADA DE LOS MODELOS ----------------------------------

## Construyo la Support Vector Machine
svm = SVC(C = 1, gamma = 1, kernel = 'rbf')

## Especifico la cantidad de folds que voy a utilizar para poder hacer la validación cruzada
k_folds = KFold(n_splits = 10, shuffle = True)

## Hago la validación cruzada del modelo
scores_svm = cross_val_score(svm, comprimidos_total, vector_etiquetas, cv = k_folds)

## Construyo el clasificador LDA
lda = LinearDiscriminantAnalysis()

## Hago la validación cruzada del modelo
scores_lda = cross_val_score(lda, comprimidos_total, vector_etiquetas, cv = k_folds)

## Inicializo una lista en la cual voy a guardar los errores de predicción para el modelo
errores_prediccion = []

## Genero una variable la cual especifique el modelo que voy a usar para hacer la validacion
modelo = 'svm'

## Itero para cada uno de los pacientes en el conjunto de IDs de entrenamiento y validacion
for id_paciente in np.sort(ids_pacientes):

    ## Construyo un bloque try except para atrapar errores
    try:

        ## Asigno por defecto el conjunto de entrenamiento de los estables al vector total de estables
        train_estables = normalize(estables, norm = "l2", axis = 0)

        ## Asigno por defecto el conjunto de entrenamiento de los inestables al vector total de inestables
        train_inestables = normalize(inestables, norm = "l2", axis = 0)

        ## En caso de que el paciente sea estable
        if id_paciente in id_estables:

            ## Selecciono los segmentos asociados al paciente como conjunto de validacion
            validation_set = train_estables[ids_posiciones_estables[str(id_paciente)][0] : ids_posiciones_estables[str(id_paciente)][1] + 1, :]

            ## Saco el conjunto de validación del conjunto de entrenamiento de estables
            train_estables = np.concatenate((train_estables[ : ids_posiciones_estables[str(id_paciente)][0], :], train_estables[ids_posiciones_estables[str(id_paciente)][1] + 1 :, :]))

            ## Genero el vector de etiquetas para el conjunto de validación correspondiente
            etiquetas_val = np.zeros(len(validation_set))

        ## En caso de que el paciente sea inestable
        elif id_paciente in id_inestables:

            ## Selecciono los segmentos asociados al paciente como conjunto de validacion
            validation_set = train_inestables[ids_posiciones_inestables[str(id_paciente)][0] : ids_posiciones_inestables[str(id_paciente)][1] + 1, :]

            ## Saco el conjunto de validación del conjunto de entrenamiento de inestables
            train_inestables = np.concatenate((train_inestables[ : ids_posiciones_inestables[str(id_paciente)][0], :], train_inestables[ids_posiciones_inestables[str(id_paciente)][1] + 1 :, :]))

            ## Genero el vector de etiquetas para el conjunto de validación correspondiente
            etiquetas_val = np.ones(len(validation_set))

        ## Obtengo el vector de etiquetas según estabilidad (0 es estable, 1 es inestable)
        etiquetas = np.concatenate((np.zeros(train_estables.shape[0]), np.ones(train_inestables.shape[0])))

        ## Hago la concatenación de las matrices para obtener el conjunto de entrenamiento total
        train_set = np.concatenate((train_estables, train_inestables))

        ## En caso de que el modelo elegido sea LDA
        if modelo == 'lda':

            ## Construyo el modelo inicial
            lda_model = LinearDiscriminantAnalysis()

            ## Hago el entrenamiento del modelo usando el conjunto de entrenamiento y sus etiquetas
            lda_model.fit(train_set, etiquetas)

            ## Hago la prediccion del modelo para el validation set
            val_predict = lda_model.predict(validation_set)
        
        ## En caso de que el modelo elegido sea SVM
        elif modelo == 'svm':

            ## Construyo el modelo inicial
            svm_model = SVC(C = 1, gamma = 1, kernel = 'rbf')

            ## Hago el entrenamiento del modelo usando el conjunto de entrenamiento y sus etiquetas
            svm_model.fit(train_set, etiquetas)

            ## Hago la prediccion del modelo para el validation set
            val_predict = svm_model.predict(validation_set)
        
        ## Calculo el error correspondiente al modelo como el cociente entre los segmentos mal clasificados del paciente y los segmentos totales
        error = np.logical_xor(val_predict, etiquetas_val)[np.logical_xor(val_predict, etiquetas_val) == True].shape[0] / np.logical_xor(val_predict, etiquetas_val).shape[0]

        ## Agrego el error de predicción junto con el ID del paciente usado para la validación
        errores_prediccion.append([id_paciente, error])

    ## En caso de que haya algún error
    except:

        ## Que continúe con la siguiente muestra
        continue

## Hago la conversión de la matriz de errores de predicción a precisiones
errores_prediccion = np.array((errores_prediccion))

## Inicializo un vector en donde voy a guardar las precisiones de las predicciones
precisiones = []

## Itero para cada una de las predicciones
for i in range (len(errores_prediccion)):   

    ## Me quedo con el ID del paciente correspondiente y la precision que se calcula en base al error
    precisiones.append([errores_prediccion[i][0], 1 - errores_prediccion[i][1]])

## Hago la conversión a vector numpy
precisiones = np.array(precisiones)