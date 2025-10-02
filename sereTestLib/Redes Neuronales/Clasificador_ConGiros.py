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
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.preprocessing import normalize
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SequentialFeatureSelector

## Defino una función que me permita ver si dos matrices tienen al menos una fila en común
def FilasEnComun(matriz1, matriz2):

    ## Itero para todas las filas de matriz1
    for i in range (matriz1.shape[0]):

        ## Itero para todas las filas de matriz2
        for j in range (matriz2.shape[0]):

            ## En caso de que exista una fila en común
            if (matriz1[i, :] == matriz2[j, :]).all():

                ## Retorno True
                return True
    
    ## En caso de que no haya filas en común, retorno False
    return False

## Construyo una función que me haga el cargado de todos los segmentos comprimidos estables
def CargarComprimidosEstables():

    ## Construyo una lista donde voy a guardar las muestras etiquetadas como estables
    estables = np.zeros((1, 256))

    ## Inicializo un diccionario cuya clave va a ser el ID del paciente y el valor una tupla con las posiciones de sus objetos en la matriz en estables
    ids_posiciones_estables = {}

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
    
    ## Selecciono únicamente aquellas muestras no nulas (elimino el dummy vector) para estables
    estables = estables[1:,:]

    ## Retorno los segmentos comprimidos estables y los ids de las posiciones
    return estables, ids_posiciones_estables

## Construyo una función que me haga el cargado de todos los segmentos comprimidos inestables
def CargarComprimidosInestables():

    ## Construyo una lista donde voy a guardar las muestras etiquetadas como inestables
    inestables = np.zeros((1, 256))

    ## Inicializo un diccionario cuya clave va a ser el ID del paciente y el valor una tupla con las posiciones de sus objetos en la matriz en inestables
    ids_posiciones_inestables = {}

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

    ## Selecciono únicamente aquellas muestras no nulas (elimino el dummy vector) para inestables
    inestables = inestables[1:,:]

    ## Retorno los segmentos comprimidos estables y los ids de las posiciones
    return inestables, ids_posiciones_inestables

## Construyo una función que me combine todos los datos estables e inestables y me los normalice
def CombinarNormalizar(estables, inestables):

    ## Construyo una matriz con todos los segmentos comprimidos
    comprimidos_total = np.concatenate((estables, inestables))

    ## Construyo una matriz con todas las etiquetas
    vector_etiquetas =  np.concatenate((np.zeros(len(estables)), np.ones(len(inestables))))

    # Hago la normalización por columna es decir por feature
    comprimidos_total = normalize(comprimidos_total, norm = "l2", axis = 0)

    ## Retorno el vector de etiquetas y la matriz de comprimidos
    return comprimidos_total, vector_etiquetas

## Defino una función que me haga la validación cruzada individual aleatorio
def KFoldIndividualAleatoria(comprimidos_total, vector_etiquetas, folds, clasificador):

    ## Especifico la cantidad de folds que voy a utilizar para poder hacer la validación cruzada
    k_folds = KFold(n_splits = folds, shuffle = True)

    ## En caso de que el clasificador elegido sea la SVM
    if clasificador == 'svm':

        ## Construyo la Support Vector Machine
        clas = SVC(C = 1, gamma = 1, kernel = 'rbf')

    ## En caso de que el clasificador elegido sea LDA
    elif clasificador == 'lda':

        ## Construyo el clasificador LDA
        clas = LinearDiscriminantAnalysis()

    ## Hago la validación cruzada del modelo
    scores = cross_val_score(clas, comprimidos_total, vector_etiquetas, cv = k_folds)

    ## Retorno los scores de LDA y SVM
    return scores

## Defino una función que me construya la matriz de confusión para el desempeño K-Fold de los modelos
def KFoldAleatoriaMatrizConf(comprimidos_total, vector_etiquetas, folds, clasificador):

    ## Especifico la cantidad de folds que voy a utilizar para poder hacer la validación cruzada
    k_folds = KFold(n_splits = folds, shuffle = True)

    ## En caso de que el clasificador elegido sea la SVM
    if clasificador == 'svm':

        ## Construyo la Support Vector Machine
        clas = SVC(C = 1, gamma = 1, kernel = 'rbf')

    ## En caso de que el clasificador elegido sea LDA
    elif clasificador == 'lda':

        ## Construyo el clasificador LDA
        clas = LinearDiscriminantAnalysis()
    
    ## Obtengo las predicciones del clasificador mediante validación cruzada
    vector_predicciones = cross_val_predict(clas, comprimidos_total, vector_etiquetas, cv = k_folds)

    ## Construyo y despliego la matriz de confusión
    cm = confusion_matrix(vector_etiquetas, vector_predicciones)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    disp.plot()
    plt.show()

## Defino una función que me haga una feature selection para poder hacer la clasificacion
def FeatureSelection(comprimidos_total, etiquetas, clasificador):

    ## En caso de que el modelo elegido sea LDA
    if clasificador == 'lda':

        ## Construyo el modelo de análisis en discriminante lineal
        clas = LinearDiscriminantAnalysis()

    ## En caso de que el modelo elegido sea SVM
    elif clasificador == 'svm':

        ## Construyo el modelo de Support Vector Machine
        clas = SVC(C = 1, gamma = 1, kernel = 'rbf')

    ## En caso de que el modelo elegido sea PERCEPTRON
    elif clasificador == 'perceptron':

        ## Construyo el modelo del perceptron
        clas = Perceptron()

    ## Construyo un selector de features. La cantidad de features óptima se selecciona automáticamente
    selector_features = SequentialFeatureSelector(clas)

    ## Hago el ajuste del selector de features
    selector_features.fit(comprimidos_total, etiquetas)

    ## Obtengo las columnas de los features que fueron seleccionados por SFFS
    columnas_features = np.where(selector_features.get_support() == True)

    ## Filtro la matriz de features con aquellas columnas que fueron seleccionados
    features_filt = comprimidos_total[:, columnas_features]

    ## Ajusto las dimensiones de lo anterior para que quede una matriz
    features_filt = np.reshape(features_filt, (features_filt.shape[0], features_filt.shape[2]))

    ## Retorno la matriz de datos luego de hacer el filtrado
    return features_filt

## Hago una función que me haga la LOO donde cada validation set son los registros de un paciente
def LOOPorPaciente(estables, inestables, ids_posiciones_estables, ids_posiciones_inestables, clasificador):

    ## Obtengo una lista de identificadores con todos los pacientes, ya sean estables o inestables
    ids_pacientes = np.concatenate((id_estables, id_inestables), axis = 0)

    ## Inicializo una lista en la cual voy a guardar los errores de predicción para el modelo
    errores_prediccion = []

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
            if clasificador == 'lda':

                ## Construyo el modelo de análisis en discriminante lineal
                clas = LinearDiscriminantAnalysis()

            ## En caso de que el modelo elegido sea SVM
            elif clasificador == 'svm':

                ## Construyo el modelo de Support Vector Machine
                clas = SVC(C = 1, gamma = 1, kernel = 'rbf')

            ## En caso de que el modelo elegido sea PERCEPTRON
            elif clasificador == 'perceptron':

                ## Construyo el modelo del perceptron
                clas = Perceptron()

            ## Hago el ajuste del perceptrón para el conjunto de validación
            clas.fit(train_set, etiquetas)

            ## Hago la predicción para el conjunto de validación
            val_predict = clas.predict(validation_set)
            
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

    ## Retorno la matriz con la ID de cada paciente y la precisión asociada
    return precisiones

## Ejecución principal del programa
if __name__== '__main__':

    ## Cargado de comprimidos estables
    estables, ids_posiciones_estables = CargarComprimidosEstables()

    ## Cargado de comprimidos inestables
    inestables, ids_posiciones_inestables = CargarComprimidosInestables()

    ## Hago la combinación de estables e inestables y normalizacion
    comprimidos_total, vector_etiquetas = CombinarNormalizar(estables, inestables)

    ## Especifico el modelo de clasificador que voy a usar
    clasificador = 'lda'

    ## Hago la selección de features usando SFFS
    comprimidos_total = FeatureSelection(comprimidos_total, vector_etiquetas, clasificador)

    ## Obtengo la matriz de confusión con validación KFold
    KFoldAleatoriaMatrizConf(comprimidos_total, vector_etiquetas, 10, clasificador)

    ## Hago la validación LOO individual aleatoria
    scores = KFoldIndividualAleatoria(comprimidos_total, vector_etiquetas, 10, clasificador)

    ## Hago la validación LOO por paciente
    precisiones = LOOPorPaciente(estables, inestables, ids_posiciones_estables, ids_posiciones_inestables, clasificador)