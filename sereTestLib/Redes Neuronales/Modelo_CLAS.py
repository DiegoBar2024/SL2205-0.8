from math import isnan
import os, sys

from sklearn.model_selection import GridSearchCV
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

#logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
#from tensorflow.python.keras.backend import arange

import matplotlib.pyplot as plt
import numpy as np
#from numpy.core.numeric import NaN
from termcolor import colored
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score,confusion_matrix
####################################################################
# Libreria para trabajar con
# los modelos de scikitlearn
####################################################################
from joblib import dump, load
import json
from numpyencoder import NumpyEncoder
import scipy.stats as stats
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
import keras
from keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import sklearn

##################################
# Librerias y parametros Sere
##################################
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/clasificador')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/autoencoder')
from parameters import *
from Extras_CLAS import *
from GeneradorDatos import *

## Función que me construye un modelo de red neuronal fully connected arbitrario (la voy a usar para el clasificador)
## <<n_hidden>> me da el número de hidden layers que va a tener mi modelo
## <<n_neurons>> me da el número de neuronas que va a tener cada hidden layer de mi modelo. Se asume en éste caso que todas las hidden layers tienen el mismo número <<n_neurons>> de neuronas 
## <<input_shape>> me da la forma de la entrada que va a tener la red neuronal. Por defecto (None, 256) significa que la entrada va a ser un array unidimensional de tamaño 256 (256 características)
def build_model(n_hidden, n_neurons, input_shape = (None, 256)):
    
    ## Inicializo un modelo Keras Secuencial inicialmente vacío
    model = keras.models.Sequential()

    ## Agrego al modelo una capa de entrada diciendole que la entrada a la red tiene una forma especificada por <<input_shape>> la cual en mi caso es por defecto (None, 256)
    ## Dicho de otro modo le estoy diciendo a mi modelo que la entrada es un vector unidimensional de 256 elementos 
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    ## Itero para cada uno de los hidden layers que especifiqué a la entrada. Es decir voy a tener layer = 0, 1, ..., n_hidden - 1
    for layer in range(n_hidden):

        ## Agrego primero una "Dense Layer" (fully connected) consistente de una cantidad <<n_neurons>> de neuronas y una función de activación ReLU
        ## Recuerdo que una Dense Layer implementa la combinación lineal de las salidas de todas las neuronas del layer previo sumado a un termino de bias
        model.add(keras.layers.Dense(n_neurons, activation="relu"))

        ## Luego agrego una segunda "Dense Layer" (fully connected) consistente de una única neurona
        ## Ésta neurona va a calcular una combinación lineal de las salidas <<n_neurons>> sumado a un término de bias 
        model.add(keras.layers.Dense(1))

        ## Se hace la compilación del modelo tomando como función de error la MSE ('Mean Squared Error') la cual es la función de error de MÍNIMOS CUADRADOS
        ## Luego se toma como optimizador el SGD ('Stochastic Gradient Descent') el cual va a ser el método de optimización que va a usar la red neuronal para corregir sus parámetros luego de hacer el cálculo del MSE (se aplica backpropagation)
        model.compile(loss = "mean_squared_error", optimizer = "sgd")

    return model

def entrenamiento_clasificador(clf_name, unstable_train, stable_train, unstable_validation, stable_validation, autoencoder_model, clasificador, scalogram_path, activities = act_clf):
    """
    Function that creates and fits the classifier with the training set and predicts the validation set.

    Parameters:
    -----------
    unstable_train: list
    stable_train: list
    unstable_validation: list
    stable_validation: list
    autoencoder_model:
        Trained autoencoder model
    activities: list
        List of activities to train and validate the classifier.
    """

    ## Parámetros del entrenamiento del clasificador
    paramsT = {'data_dir' : scalogram_path,
                        'dim': inDim,
                        'batch_size': batch_size,
                        'shuffle': True,
                        'activities': activities}

    ## Parámetros de la validación del clasificador
    paramsV = {'data_dir' : scalogram_path,
                        'dim': inDim,
                        'batch_size': batch_size,
                        'shuffle': False,
                        'activities': activities}

    ## La palabra <<intermediate>> hace referencia al espacio codificado del autoencoder para cada una de las muestras
    ## Muestras de entrenamiento inestables (espacio comprimido del autoencoder)
    ## Tiene la forma de (cantidad de muestras inestables de entrenamiento, 256)
    unstable_train_intermediate = patient_group_aelda(unstable_train, autoencoder_model, layer_name = layer_name, **paramsT)

    ## Muestras de entrenamiento estables (espacio comprimido del autoencoder)
    ## Tiene la forma de (cantidad de muestras estables de entrenamiento, 256)
    stable_train_intermediate = patient_group_aelda(stable_train, autoencoder_model, layer_name = layer_name, **paramsT)

    ## Muestras de validación inestables (espacio comprimido del autoencoder)
    ## Tiene la forma de (cantidad de muestras inestables de validación, 256)
    unstable_validate_intermediate = patient_group_aelda(unstable_validation, autoencoder_model, layer_name = layer_name, **paramsV)

    ## Muestras de validación inestables (espacio comprimido del autoencoder)
    ## Tiene la forma de (cantidad de muestras estables de validación, 256)
    stable_validate_intermediate = patient_group_aelda(stable_validation, autoencoder_model, layer_name = layer_name, **paramsV)

    ## CRITERIO DE ASIGNACIÓN DE ETIQUETAS
    ## Los pacientes con la etiqueta "0" son ESTABLES
    ## Los pacientes con la etiqueta "1" son INESTABLES

    ## Genero un vector con las etiquetas correspondientes para cada uno de los pacientes en función de su estabilidad o no para el conjunto de ENTRENAMIENTO
    ground_truth_train = np.concatenate([np.zeros(np.shape(stable_train_intermediate)[0]), np.ones(np.shape(unstable_train_intermediate)[0])])
    # ground_truth_train = np.concatenate([- np.ones(np.shape(stable_train_intermediate)[0]), np.ones(np.shape(unstable_train_intermediate)[0])]) 

    ## <<train_values>> me va a contener las muestras que voy a usar para el entrenamiento del clasificador, según la etiqueta correspondiente
    ## Tiene la forma de (cantidad de muestras de entrenamiento, 256)
    train_values = np.concatenate([stable_train_intermediate, unstable_train_intermediate], axis = 0)

    ## Genero un vector con las etiquetas correspondientes para cada uno de los pacientes en función de su estabilidad o no para el conjunto de validación
    ground_truth_validate = np.concatenate([np.zeros(np.shape(stable_validate_intermediate)[0]), np.ones(np.shape(unstable_validate_intermediate)[0])]) 
    # ground_truth_validate = np.concatenate([- np.ones(np.shape(stable_validate_intermediate)[0]), np.ones(np.shape(unstable_validate_intermediate)[0])]) 

    ## <<val_values>> me va a contener las muestras que voy a usar para el entrenamiento del clasificador, según la etiqueta correspondiente
    ## Tiene la forma de (cantidad de muestras de validación, 256)
    val_values = np.concatenate([stable_validate_intermediate, unstable_validate_intermediate], axis = 0)
    
    ## Obtengo el clasificador luego de realizar el entrenamiento
    clf_trained = train_clasificador(train_values, ground_truth_train, clf_name + '.joblib', clasificador, val_values, ground_truth_validate)

    ## Hago la predicción para el conjunto de entrenamiento inestable
    unstable_predictions_train = predict_clf(clf_trained, unstable_train_intermediate, clasificador)

    ## Hago la predicción para el conjunto de entrenamiento estable
    stable_predictions_train = predict_clf(clf_trained, stable_train_intermediate, clasificador)

    ## Etiquetas del conjunto de entrenamiento
    labels_train = np.concatenate([stable_predictions_train, unstable_predictions_train], axis = 0)

    ## Hago la predicción para el conjunto de validación inestable
    unstable_predictions = predict_clf(clf_trained, unstable_validate_intermediate, clasificador)

    ## Hago la predicción para el conjunto de validación estable
    stable_predictions = predict_clf(clf_trained, stable_validate_intermediate, clasificador)

    ## Etiquetas del conjunto de validación
    labels_validate = np.concatenate([stable_predictions, unstable_predictions], axis = 0)

## Función que lleva a cabo el entrenamiento del clasificador
## Como entrada debo colocarle el espacio codificado siendo éste la salida del autoencoder para cada escalograma
## Como salida debo colocarle el label el cual me especifica si el escalograma correspondiente al vector de entrada
def train_clasificador(values, ground_truth, file_clf, clasificador, X_val = None, y_val = None):
    """
    Function that trains classifier based on a the intermediate layer of a group of samples.
    and respective ground_truth.
    Parameters
    ----------
    values: list
        intermediate layer of the samples
    ground_truth: list
        value labels
    file_clf: str
    clasificador: str
        'lda', 'svm', 'perceptron', 'hierarchical'

    Returns
    -------
    clf:
        Trained classifier
    """
    
    # En caso de que no exista
    if not os.path.exists(model_path_clf):

        ## Se crea el directorio de salida de datos
        os.makedirs(model_path_clf)

    ## En caso de que el clasificador sea del tipo LDA
    if clasificador == 'lda':

        ## Especifico como clasificador aquel de análisis de discriminante lineal
        clf = LinearDiscriminantAnalysis()

        ## Llevo a cabo el entrenamiento del clasificador
        ## <<values>> es la secuencia de valores de entrada
        ## <<ground_truth>> es la secuencia de valores de salida
        clf.fit(values, ground_truth)

    ## En caso de que el clasificador sea del tipo PERCEPTRON
    elif clasificador == 'perceptron':

        ## Construyo un perceptrón mediante el uso de Keras como una capa secuencial de una sola neurona
        clf = keras.models.Sequential([keras.layers.Dense(1, activation = "sigmoid", input_shape = (np.shape(values)[1],))])

        ## Impongo un tipo de regularización del tipo Early Stopping
        es = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min', verbose = 1)

        ## Compilo el modelo especificando error MSE y una optimización usando SGD (Stochastic Gradient Descent)
        clf.compile(loss = "mean_squared_error", optimizer = "sgd")

        ## Llevo a cabo el entrenamiento del clasificador y especifico los diferentes parámetros de entrenamiento
        clf.fit(values, ground_truth, epochs = num_epochs, validation_data = (X_val, y_val), callbacks = [es])
    
    ## En caso de que el clasificador sea del tipo NN (Red Neuronal)
    elif clasificador == "NN":

        ## Impongo un tipo de regularización del tipo Early Stopping
        es = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min', verbose = 1)

        ## Hago la construcción de la red neuronal correspondiente
        clf = build_model(n_neurons = 40, n_hidden = 5)

        ## Llevo a cabo el entrenamiento del clasificador
        ## <<values>> es la secuencia de valores de entrada
        ## <<ground_truth>> es la secuencia de valores de salida
        clf.fit(values, ground_truth, epochs = num_epochs, validation_data = (X_val, y_val), callbacks = [es])

    ## En caso de que el clasificador sea del tipo SVM
    elif clasificador == 'svm':
        
        ## Construyo la Support Vector Machine
        clf = SVC(C = 1, gamma = 1, kernel = 'rbf')

        ## Llevo a cabo el entrenamiento del clasificador
        ## <<values>> es la secuencia de valores de entrada
        ## <<ground_truth>> es la secuencia de valores de salida
        clf.fit(values, ground_truth)

    ## En caso de que el clasificador sea del tipo HIERARCHICAL
    elif clasificador == 'hierarchical':
        
        ## Especifico los parámetros del clasificador
        clf = AgglomerativeClustering(compute_full_tree = True, n_clusters = 2, compute_distances = True)

        ## Llevo a cabo el entrenamiento del clasificador
        ## <<values>> es la secuencia de valores de entrada
        ## <<ground_truth>> es la secuencia de valores de salida
        clf.fit(values, ground_truth)

    ## En caso de que el clasificador sea del tipo KMEANS
    elif clasificador == 'kmeans':

        ## Le indico al algoritmo KMEANS que tengo dos clusters
        ## Recuerdo que KMEANS es un algoritmo de clustering de modo que él por su cuenta va a buscar y organizar los clústers
        clf = KMeans(n_clusters = 2)

        ## Llevo a cabo el entrenamiento del clasificador
        ## <<values>> es la secuencia de valores de entrada
        ## <<ground_truth>> es la secuencia de valores de salida
        clf.fit(values, ground_truth)
    
    ## En caso de que el clasificador sea del tipo RF (Random Forest)
    elif clasificador == 'RF':

        ## Especifico los parámetros del clasificador Random Forest
        clf = RandomForestClassifier(n_estimators = 100, min_samples_split = 10, max_depth = 5)

        ## Llevo a cabo el entrenamiento del clasificador
        ## <<values>> es la secuencia de valores de entrada
        ## <<ground_truth>> es la secuencia de valores de salida
        clf.fit(values, ground_truth)

    dump(clf, model_path_clf + file_clf)

    ## Retorno el clasificador entrenado
    return clf

def predict_clf(clf, intermediate_layer, clasificador = "None"):
    """
    Function that recibes a lda classifier and the intermediate layer of a group of samples, and predict
    the samples classification.
    """

    ## En caso de que tenga un clasificador jerárquico (clustering)
    if clasificador == "hierarchical":

        ## Hago la predicción de los datos en base al espacio latente
        pat_predictions = clf.fit_predict(intermediate_layer)
    
    ## En caso de que tenga un clasificador kmeans (clustering)
    if clasificador == "kmeans":

        ## Hago la predicción de los datos en base al espacio latente
        pat_predictions = clf.predict(intermediate_layer)

    ## En caso de que el clasificador no sea "hierarchical"
    else:

        ## Determino la predicción del clasificador ante mi muestra de entrada
        pat_predictions = clf.predict(intermediate_layer)

        ## En caso de que el clasificador usado sea un perceptron
        if clasificador == "perceptron":

            ## En caso que el valor de la predicción numérica sea mayor a 0.5, asigno la variable <<pat_predictions>> a True (escalograma inestable)
            ## En caso que el valor de la predicción numérica sea menor a 0.5, asigno la variable <<pat_predictions>> a False (escalograma estable)
            pat_predictions = pat_predictions > 0.5

        ## En caso de que el clasificador sea una red neuronal
        if clasificador == "NN":

            ## En caso que el valor de la predicción numérica sea mayor a 0.5, asigno la variable <<pat_predictions>> a True (escalograma inestable)
            ## En caso que el valor de la predicción numérica sea menor a 0.5, asigno la variable <<pat_predictions>> a False (escalograma estable)
            pat_predictions = pat_predictions > 0.5

        ## En caso de que el clasificador usado sea un LDA
        if clasificador == "lda":

            ## En caso que el valor de la predicción numérica sea mayor a 0.5, asigno la variable <<pat_predictions>> a True (escalograma inestable)
            ## En caso que el valor de la predicción numérica sea menor a 0.5, asigno la variable <<pat_predictions>> a False (escalograma estable)
            pat_predictions = pat_predictions > 0.5
            
        ## En caso de que el clasificador usado sea un SVM
        if clasificador == "svm":

            ## En caso que el valor de la predicción numérica sea mayor a 0.5, asigno la variable <<pat_predictions>> a True (escalograma inestable)
            ## En caso que el valor de la predicción numérica sea menor a 0.5, asigno la variable <<pat_predictions>> a False (escalograma estable)
            pat_predictions = pat_predictions > 0.5

    return pat_predictions