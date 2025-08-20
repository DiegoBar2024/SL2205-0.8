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

    ## <<train_values>> me va a contener las muestras que voy a usar para el entrenamiento del clasificador, según la etiqueta correspondiente
    ## Tiene la forma de (cantidad de muestras de entrenamiento, 256)
    train_values = np.concatenate([stable_train_intermediate, unstable_train_intermediate], axis = 0)

    ## Genero un vector con las etiquetas correspondientes para cada uno de los pacientes en función de su estabilidad o no para el conjunto de validación
    ground_truth_validate = np.concatenate([np.zeros(np.shape(stable_validate_intermediate)[0]), np.ones(np.shape(unstable_validate_intermediate)[0])]) 

    ## <<val_values>> me va a contener las muestras que voy a usar para el entrenamiento del clasificador, según la etiqueta correspondiente
    ## Tiene la forma de (cantidad de muestras de validación, 256)
    val_values = np.concatenate([stable_validate_intermediate, unstable_validate_intermediate], axis = 0)
    
    ## Obtengo el clasificador luego de realizar el entrenamiento
    train_clasificador(train_values, ground_truth_train, clf_name + '.joblib', clasificador, val_values, ground_truth_validate)

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

    ## En caso de que el clasificador sea del tipo SVM
    elif clasificador == 'svm':
        
        ## Construyo la Support Vector Machine con los valores de sus hiperparámetros
        clf = SVC(C = 1, gamma = 1, kernel = 'rbf')

        ## Llevo a cabo el entrenamiento del clasificador
        ## <<values>> es la secuencia de valores de entrada
        ## <<ground_truth>> es la secuencia de valores de salida
        clf.fit(values, ground_truth)

    ## Guardo el modelo entrenado en la ruta de salida
    dump(clf, model_path_clf + file_clf)

    ## Retorno el clasificador entrenado
    return clf

def predict_clf(clf, intermediate_layer, clasificador = "None"):
    """
    Function that recibes a lda classifier and the intermediate layer of a group of samples, and predict
    the samples classification.
    """

    ## Determino la predicción del clasificador ante mi muestra de entrada
    pat_predictions = clf.predict(intermediate_layer)

    ## En caso que el valor de la predicción numérica sea mayor a 0.5, asigno la variable <<pat_predictions>> a True (escalograma inestable)
    ## En caso que el valor de la predicción numérica sea menor a 0.5, asigno la variable <<pat_predictions>> a False (escalograma estable)
    pat_predictions = pat_predictions > 0.5
        
    return pat_predictions