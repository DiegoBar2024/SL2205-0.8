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
import wandb

##################################
# Librerias y parametros Sere
##################################
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/clasificador')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/autoencoder')
from parameters import *
from extras import clasificador_name_creation, patient_group_aelda
from DataClassAuto import DataGeneratorAuto

## <<list_ids>> es la lista de IDs de los pacientes cuyos datos voy a usar
## Función que me calcula la potencia de los escalogramas
def calcular_potencia(list_ids, **params):

    scalograms = DataGeneratorAuto(list_IDs=list_ids, **params)

    pow=np.zeros((np.shape(scalograms)[0],6))
    #print(np.shape(pow))
    for j in range(np.shape(scalograms)[0]):
        for i in range (6):
            chan=scalograms[j][0][ 0,:, :, i]
            pow[j,i]=np.mean(chan**2)
    return pow

## Función que me construye un modelo de red neuronal fully connected arbitrario (la voy a usar para el clasificador)
## <<n_hidden>> me da el número de hidden layers que va a tener mi modelo
## <<n_neurons>> me da el número de neuronas que va a tener cada hidden layer de mi modelo. Se asume en éste caso que todas las hidden layers tienen el mismo número <<n_neurons>> de neuronas 
## <<input_shape>> me da la forma de la entrada que va a tener la red neuronal. Por defecto (None, 256) significa que la entrada va a ser un array unidimensional de tamaño 256 (256 características)
def build_model(n_hidden, n_neurons, input_shape = (None,256)):
    
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

def entrenamiento_clasificador(unstable_train, stable_train, unstable_validation, stable_validation, autoencoder_model, clasificador, scalogram_path_train = dir_preprocessed_data_train, scalogram_path_val = dir_preprocessed_data_test, activities = act_clf):
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
    paramsT = {'data_dir' : scalogram_path_train,
                        'dim': inDim,
                        'batch_size': batch_size,
                        'shuffle': True,
                        'activities':activities}

    ## Parámetros de la validación del clasificador
    paramsV = {'data_dir' : scalogram_path_val,
                        'dim': inDim,
                        'batch_size': batch_size,
                        'shuffle': False,
                        'activities':activities}

    ## La palabra <<intermediate>> hace referencia al espacio codificado del autoencoder para cada una de las muestras
    ## Muestras de entrenamiento inestables (espacio comprimido del autoencoder)
    unstable_train_intermediate = patient_group_aelda(unstable_train, autoencoder_model, layer_name = layer_name, **paramsT)

    ## Muestras de entrenamiento estables (espacio comprimido del autoencoder)
    stable_train_intermediate = patient_group_aelda(stable_train, autoencoder_model, layer_name = layer_name, **paramsT)

    ## Muestras de validación inestables (espacio comprimido del autoencoder)
    unstable_validate_intermediate = patient_group_aelda(unstable_validation, autoencoder_model, layer_name = layer_name, **paramsV)

    ## Muestras de validación inestables (espacio comprimido del autoencoder)
    stable_validate_intermediate = patient_group_aelda(stable_validation, autoencoder_model, layer_name = layer_name, **paramsV)

    ## CRITERIO DE ASIGNACIÓN DE ETIQUETAS
    ## Los pacientes con la etiqueta "0" son INESTABLES
    ## Los pacientes con la etiqueta "1" son ESTABLES

    ## Genero un vector con las etiquetas correspondientes para cada uno de los pacientes en función de su estabilidad o no para el conjunto de entrenamiento
    ground_truth_train = np.concatenate([np.zeros(np.shape(stable_train_intermediate)[0]), np.ones(np.shape(unstable_train_intermediate)[0])]) 

    ## <<train_values>> me va a contener las muestras que voy a usar para el entrenamiento del clasificador, según la etiqueta correspondiente
    train_values = np.concatenate([stable_train_intermediate, unstable_train_intermediate], axis = 0)

    ## Genero un vector con las etiquetas correspondientes para cada uno de los pacientes en función de su estabilidad o no para el conjunto de validación
    ground_truth_validate = np.concatenate([np.zeros(np.shape(stable_validate_intermediate)[0]), np.ones(np.shape(unstable_validate_intermediate)[0])]) # 0=stable 1=unstable

    ## <<val_values>> me va a contener las muestras que voy a usar para el entrenamiento del clasificador, según la etiqueta correspondiente
    val_values = np.concatenate([stable_validate_intermediate,unstable_validate_intermediate],axis=0)
    
    ## Obtengo el clasificador luego de realizar el entrenamiento
    clf_trained = train_clasificador(train_values,ground_truth_train,clasificador_name +'.joblib', clasificador,val_values,ground_truth_validate )

    ## Hago la predicción para el conjunto de entrenamiento
    unstable_predictions_train = predict_clf(clf_trained,unstable_train_intermediate, clasificador)
    stable_predictions_train = predict_clf(clf_trained,stable_train_intermediate, clasificador)
    labels_train = np.concatenate([stable_predictions_train,unstable_predictions_train],axis=0)

    ## Hago la predicción para el conjunto de validación
    unstable_predictions = predict_clf(clf_trained,unstable_validate_intermediate, clasificador)
    stable_predictions = predict_clf(clf_trained,stable_validate_intermediate, clasificador)
    labels_validate = np.concatenate([stable_predictions,unstable_predictions],axis=0)

    ## En caso de que el clasificador sea del tipo LDA, hago una impresión adicional
    if clasificador == 'lda':
        print_write_classifier_stats(ground_truth_train,labels_train, ground_truth_validate,labels_validate,0,0,0,0,0,clasificador,activities,extra)
    
    else:
        print_write_classifier_stats(ground_truth_train, labels_train, ground_truth_validate,labels_validate,0, 0, 0, 0,0,clasificador,activities,extra)

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
    
    # En caso de que no exista, se crea el directorio de salida de datos
    if not os.path.exists(model_path_clf):
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
        clf = keras.models.Sequential([keras.layers.Dense(1, activation = "sigmoid", input_shape = (None,np.shape(values)[1]))])

        ## Impongo un tipo de regularización del tipo Early Stopping
        es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

        ## Compilo el modelo especificando error MSE y una optimización usando SGD (Stochastic Gradient Descent)
        clf.compile(loss = "mean_squared_error", optimizer = "sgd")

        ## Llevo a cabo el entrenamiento del clasificador y especifico los diferentes parámetros de entrenamiento
        clf.fit(values,ground_truth,  epochs=100, validation_data=(X_val, y_val),callbacks=[es])
    
    ## En caso de que el clasificador sea del tipo NN (Red Neuronal)
    elif clasificador == "NN":

        ## Impongo un tipo de regularización del tipo Early Stopping
        es = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min', verbose = 1)

        ## Hago la construcción de la red neuronal correspondiente
        clf = build_model(n_neurons = 40, n_hidden = 5)

        ## Llevo a cabo el entrenamiento del clasificador
        ## <<values>> es la secuencia de valores de entrada
        ## <<ground_truth>> es la secuencia de valores de salida
        clf.fit(values,ground_truth,  epochs=100, validation_data=(X_val, y_val),callbacks=[es])

    ## En caso de que el clasificador sea del tipo SVM
    elif clasificador == 'svm':
        
        ## Construyo la Support Vector Machine
        clf = SVC(C = 1, gamma = 1, kernel = 'rbf')

        ## Llevo a cabo el entrenamiento del clasificador
        ## <<values>> es la secuencia de valores de entrada
        ## <<ground_truth>> es la secuencia de valores de salida
        clf.fit(values,ground_truth)

    ## En caso de que el clasificador sea del tipo HIERARCHICAL
    elif clasificador == 'hierarchical':
        
        ## Especifico los parámetros del clasificador
        clf = AgglomerativeClustering(compute_full_tree = True, n_clusters = 2, compute_distances = True)

        ## Llevo a cabo el entrenamiento del clasificador
        ## <<values>> es la secuencia de valores de entrada
        ## <<ground_truth>> es la secuencia de valores de salida
        clf.fit(values,ground_truth)

    ## En caso de que el clasificador sea del tipo KMEANS
    elif clasificador == 'kmeans':

        ## Le indico al algoritmo KMEANS que tengo dos clusters
        ## Recuerdo que KMEANS es un algoritmo de clustering de modo que él por su cuenta va a buscar y organizar los clústers
        clf = KMeans(n_clusters = 2)

        ## Llevo a cabo el entrenamiento del clasificador
        ## <<values>> es la secuencia de valores de entrada
        ## <<ground_truth>> es la secuencia de valores de salida
        clf.fit(values, ground_truth)

    ## En caso de que el clasificador sea del tipo VOTING
    ## En éste caso defino varios clasificadores y comparo el desempeño para cada uno de ellos
    elif clasificador == "voting":

        ## Defino el clasificador LDA (Linear Discriminant Analysis)
        clf1 = LinearDiscriminantAnalysis()

        ## Especifico los parámetros del clasificador SVM (Support Vector Machine)
        clf2 = SVC(C = 1, gamma = 1, kernel = 'rbf', probability = True)

        ## Especifico los parámetros del clasificador RF (Random Forest)
        clf3 = RandomForestClassifier(n_estimators = 100, min_samples_split = 10, max_depth = 5)

        clf = VotingClassifier(estimators=[('lda', clf1), ('svc', clf2),("rf",clf3)], voting='hard')
        clf.fit(values,ground_truth)
    
    ## En caso de que el clasificador sea del tipo RF (Random Forest)
    elif clasificador=='RF':

        ## Especifico los parámetros del clasificador Random Forest
        clf = RandomForestClassifier(n_estimators = 100, min_samples_split = 10, max_depth = 5)

        ## Llevo a cabo el entrenamiento del clasificador
        ## <<values>> es la secuencia de valores de entrada
        ## <<ground_truth>> es la secuencia de valores de salida
        clf.fit(values,ground_truth)

    dump(clf,model_path_clf+file_clf)
    #hay que ver como guardarlo para levantarlo luego

    ## Retorno el clasificador entrenado
    return clf

def predict_clf(clf,intermediate_layer, clasificador="None"):
    """
    Function that recibes a lda classifier and the intermediate layer of a group of samples, and predict
    the samples classification.
    """
    if clasificador== "hierarchical":
        clf_predict = clf.fit_predict(intermediate_layer)
        plt.title("Hierarchical Clustering Dendrogram")
        # plot the top three levels of the dendrogram
        plot_dendrogram(clf, truncate_mode="level", p=3)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        #plt.show()
        plt.savefig(results_path_day + "dendogram.png")

    else:
        clf_predict= clf.predict(intermediate_layer)
        if clasificador=="perceptron":
            #print(clf_predict)
            clf_predict=clf_predict>0.5
            #print(clf_predict)
        if clasificador=="NN":
            clf_predict=clf_predict>0.5

    return clf_predict

def print_write_classifier_stats(ground_truth_train, labels_train, ground_truth_validate,labels_validate,mean_unstables, std_unstables, mean_stables, std_stables,domine,clasificador,activities=act_clf,extra_name=''):
    """
    Function that takes train and validation group stats and save it in a text file
    """
    lda_model_stats_train= BinaryModelStats(ground_truth_train,labels_train,mean_unstables, std_unstables, mean_stables, std_stables,domine)

    #json_stats = json.dumps(lda_model_stats_train.__dict__,cls=NumpyEncoder,indent=4)

    print(colored("Verdaderos positivos Train                  : " + str(lda_model_stats_train.tpos)           ,'green'))
    print(colored("Verdaderos negativos Train                  : " + str(lda_model_stats_train.tneg)           ,'green'))
    print(colored("Falsos positivos Train                      : " + str(lda_model_stats_train.fpos)           ,  'red'))
    print(colored("Falsos negativos Train                      : " + str(lda_model_stats_train.fneg)           ,  'red'))
    print(colored("Valor predictivo Positivo (Precision) Train : " + str(lda_model_stats_train.precision)  +"%",'green'))
    print(colored("Valor predictivo Negativo Train             : " + str(lda_model_stats_train.npv)        +"%",'green'))
    print(colored("Tasa de Falsos Positivos (Fall out) Train   : " + str(lda_model_stats_train.fpr)        +"%",  'red'))
    print(colored("Tasa de Falsos Negativos Train              : " + str(lda_model_stats_train.fnr)        +"%",  'red'))
    print(colored("Accuracy (aciertos totales) Train           : " + str(lda_model_stats_train.accuracy)   +"%",'green'))
    print(colored("Errores Train                               : " + str(lda_model_stats_train.error)      +"%",  'red'))
    print(colored("Sensitivity/Recall Score Train              : " + str(lda_model_stats_train.sensitivity)+"%",'green'))
    print(colored("Specificity Score Train                     : " + str(lda_model_stats_train.specificity)+"%",'green'))
    print(colored("False discovery rate Train                  : " + str(lda_model_stats_train.fdr)        +"%",'green'))
    print(colored("F_1 Score Train                             : " + str(lda_model_stats_train.f_1)        +"%",'green'))


    clasificador_basic_name = clasificador_name

    if not os.path.exists(results_train_path):   #Crear directorio de salida de datos
        os.makedirs(results_train_path)
        file1 = open(results_train_path+"resultados_"+extra_name+''.join(activities)+".txt","w")
    else:
        file1 = open(results_train_path+"resultados_"+extra_name+''.join(activities)+".txt","a")
    file1.write("Modelo de autoencoder utilizado " + autoencoder_name +"\n")
    file1.write("Modelo utilizado " + clasificador_basic_name +"\n")
    file1.write("Resultados de" + "+".join(activities) +  "Values\n")
    file1.write("Verdaderos positivos Train                  : " + str(lda_model_stats_train.tpos)        + "   ----> Inestable clasificados como Inestable\n" )
    file1.write("Verdaderos negativos Train                  : " + str(lda_model_stats_train.tneg)        + "   ----> Estable clasificados como Estable\n" )
    file1.write("Falsos positivos Train                      : " + str(lda_model_stats_train.fpos)        + "   ----> Estable clasificados como Inestable\n")
    file1.write("Falsos negativos Train                      : " + str(lda_model_stats_train.fneg)        + "   ----> Inestable clasificados como Estable\n")
    file1.write("Valor predictivo Positivo (Precision) Train : " + str(lda_model_stats_train.precision)   + "%\n")
    file1.write("Valor predictivo Negativo Train             : " + str(lda_model_stats_train.npv)         + "%\n")
    file1.write("Tasa de Falsos Positivos (Fall out) Train   : " + str(lda_model_stats_train.fpr)         + "%\n")
    file1.write("Tasa de Falsos Negativos Train              : " + str(lda_model_stats_train.fnr)         + "%\n")
    file1.write("Accuracy (aciertos totales) Train           : " + str(lda_model_stats_train.accuracy)    + "%\n")
    file1.write("Errores Train                               : " + str(lda_model_stats_train.error)       + "%\n")
    file1.write("Specificity Score Train                     : " + str(lda_model_stats_train.specificity) + "%\n")
    file1.write("F_1 Score Train                             : " + str(lda_model_stats_train.f_1)         + "%\n")
    file1.close()

    #file2 = open(model_path + clasificador_basic_name+ "_sere_train_summary.json","w")
    #file2.write(json_stats)
    #####################################################
    metrics_train = {
    "Verdaderos positivos Train": lda_model_stats_train.tpos,
    "Falsos positivos Train":lda_model_stats_train.fpos,
    "Falsos negativos Train":lda_model_stats_train.fneg,
    "Valor predictivo Positivo Train":lda_model_stats_train.precision,
    "Valor predictivo Negativo Train":lda_model_stats_train.npv,
    "Tasa de Falsos Positivos Train":lda_model_stats_train.fpr,
    "Tasa de Falsos Negativos Train":lda_model_stats_train.fnr,
    "Accuracy Train":lda_model_stats_train.accuracy,
    "Errores Train":lda_model_stats_train.error,
    "Specificity Score Train":lda_model_stats_train.specificity,
    "F_1 Score Train":lda_model_stats_train.f_1}

    wandb.log({"train": metrics_train})


    lda_model_stats_val= BinaryModelStats(ground_truth_validate,labels_validate,mean_unstables, std_unstables, mean_stables, std_stables,domine)

    json_stats = json.dumps(lda_model_stats_val.__dict__,cls=NumpyEncoder,indent=4)

    print(colored("Verdaderos positivos Val                  : " + str(lda_model_stats_val.tpos)           ,'green'))
    print(colored("Verdaderos negativos Val                  : " + str(lda_model_stats_val.tneg)           ,'green'))
    print(colored("Falsos positivos Val                      : " + str(lda_model_stats_val.fpos)           ,  'red'))
    print(colored("Falsos negativos Val                      : " + str(lda_model_stats_val.fneg)           ,  'red'))
    print(colored("Valor predictivo Positivo (Precision) Val : " + str(lda_model_stats_val.precision)  +"%",'green'))
    print(colored("Valor predictivo Negativo Val             : " + str(lda_model_stats_val.npv)        +"%",'green'))
    print(colored("Tasa de Falsos Positivos (Fall out) Val   : " + str(lda_model_stats_val.fpr)        +"%",  'red'))
    print(colored("Tasa de Falsos Negativos Val              : " + str(lda_model_stats_val.fnr)        +"%",  'red'))
    print(colored("Accuracy (aciertos totales) Val           : " + str(lda_model_stats_val.accuracy)   +"%",'green'))
    print(colored("Errores Val                               : " + str(lda_model_stats_val.error)      +"%",  'red'))
    print(colored("Sensitivity/Recall Score Val              : " + str(lda_model_stats_val.sensitivity)+"%",'green'))
    print(colored("Specificity Score Val                     : " + str(lda_model_stats_val.specificity)+"%",'green'))
    print(colored("False discovery rate Val                  : " + str(lda_model_stats_val.fdr)        +"%",'green'))
    print(colored("F_1 Score Val                             : " + str(lda_model_stats_val.f_1)        +"%",'green'))


    clasificador_basic_name = clasificador_name

    if not os.path.exists(results_train_path):   #Crear directorio de salida de datos
        os.makedirs(results_train_path)
        file1 = open(results_train_path+"resultados_"+extra_name+''.join(activities)+".txt","w")
    else:
        file1 = open(results_train_path+"resultados_"+extra_name+''.join(activities)+".txt","w")
    file1.write("Modelo de autoencoder utilizado " + autoencoder_name +"\n")
    file1.write("Modelo utilizado " + clasificador_basic_name +"\n")
    file1.write("Resultados de" + "+".join(activities) +  "Values\n")
    file1.write("Verdaderos positivos Val                  : " + str(lda_model_stats_val.tpos)        + "   ----> Inestable clasificados como Inestable\n" )
    file1.write("Verdaderos negativos Val                  : " + str(lda_model_stats_val.tneg)        + "   ----> Estable clasificados como Estable\n" )
    file1.write("Falsos positivos Val                      : " + str(lda_model_stats_val.fpos)        + "   ----> Estable clasificados como Inestable\n")
    file1.write("Falsos negativos Val                      : " + str(lda_model_stats_val.fneg)        + "   ----> Inestable clasificados como Estable\n")
    file1.write("Valor predictivo Positivo (Precision) Val : " + str(lda_model_stats_val.precision)   + "%\n")
    file1.write("Valor predictivo Negativo Val             : " + str(lda_model_stats_val.npv)         + "%\n")
    file1.write("Tasa de Falsos Positivos (Fall out) Val   : " + str(lda_model_stats_val.fpr)         + "%\n")
    file1.write("Tasa de Falsos Negativos Val              : " + str(lda_model_stats_val.fnr)         + "%\n")
    file1.write("Accuracy (aciertos totales) Val           : " + str(lda_model_stats_val.accuracy)    + "%\n")
    file1.write("Errores Val                               : " + str(lda_model_stats_val.error)       + "%\n")
    file1.write("Specificity Score Val                     : " + str(lda_model_stats_val.specificity) + "%\n")
    file1.write("F_1 Score Val                             : " + str(lda_model_stats_val.f_1)         + "%\n")
    file1.close()

    file2 = open(model_path_clf + clasificador_basic_name+ "_sere_train_summary.json","w")
    file2.write(json_stats)
    #####################################################
    metrics_val = {
    "Verdaderos positivos Val": lda_model_stats_val.tpos,
    "Falsos positivos Val":lda_model_stats_val.fpos,
    "Falsos negativos Val":lda_model_stats_val.fneg,
    "Valor predictivo Positivo Val":lda_model_stats_val.precision,
    "Valor predictivo Negativo Val":lda_model_stats_val.npv,
    "Tasa de Falsos Positivos Val":lda_model_stats_val.fpr,
    "Tasa de Falsos Negativos Val":lda_model_stats_val.fnr,
    "Accuracy Val":lda_model_stats_val.accuracy,
    "Errores Val":lda_model_stats_val.error,
    "Specificity Score Val":lda_model_stats_val.specificity,
    "F_1 Score Val":lda_model_stats_val.f_1}
    wandb.log({"val": metrics_val})



def plot_clf_transformation(clf,unstable_intermediate,stable_intermediate,ground_truth, labels_validate,activities,clasificador,controlgroup_intermediate=[],extra_name=''):
    """
    """
    stable_label = 0
    unstable_label = 1
    clf_basic_name = clasificador_name_creation(activities, clasificador)
    unstable_transformations = clf.transform(unstable_intermediate)
    stable_transformations   = clf.transform(stable_intermediate)

    mean_unstables = 0 if np.isnan(np.mean(unstable_transformations)) else np.mean(unstable_transformations)
    std_unstables  = 0 if np.isnan(np.std(unstable_transformations)) else np.std(unstable_transformations)
    mean_stables   = 0 if np.isnan(np.mean(stable_transformations)) else np.mean(stable_transformations)
    std_stables    = 0 if np.isnan(np.std(stable_transformations)) else np.std(stable_transformations)
    minimo = np.min([np.min(unstable_transformations),np.min(stable_transformations)])
    maximo = np.max([np.max(unstable_transformations),np.max(stable_transformations)])
    domine = np.linspace(minimo, maximo, 300)

    if np.size(controlgroup_intermediate):
        controlgroup_transformations = clf.transform(controlgroup_intermediate)
        mean_control = 0 if np.isnan(np.mean(controlgroup_transformations)) else np.mean(controlgroup_transformations)
        std_control  = 0 if np.isnan(np.std(controlgroup_transformations)) else np.std(controlgroup_transformations)
        minimo = np.max([np.max(controlgroup_transformations),minimo])
        maximo = np.max([np.max(controlgroup_transformations),maximo])
        domine = np.linspace(minimo, maximo, 300)
        estimated_pdf_control = stats.norm.pdf(domine, mean_control, std_control)
        estimated_pdf_control/=np.max(estimated_pdf_control)

    estimated_pdf_unstables = stats.norm.pdf(domine, mean_unstables, std_unstables)
    estimated_pdf_unstables/= np.max(estimated_pdf_unstables)
    estimated_pdf_stables   = stats.norm.pdf(domine, mean_stables, std_stables)
    estimated_pdf_stables  /= np.max(estimated_pdf_stables)

    plt.title('True values Histogram')
    plt.hist(unstable_transformations, range = (minimo,maximo),bins=100, alpha=0.4, label='Unstable')
    plt.hist(stable_transformations,   range = (minimo,maximo),bins=100, alpha=0.4, label='Stable')
    plt.plot(domine, estimated_pdf_unstables*25,label='Estimated unstables distribution')
    plt.plot(domine, estimated_pdf_stables*25,  label='Estimated stables distdistribution')
    plt.axvline(mean_unstables, color='b', linestyle='dashed', linewidth=1)
    plt.axvline(mean_stables,   color='y', linestyle='dashed', linewidth=1)
    if np.size(controlgroup_intermediate):
        plt.hist(controlgroup_transformations,    range = (minimo,maximo),bins=100, alpha=0.4, label='Control')
        plt.plot(domine, estimated_pdf_control*25,label='Estimated control group distribution')
        plt.axvline(mean_stables,   color='g',    linestyle='dashed', linewidth=1)
    plt.xlabel("Sere Index")
    plt.ylabel("Votes per bins")
    plt.legend(fontsize='x-small')
    plt.savefig(model_path_clf + clf_basic_name+ "_"+clasificador+"_hist.jpg")
    plt.close()
    #mlflow.log_artifact(model_path + clf_basic_name+"_"+clasificador+"_hist.jpg")


    plt.title('Estimated distributions over SERE index')
    plt.plot(domine, estimated_pdf_unstables*25,label='Estimated unstables distribution')
    plt.plot(domine, estimated_pdf_stables*25,  label='Estimated stables distdistribution')
    plt.axvline(mean_unstables, color='b', linestyle='dashed', linewidth=1)
    plt.axvline(mean_stables,   color='y', linestyle='dashed', linewidth=1)
    if np.size(controlgroup_intermediate):
        plt.plot(domine, estimated_pdf_control*25,label='Estimated control group distribution')
        plt.axvline(mean_stables,   color='g',    linestyle='dashed', linewidth=1)
    plt.xlabel("Sere Index")
    plt.ylabel("Votes per bins")
    plt.legend(fontsize='x-small')
    plt.savefig(model_path_clf + clf_basic_name+ "_stimated_distribution.jpg")
    plt.close()
#    mlflow.log_artifact(model_path + clf_basic_name+"_stimated_distribution.jpg")


    transformations = np.concatenate([stable_transformations,unstable_transformations],axis=0)

    transforms_labeled_unstable = transformations[labels_validate == unstable_label]
    transforms_labeled_stable   = transformations[labels_validate == stable_label]

    ## Clasificación grafica
    plt.title('Prediction Histogram')
    plt.hist(transforms_labeled_unstable, range = (minimo,maximo),bins=100, alpha=0.4, label='Unstables')
    plt.hist(transforms_labeled_stable,   range = (minimo,maximo),bins=100, alpha=0.4, label='Stables')
    plt.plot(domine, estimated_pdf_unstables*25,label='Estimated unstables distribution')
    plt.plot(domine, estimated_pdf_stables*25,  label='Estimated stables distdistribution')
    plt.axvline(mean_unstables, color='b', linestyle='dashed', linewidth=1)
    plt.axvline(mean_stables,   color='y', linestyle='dashed', linewidth=1)
    plt.xlabel("Sere Index")
    plt.ylabel("Votes per bins")
    plt.legend(fontsize='x-small')
    plt.savefig(model_path_clf + clf_basic_name+"_"+clasificador+"_hist_predict.jpg")
    plt.close()
#    mlflow.log_artifact(model_path + clf_basic_name+"_"+clasificador+"_hist_predict.jpg")

    # Unstables are positive (1) and Stables are negative(0)
    display_labels = ["Stables","Unstables"]
    X_test = np.concatenate([stable_intermediate,unstable_intermediate],axis=0)
    #plot_confusion_matrix(clf, X_test, ground_truth,display_labels=display_labels,normalize='true',cmap=plt.cm.Wistia)
    plt.savefig(model_path_clf + clf_basic_name+"_"+clasificador+"_confusion_matrix_normalized.jpg")
    plt.close()
#    mlflow.log_artifact(model_path + clf_basic_name+"_"+clasificador+"_confusion_matrix_normalized.jpg")

    #plot_confusion_matrix(clf, X_test, ground_truth,display_labels=display_labels,cmap=plt.cm.Wistia)
    plt.savefig(model_path_clf + clf_basic_name+"_"+clasificador+"_confusion_matrix.jpg")
    plt.close()
#    mlflow.log_artifact(model_path + clf_basic_name+"_"+clasificador+"_confusion_matrix.jpg")
    return mean_unstables, std_unstables, mean_stables, std_stables, [maximo,minimo]

def plot_dendrogram(model, **kwargs):
    """
    Function that plots the dendogram of the hierarchical classifier
    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

class ModelStats:
    """
    Statical values extracted from
    https://stackoverflow.com/a/43331484/16918860

    Function that takes validation group stats and save it in a text file
    """
    def __init__(self, ground_truth_validate, labels_validate, mean_unstables, std_unstables, mean_stables, std_stables, domine):
        # About AE and Lda
        self.lda_model_version = clasificador_model_version
        self.ae_model_version  = ae_model_version
        self.git_version_commit = git_version_commit

        self.train_date        = date_train
        self.ae_mode           = modo_ae
        self.extra             = extra

        # About used dataset
        #self.train_stables          = stable_train
        #self.train_unstables        = unstable_train
        #self.validation_stables     = stable_validation
        #self.validation_unstables   = stable_validation

        #About dataclass distribution
        self.mean_unstables = mean_unstables
        self.std_unstables  = std_unstables
        self.mean_stables   = mean_stables
        self.std_stables    = std_stables
        self.domine         = domine

        # True Negatives, False Positives, False Negative, True Positive
        self.conf_matrix = confusion_matrix(ground_truth_validate,labels_validate)
        self.fpos = self.conf_matrix.sum(axis=0) - np.diag(self.conf_matrix)
        self.fneg = self.conf_matrix.sum(axis=1) - np.diag(self.conf_matrix)
        self.tpos = np.diag(self.conf_matrix)
        self.tneg = self.conf_matrix.sum() - (self.fpos + self.fneg + self.tpos)

        # Sensitivity, hit rate, recall, or true positive rate
        self.sensitivity = self.tpos/(self.tpos+self.fneg)
        # Specificity or true negative rate
        self.specificity = self.tneg/(self.tneg+self.fpos)
        # Precision or positive predictive value
        self.precision = self.tpos/(self.tpos+self.fpos)
        # Negative predictive value
        self.npv = self.tneg/(self.tneg+self.fneg)
        # Fall out or false positive rate
        self.fpr = self.fpos/(self.fpos+self.tneg)
        # False negative rate
        self.fnr = self.fneg/(self.tpos+self.fneg)
        # False discovery rate
        self.fdr = self.fpos/(self.tpos+self.fpos)
        # Accuracy/Exactitud -> Overall accuracy. Mean percentage of good classified segments
        self.accuracy = (self.tpos+self.tneg)/(self.tpos+self.fpos+self.fneg+self.tneg)
        # Error. Mean percentage of bad classified segments
        self.error   = (self.fpos+self.fneg)/2
        # Score F_1, Harmonic mean
        self.f_1 = f1_score(ground_truth_validate,labels_validate)

class BinaryModelStats:
    """
    Statical values extracted from
    https://stackoverflow.com/a/43331484/16918860

    For binary output label. Just for stable and unstable label
    """
    def __init__(self, ground_truth_validate, labels_validate, mean_unstables, std_unstables, mean_stables, std_stables, domine):
        # About AE and Lda
        self.lda_model_version = clasificador_model_version
        self.ae_model_version  = ae_model_version
        self.git_version_commit = git_version_commit
        self.train_date        = date_train
        self.ae_mode           = modo_ae
        self.extra             = extra

        # About used dataset
        #self.train_stables          = stable_train
        #self.train_unstables        = unstable_train
        #self.validation_stables     = stable_validation
        #self.validation_unstables   = stable_validation

        #About dataclass distribution
        self.mean_unstables = mean_unstables
        self.std_unstables  = std_unstables
        self.mean_stables   = mean_stables
        self.std_stables    = std_stables
        self.domine         = domine
        #print("len ground truth validate ", len(ground_truth_validate))
        #print("len labels_validate ", len(labels_validate))
        # True Negatives, False Positives, False Negative, True Positive
        self.tneg,self.fpos,self.fneg,self.tpos = confusion_matrix(ground_truth_validate,labels_validate).ravel()
        # Sensitivity, hit rate, recall, or true positive rate
        self.sensitivity = 100*self.tpos/(self.tpos+self.fneg)
        # Specificity or true negative rate
        self.specificity = 100*self.tneg/(self.tneg+self.fpos)
        # Precision or positive predictive value
        self.precision   = 100*self.tpos/(self.tpos+self.fpos)
        # Negative predictive value
        self.npv = 100*self.tneg/(self.tneg+self.fneg)
        # Fall out or false positive rate
        self.fpr = 100*self.fpos/(self.fpos+self.tneg)
        # False negative rate
        self.fnr = 100*self.fneg/(self.tpos+self.fneg)
        # False discovery rate
        self.fdr = 100*self.fpos/(self.tpos+self.fpos)
        # Accuracy/Exactitud -> Overall accuracy. Mean percentage of good classified segments
        self.accuracy = 100*(self.tpos+self.tneg)/(self.tpos+self.fpos+self.fneg+self.tneg)
        # Error. Mean percentage of bad classified segments
        self.error    = 100*(self.fpos+self.fneg)/(self.fpos+self.fneg+self.tpos+self.tneg)
        # Score F_1, Harmonic mean
        self.f_1      = f1_score(ground_truth_validate,labels_validate)

