import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/clasificador')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
from parameters import *
from joblib import  load
import time
import pandas as pd
import json
import os
from Extras_CLAS import *

def evaluar_aelda(nombre_clasificador, nombre_autoencoder, ruta_clasificador, clasificador, autoencoder_model, id_muestra : int, result : results, clf_model_file, activities = act_clf):
    """
    Function that makes the prediction for the sample id.

    Parameters:
    -----------
    clasificador: str
        classification method
    autoencoder_model: 
        trained autoencoder model
    id_muestra: int
        sample id
    clf_model_file: 
        trained classifier model
    activities: list
        List of activities.
    
    """
    ## Especifico los parámetros de la inferencia
    paramsV = {'data_dir' : dir_escalogramas_nuevo_test,
                        'dim': inDim,
                        'batch_size': batch_size,
                        'shuffle': False,
                        'activities': activities,
                        'long_sample': result.long_sample}

    ## Cargo el modelo del clasificador ya entrenado según la ruta del clasificador
    clf_trained = load(ruta_clasificador + clf_model_file)

    ## Aplico el clasificador correspondiente a la representación latente de 256 características del paciente
    result = classify_patient_aelda(id_muestra, clf_trained, autoencoder_model, clasificador, result, layer_name = 'Dense_encoder', **paramsV)

    ## Escribo la información del paciente en el fichero correspondiente
    write_patient_info(nombre_clasificador, nombre_autoencoder, id_muestra, activities, result, extra, clasificador = clasificador)

def classify_patient_aelda(patient_number, clf_model, modelo, clasificador, result:results, layer_name='Dense_encoder',**params):
    """
    Function that takes as input the patient number and the parameters for the data generator
    and returns the percentaje of stable and unstable samples.
    If the patient doesnt have samples for the activity, returns Nan
    """
    
    ## Defino la etiqueta de estable como 0
    stable_label = 0

    ## Defino la etiqueta de inestable como 1
    unstable_label = 1

    ## <<pat_intermediate>> va a ser una matriz cuyas filas y columnas se encuentran definidas del siguiente modo:
    ## La i-ésima fila va a representar al i-ésimo segmento
    ## La j-ésima columna va a representar la j-ésima caracteristica
    pat_intermediate = patient_group_aelda([patient_number], modelo, layer_name, **params)

    ## En caso de que exista al menos un elemento en <<pat_intermediate>> entro al condicional <<if>>
    if pat_intermediate.any():

        ## En caso de que el clasificador sea del tipo "hierachical"
        if clasificador == "hierarchical":

            ## Determino la predicción del clasificador ante mi muestra de entrada
            pat_predictions = clf_model.fit_predict(pat_intermediate)

        ## En caso de que el clasificador no sea "hierarchical"
        else:

            ## Determino la predicción del clasificador ante mi muestra de entrada
            pat_predictions = clf_model.predict(pat_intermediate)

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

        ## <<activity_str>> va a contener una cadena con el nombre de la actividad que se está estudiando
        ## Por ejemplo, para el caso que tenemos de la marcha, se obtiene <<activity_str>> = 'Caminando'
        activity_str = ''.join(params['activities'])
        
        ## Obtengo el procentaje de escalogramas estables correspondientes al paciente
        result.stable_percentage[activity_str] = np.size(pat_predictions[pat_predictions == stable_label]) * 100 / np.shape(pat_intermediate)[0]
        
        ## Obtengo el porcentaje de escalogramas inestables correspondientes al paciente
        result.unstable_percentage[activity_str] = np.size(pat_predictions[pat_predictions == unstable_label]) * 100 / np.shape(pat_intermediate)[0]
        
        ## Obtengo la cadena de actividad y se la asocio al resultado
        result.activities.append(activity_str)

    ## Retorno el resultado de la inferencia
    return result

def write_patient_info(nombre_clasificador, nombre_autoencoder, pat_number, activities, result:results, extra_name, clasificador):
    """
    Function that takes patient stats and save it in a text file
    """

    ## Especifico la ruta del directorio en el cual voy a almacenar los resultados obtenidos para el paciente dado
    ruta_results = results_path + "/S" + str(pat_number) + "/"

    ## Especifico la ruta completa del archivo de texto para el cual se almacenarán los resultados de la inferencia del paciente
    ruta_inferencia = ruta_results + "Resultados" + ".txt"

    ## En caso de que la ruta correspondiente al paciente no exista
    if not os.path.exists(ruta_results):

        ## Creo el directorio correspondiente en donde voy a guardar los resultados de la inferencia del paciente
        os.makedirs(ruta_results)

    ## En caso de que la ruta correspondiente al archivo de texto de inferencia ya exista
    if os.path.exists(ruta_inferencia):

        ## Borro el archivo de texto con la inferencia anterior para poder escribir uno nuevo
        os.remove(ruta_inferencia)

    ## Abro el archivo sobre el cual voy a escribir
    file1 = open(ruta_inferencia, "a")

    ## Escritura de los resultados de la inferencia
    file1.write("Actividad considerada: {}\n".format(''.join(activities)))

    ## Escritura del número identificador del paciente
    file1.write("Número de paciente: " + str(pat_number) + "\n")

    ## Escritura del nombre del modelo de autoencoder utilizado
    file1.write("Modelo de autoencoder utilizado: " + nombre_autoencoder + "\n")

    ## Escritura del nombre del modelo de clasificador utilizado
    file1.write("Modelo de clasificador utilizado: " + nombre_clasificador + "\n")
    
    ## En caso de que existan muestras que me permitan hacer la inferencia
    try:
        
        ## Escribo el porcentaje de los segmentos que son clasificados como 'estable'
        file1.write("Porcentaje de segmentos clasificados 'estable': " + str(result.stable_percentage[''.join(activities)]) +"%\n")
        
        ## Escribo el porcentaje de los segmentos que son clasificados como 'inestable'
        file1.write("Porcentaje de segmentos clasificados 'inestable': " + str(result.unstable_percentage[''.join(activities)]) +"%\n")
    
    ## En caso de que no haya muestras para poder hacer la clasificación, arrojo una excepción
    except:

        ## Imprimo el mensaje de aviso correspondiente
        file1.write("La muestra no tiene muestras para la actividad:   " + ''.join(activities) + "%\n")

    ## Escribo un salto de línea en el archivo
    file1.write("\n")

    ## Cierro el archivo de texto
    file1.close()