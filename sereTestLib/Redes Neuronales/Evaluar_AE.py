import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/clasificador')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
from parameters import *
from sereTestLib.clasificador.extras import patient_group_aelda, clasificador_name_creation
from joblib import  load
import time
import pandas as pd
import json
import os

def evaluar_aelda(clasificador, autoencoder_model, id_muestra : int, result : results, clf_model_file, activities = act_clf):
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
    paramsV= {'data_dir' : dir_escalogramas_nuevo_test,
                        'dim': inDim,
                        'batch_size': batch_size,
                        'shuffle': False,
                        'activities': activities,
                        'long_sample': result.long_sample}

    ## Cargo el modelo del clasificador ya entrenado según la ruta del clasificador
    clf_trained = load(model_path_clf + clf_model_file)

    ## Aplico el clasificador correspondiente a la representación latente de 256 características del paciente
    result = classify_patient_aelda(id_muestra, clf_trained, autoencoder_model, clasificador, result, layer_name = 'Dense_encoder', **paramsV)

    ## Escribo la información del paciente en el fichero correspondiente
    write_patient_info(id_muestra, activities, result, extra, clasificador = clasificador)

def classify_patient_aelda(patient_number, clf_model, modelo, clasificador, result:results,layer_name='Dense_encoder',**params):
    """
    Function that takes as input the patient number and the parameters for the data generator
    and returns the percentaje of stable and unstable samples.
    If the patient doesnt have samples for the activity, returns Nan
    """

    ## <<pat_intermediate>> contiene la representación reducida de los escalogramas correspondientes a un paciente
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
            ## Por defecto el clasificador utilizado por la empresa es un perceptrón, de modo que entraría acá
            if clasificador == "perceptron":

                ## En caso que el valor de la predicción numérica sea mayor a 0.5, asigno la variable <<pat_predictions>> a True (paciente inestable)
                ## En caso que el valor de la predicción numérica sea menor a 0.5, asigno la variable <<pat_predictions>> a False (paciente estable)
                pat_predictions = pat_predictions > 0.5

            ## En caso de que el clasificador sea una red neuronal
            if clasificador == "NN":

                ## En caso que el valor de la predicción numérica sea mayor a 0.5, asigno la variable <<pat_predictions>> a True (paciente inestable)
                ## En caso que el valor de la predicción numérica sea menor a 0.5, asigno la variable <<pat_predictions>> a False (paciente estable)
                pat_predictions = pat_predictions > 0.5

    ## Retorno el resultado de la inferencia
    return result

def write_patient_info(pat_number, activities, result:results, extra_name, clasificador):
    """
    Function that takes patient stats and save it in a text file
    """

    ## Obtengo el nombre del clasificador
    clf_basic_name = clasificador_name

    ## Abro el archivo sobre el cual voy a escribir
    file1 = open(results_path + "Resultados_" + date + '_' + extra_name + ''.join(activities) + ".txt", "a")

    ## Escritura de los resultados de la inferencia
    file1.write("Resultados de" +' + '.join(activities)+  "Values\n")
    file1.write("Paciente " + str(pat_number)+ "\n")
    file1.write("Modelo de autoencoder utilizado " + autoencoder_name +"\n")
    file1.write("Modelo de clasificador utilizado " + clf_basic_name +"\n")
    file1.write("Commit de código " + git_version_commit +"\n")
    try:
        file1.write("Porcentaje de segmentos clasificados 'estable':   " + str(result.stable_percentage[''.join(activities)]) +"%\n")
        file1.write("Porcentaje de segmentos clasificados 'inestable': " + str(result.unstable_percentage[''.join(activities)]) +"%\n")
        file1.write("Tiempo de la actividad: " + result.activities_time_format[''.join(activities)] +"\n")
        file1.write("Tiempo inactivo: " + result.activities_time_format['Sentado'] +"\n")
        if result.stable_percentage[''.join(activities)]<result.unstable_percentage[''.join(activities)]:
            clasificacion = 1
        else:
            clasificacion = 0
    except:
        file1.write("La muestra no tiene muestras para la actividad:   " + ''.join(activities) +"%\n")
        clasificacion= "-"
    file1.write("\n")
    file1.close()
    if activities == ["Caminando"]:
        if not os.path.exists(results_train_path+'clasificaciones.csv'):
            copy_csv(dir_etiquetas+'clasificaciones_antropometricos.csv', results_train_path+'clasificaciones.csv')
        data = pd.read_csv (results_train_path+'clasificaciones.csv')   
        encontre=False
        i = 0
        nsamples=data.shape[0]
        while not encontre and nsamples>0: 
            if(data.loc[i, "sampleid"]==pat_number):
                data.loc[i, "Clasificacion"]= clasificacion
                data.loc[i, "stable"]=result.stable_percentage[''.join(activities)]
                data.loc[i, "unstable"]=result.unstable_percentage[''.join(activities)]
                if data.loc[i, "Clasificacion"] != data.loc[i, "Crit1"]:
                    data.loc[i, "No coincide"]= "*" 
                encontre = True
                data.to_csv(results_train_path+'clasificaciones.csv', index=False)

            i = i + 1
            nsamples = nsamples - 1
            
def copy_csv(filename, filenamecopy):
    """
    Auxiliar function to copy a csv file
    """
    df = pd.read_csv(filename)
    #print(df.head())
    #df=df[["sampleid", "Caida", "Auxiliar", "Inestabilidad", "Vertigos", "Sicofarmacos", "Canlitiasis, artrosis, etc", "Crit4","Clasificacion"]]
    df.to_csv(filenamecopy, index=False)