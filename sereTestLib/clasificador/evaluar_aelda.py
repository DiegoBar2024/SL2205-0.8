from sereTestLib.parameters import *
from sereTestLib.clasificador.extras import patient_group_aelda, clasificador_name_creation
from joblib import  load
import time
import pandas as pd
import json
import os

def evaluar_aelda(clasificador, autoencoder_model,id_muestra:int, result:results, clf_model_file ,activities = act_clf):
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
    # TODO: Ver porque no funciona usar directorio_muestra
    paramsV= {'data_dir' : dir_preprocessed_data_test,
                        'dim': inDim,
                        'batch_size': batch_size,
                        'shuffle': False,
                        'activities':activities,
                        'long_sample':result.long_sample}

    ## Cargo el modelo del clasificador ya entrenado
    clf_trained = load(model_path_clf + clf_model_file)

    result = classify_patient_aelda(id_muestra,clf_trained, autoencoder_model,clasificador,result,layer_name='Dense_encoder',**paramsV)
    #print(result['Caminando'])
    write_patient_info(id_muestra,activities,result,extra, clasificador=clasificador)


def classify_patient_aelda(patient_number,clf_model, modelo, clasificador, result:results,layer_name='Dense_encoder',**params):
    """
    Function that takes as input the patient number and the parameters for the data generator
    and returns the percentaje of stable and unstable samples.
    If the patient doesnt have samples for the activity, returns Nan
    """

    ## Defino la etiqueta de estable como 0
    stable_label = 0

    ## Defino la etiqueta de inestable como 1
    unstable_label = 1

    ## <<pat_intermediate>> contiene la representación reducida de los escalogramas correspondientes a un paciente
    pat_intermediate = patient_group_aelda([patient_number],modelo,layer_name, **params)

    ## En caso de que exista al menos un elemento en <<pat_intermediate>> entro al condicional <<if>>
    if pat_intermediate.any():

        ## En caso de que el clasificador sea del tipo "hierachical"
        if clasificador == "hierarchical":

            ## Determino la predicción del clasificador ante mi muestra de entrada
            pat_predictions = clf_model.fit_predict(pat_intermediate)

        else:

            pat_predictions = clf_model.predict(pat_intermediate)

            ## En caso de que el clasificador sea de tipo <<perceptron>>
            if clasificador == "perceptron":
                pat_predictions=pat_predictions>0.5
            if clasificador=="NN":
                pat_predictions=pat_predictions>0.5
        act_time = np.size(pat_predictions)*static_window_secs
        activity_str = ''.join(params['activities'])
        result.stable_percentage[activity_str]      = np.size(pat_predictions[pat_predictions==stable_label])*100/np.shape(pat_intermediate)[0]
        result.unstable_percentage[activity_str]    = np.size(pat_predictions[pat_predictions==unstable_label])*100/np.shape(pat_intermediate)[0]
        result.sample_stable_mean[activity_str]     = 0 #if np.isnan(np.mean(pat_evaluation[pat_predictions==stable_label])) else np.mean(pat_evaluation[pat_predictions==stable_label])
        result.sample_stable_std[activity_str]      = 0 #if np.isnan(np.std(pat_evaluation[pat_predictions==stable_label])) else np.std(pat_evaluation[pat_predictions==stable_label])
        result.sample_unstable_mean[activity_str]   = 0 #if np.isnan(np.mean(pat_evaluation[pat_predictions==unstable_label])) else np.mean(pat_evaluation[pat_predictions==unstable_label])
        result.sample_unstable_std[activity_str]    = 0 #if np.isnan(np.std(pat_evaluation[pat_predictions==unstable_label])) else np.std(pat_evaluation[pat_predictions==unstable_label])
        result.sample_mean[activity_str]            = 0 #if np.isnan(np.mean(pat_evaluation)) else np.mean(pat_evaluation)
        result.sample_std[activity_str]             = 0 #if np.isnan(np.std(pat_evaluation)) else np.std(pat_evaluation)
        result.activities_time_secs[activity_str]   = act_time
        result.activities_time_format[activity_str] = time.strftime('%H:%M:%S', time.gmtime(act_time))
        result.activities.append(activity_str)

        #clf_name =  clasificador_name_creation(params['activities'], clasificador)
        #print(model_path_clf  + clasificador_name+ "_sere_train_summary.json")
        model_info_f = open(model_path_clf  + clasificador_name+ "_sere_train_summary.json",)
        model_info_j = json.load(model_info_f)
        result.models_info[activity_str]= (model_info_j)
    return result

def write_patient_info(pat_number,activities,result:results,extra_name, clasificador):
    """
    Function that takes patient stats and save it in a text file
    """

    clf_basic_name = clasificador_name
    
    
    file1 = open(results_path+"Resultados_"+date+'_'+extra_name+''.join(activities)+".txt","a")
    file1.write("Resultados de" +' + '.join(activities)+  "Values\n")
    file1.write("Paciente " + str(pat_number)+"\n")
    file1.write("Modelo de autoencoder utilizado " + autoencoder_name +"\n")
    file1.write("Modelo de clasificador utilizado " + clf_basic_name +"\n")
    file1.write("Commit de código " + git_version_commit +"\n")
    #print(result.activities_time_format['Sentado'])
    print(result.stable_percentage)
    try:
        file1.write("Porcentaje de segmentos clasificados 'estable':   " + str(result.stable_percentage[''.join(activities)]) +"%\n")
        file1.write("Porcentaje de segmentos clasificados 'inestable': " + str(result.unstable_percentage[''.join(activities)]) +"%\n")
        file1.write("Tiempo de la actividad: " + result.activities_time_format[''.join(activities)] +"\n")
        file1.write("Tiempo inactivo: " + result.activities_time_format['Sentado'] +"\n")
        if result.stable_percentage[''.join(activities)]<result.unstable_percentage[''.join(activities)]:
            clasificacion=1
        else:
            clasificacion=0
    except:
        file1.write("La muestra no tiene muestras para la actividad:   " + ''.join(activities) +"%\n")
        clasificacion= "-"
    file1.write("\n")
    file1.close()
    if activities==["Caminando"]:
        if not os.path.exists(results_train_path+'clasificaciones.csv'):
            copy_csv(dir_etiquetas+'clasificaciones_antropometricos.csv', results_train_path+'clasificaciones.csv')
        data = pd.read_csv (results_train_path+'clasificaciones.csv')   
        encontre=False
        i=0
        nsamples=data.shape[0]
        while not encontre and nsamples>0: 
            if(data.loc[i, "sampleid"]==pat_number):
                data.loc[i, "Clasificacion"]= clasificacion
                data.loc[i, "stable"]=result.stable_percentage[''.join(activities)]
                data.loc[i, "unstable"]=result.unstable_percentage[''.join(activities)]
                if data.loc[i, "Clasificacion"] != data.loc[i, "Crit1"]:
                     data.loc[i, "No coincide"]= "*" 
                encontre=True
                data.to_csv(results_train_path+'clasificaciones.csv', index=False)

            i=i+1
            nsamples=nsamples-1
            
def copy_csv(filename, filenamecopy):
    """
    Auxiliar function to copy a csv file
    """
    df = pd.read_csv(filename)
    #print(df.head())
    #df=df[["sampleid", "Caida", "Auxiliar", "Inestabilidad", "Vertigos", "Sicofarmacos", "Canlitiasis, artrosis, etc", "Crit4","Clasificacion"]]
    df.to_csv(filenamecopy, index=False)

