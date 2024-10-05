from sereTestLib.Preprocesamiento.Preprocesamiento_para_inferir import Preprocesamiento_para_inferir
from sereTestLib.parameters  import *
from sereTestLib.autoencoder.ae_train_save_model import  ms_ssim, ae_load_model
from sereTestLib.clasificador.evaluar_aelda import evaluar_aelda
from sereTestLib.autoencoder.plot_testgroup_scalograms import plot_testgroup_scalograms
from sereTestLib.autoencoder.ae_train_save_model import ae_train_save_model,  ssim_loss
from sereTestLib.utils.detectar_errores import tiene_carpeta_vacia
from tensorflow import keras
import numpy as np
import time
import os
from scipy.stats import hmean
import json

def Inferencia(sample_id, result: results = results(), plot=False):
    """
    Function that classify a sample as stable or unstable.
    Parameters:
    sample id: int
    plot: bool
        If its true it plots original and reconstructed scalograms. Defaults to False
    """

    print("Paciente: ", sample_id)
    result.sample_id = sample_id
    result.long_sample = long_sample

    if long_sample:
        directorio_muestra = '/L' + str(sample_id) + "/"
    else:
        directorio_muestra = '/S' + str(sample_id) + "/"


    # se fija si la muestra tiene carpetas vacías en el preprocesado y
    # en ese caso la manda a preprocesar denuevo
    if tiene_carpeta_vacia(sample_id, [], [sample_id], only_preprocessed_data_folder=check_only_if_preprocessed_data_is_empty):
        print("Preprocesamiento")
        Preprocesamiento_para_inferir(val_test=[sample_id], upload_wandb=False)

    #run=wandb.init(project="SereTest-autoencoder",reinit=True,job_type="load ae")
    #modelo_artifact=run.use_artifact(autoencoder_name+':latest')
    #modelo_dir=modelo_artifact.download(model_path_ae)
    #run.finish()
    modelo_dir=model_path_ae
    #print(modelo_dir)
    modelo_autoencoder=ae_load_model(modelo_dir)
   # print(modelo_autoencoder.summary())

    if not os.path.exists(results_path_day):
        os.makedirs(results_path_day)
    if not os.path.exists(results_path):  # Crear directorio de salida de datos
        os.makedirs(results_path)


    if plot==True:
        print("Ploteo de escalogramas")
        plot_testgroup_scalograms([sample_id],modelo_autoencoder,cant_muestras,result,num_epochs,group=plot_group,is_long_sample=long_sample)


    clf_name = clasificador_name + '.joblib'

    files = os.listdir(dir_preprocessed_data_test+directorio_muestra)
    actividad = dict_actividades.get('Parado')

    segmentos = [file for file in files if file.startswith(tuple(actividad))]

    act_time = np.size(segmentos)*static_window_secs
    result.activities_time_secs["Parado"] = act_time
    result.activities_time_format["Parado"] = time.strftime(
        '%H:%M:%S', time.gmtime(act_time))
    actividad = dict_actividades.get('Sentado')
    segmentos = [file for file in files if file.startswith(tuple(actividad))]
    act_time = np.size(segmentos)*static_window_secs
    result.activities_time_secs["Sentado"] = act_time
    result.activities_time_format["Sentado"] = time.strftime(
        '%H:%M:%S', time.gmtime(act_time))
    result.activities_time_secs["No Identificado"] = 0
    result.activities_time_format["No Identificado"] = time.strftime(
        '%H:%M:%S', time.gmtime(0))

    result.activities_time_secs["Caminando"] = 0
    result.activities_time_format["Caminando"] = time.strftime(
        '%H:%M:%S', time.gmtime(0))
    result.ae_name=autoencoder_name
    result.clf_name=clf_name
    evaluar_aelda(clasificador, modelo_autoencoder, sample_id, result,  clf_model_file=clf_name)
    stable_per = np.asfarray(list(result.stable_percentage.values()))
    result.sere_index = hmean(stable_per)
    nroMuestra = result.sample_id
    medico = result.doctor
    paciente = result.patient
    resultadoJson = result.__dict__
    #print(result.activities_time_secs)

    print(result.activities_time_secs["Caminando"])
    respBackend = {}
    respBackend['nroMuestra'] = nroMuestra
    respBackend['medico'] = medico
    respBackend['paciente'] = paciente
#    respBackend['resultadoJson'] = json.dumps(resultadoJson)
    #resp = requests.post(
    #url_response, json=json.loads(json.dumps(respBackend)))
   # print(result.stable_percentage["".join("Caminando")])
    return result
if __name__== '__main__':
    i=105
    result=Inferencia(i,plot=True)
