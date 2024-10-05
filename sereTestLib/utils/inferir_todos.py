import pandas as pd
import numpy as np
import json
from termcolor import colored
from numpyencoder import NumpyEncoder
import os
import wandb
import matplotlib.pyplot as plt

from sereTestLib.Preprocesamiento.Preprocesamiento import Preprocesamiento

from sereTestLib.Inferencia import Inferencia
from sereTestLib.utils.ingesta_etiquetas import ingesta_etiquetas_concat
from sereTestLib.parameters import *
from sereTestLib.clasificador.entrenamiento_clasificador import BinaryModelStats
from sereTestLib.clasificador.extras import clasificador_name_creation



def inferir_todos():

    x_estables_train, x_inestables_train,x_estables_val,x_inestables_val =ingesta_etiquetas_concat()
    inferir=np.unique(np.concatenate((x_estables_train, x_inestables_train,x_estables_val,x_inestables_val), axis=None))
    #samples=list(range(250,251))+list(range(252,297))+list(range(298,314))
    # run=wandb.init(project="SereTest-autoencoder",reinit=True,job_type="load ae")

    # modelo_artifact=run.use_artifact(autoencoder_name+':latest')
    # modelo_dir=modelo_artifact.download(model_path_ae)
    # run.finish()

    # run=wandb.init(project="SereTest-clasificador",reinit=True,job_type="load clf")
    # modelo_artifact=run.use_artifact(clasificador_name+':latest')
    # run.finish()

    for i in inferir:
       Inferencia(i)
    Preprocesamiento([], [], upload_wandb=True)

    data = pd.read_csv (static_path +'clasificaciones.csv')
    data_clf_train=data[(data["grupo"]=="train")]
    write_group_info(data_clf_train, "train")
    data_clf_val=data[(data["grupo"]=="val")]
    write_group_info(data_clf_val, "val")



def write_group_info(data, grupo):
    labels=data["Clasificacion"]
    ground_truth= data["Crit1"]
    print(len(data))

    bins=40
    plt.title("Sere Index según la etiqueta")
    plt.hist(data[data['Crit1']==0]["stable"],label='Estable', range=(0,100),bins=bins)
    plt.hist(data[data['Crit1']==1]["stable"],label='Inestable', range=(0,100),bins=bins)
    plt.axvline((data[data['Crit1']==0]["stable"]).mean(), linestyle='dotted', color ="k", label="Media Estables")
    plt.axvline((data[data['Crit1']==1]["stable"]).mean(), linestyle="dashed",  color ="k",label="Media Inestables")
    plt.xlabel('Sere Index')
    plt.ylabel('Cantidad de muestras')
    plt.legend(title="Etiquetas")
    plt.savefig("Histograma_"+ grupo+".png")
    plt.close()

    lda_model_stats_train= BinaryModelStats(ground_truth,labels,0, 0, 0, 0,0)

    print(grupo)

    print(colored("Verdaderos positivos                   : " + str(lda_model_stats_train.tpos)           ,'green'))
    print(colored("Verdaderos negativos                   : " + str(lda_model_stats_train.tneg)           ,'green'))
    print(colored("Falsos positivos                       : " + str(lda_model_stats_train.fpos)           ,  'red'))
    print(colored("Falsos negativos                       : " + str(lda_model_stats_train.fneg)           ,  'red'))
    print(colored("Valor predictivo Positivo (Precision)  : " + str(lda_model_stats_train.precision)  +"%",'green'))
    print(colored("Valor predictivo Negativo              : " + str(lda_model_stats_train.npv)        +"%",'green'))
    print(colored("Tasa de Falsos Positivos (Fall out)    : " + str(lda_model_stats_train.fpr)        +"%",  'red'))
    print(colored("Tasa de Falsos Negativos               : " + str(lda_model_stats_train.fnr)        +"%",  'red'))
    print(colored("Accuracy (aciertos totales)            : " + str(lda_model_stats_train.accuracy)   +"%",'green'))
    print(colored("Errores                                : " + str(lda_model_stats_train.error)      +"%",  'red'))
    print(colored("Sensitivity/Recall Score               : " + str(lda_model_stats_train.sensitivity)+"%",'green'))
    print(colored("Specificity Score                      : " + str(lda_model_stats_train.specificity)+"%",'green'))
    print(colored("False discovery rate                   : " + str(lda_model_stats_train.fdr)        +"%",'green'))
    print(colored("F_1 Score                              : " + str(lda_model_stats_train.f_1)        +"%",'green'))
    #autoencoder_name = autoencoder_model_name_creation(act_ae)


    #clasificador_basic_name = clasificador_name_creation(act_clf, clasificador)

    if not os.path.exists(results_train_path):   #Crear directorio de salida de datos
        os.makedirs(results_train_path)

        file1=open(results_path+"Resultados_Generales_"+clasificador+'_'+date+'_'+extra+".txt","w")
    else:
        file1 = open(results_path+"Resultados_Generales_"+clasificador+'_'+date+'_'+extra+".txt","a")
    print("Guardado en:", results_path+"Resultados_Generales_"+clasificador+'_'+date+'_'+extra+".txt")
    file1.write("Modelo de autoencoder utilizado " + autoencoder_name +"\n")
    #file1.write("Modelo utilizado " + clasificador_basic_name +"\n")
    file1.write("Resultados de " + str(grupo) + "\n")
    file1.write("Verdaderos positivos                  : " + str(lda_model_stats_train.tpos)        + "   ----> Inestable clasificados como Inestable\n" )
    file1.write("Verdaderos negativos                  : " + str(lda_model_stats_train.tneg)        + "   ----> Estable clasificados como Estable\n" )
    file1.write("Falsos positivos                      : " + str(lda_model_stats_train.fpos)        + "   ----> Estable clasificados como Inestable\n")
    file1.write("Falsos negativos                      : " + str(lda_model_stats_train.fneg)        + "   ----> Inestable clasificados como Estable\n")
    file1.write("Valor predictivo Positivo (Precision) : " + str(lda_model_stats_train.precision)   + "%\n")
    file1.write("Valor predictivo Negativo             : " + str(lda_model_stats_train.npv)         + "%\n")
    file1.write("Tasa de Falsos Positivos (Fall out)   : " + str(lda_model_stats_train.fpr)         + "%\n")
    file1.write("Tasa de Falsos Negativos              : " + str(lda_model_stats_train.fnr)         + "%\n")
    file1.write("Accuracy (aciertos totales)           : " + str(lda_model_stats_train.accuracy)    + "%\n")
    file1.write("Errores                               : " + str(lda_model_stats_train.error)       + "%\n")
    file1.write("Specificity Score                     : " + str(lda_model_stats_train.specificity) + "%\n")
    file1.write("F_1 Score                             : " + str(lda_model_stats_train.f_1)         + "%\n")
    file1.close()




if __name__== '__main__':

    inferir_todos()