#%%
from termcolor import colored
from numpyencoder import NumpyEncoder
import os
import pandas as pd
import matplotlib.pyplot as plt


from sereTestLib.Inferencia import Inferencia
from sereTestLib.parameters import *
from sereTestLib.clasificador.entrenamiento_clasificador import BinaryModelStats



def inferir_todos():
    inferir=list(range(250,251))+list(range(252,297))+list(range(298,313))

    # print(inferir)
    # for i in inferir:
    #     Inferencia(i, plot=False)
    data = pd.read_csv (static_path+'clasificaciones.csv')
   
    #write_group_info(data, "bps")
    plots(data)
   
def plots(data):
    #Sere Index según la etiqueta
    #Sere Index según TUG
    #TUG vs etiqueta
    data["tug"]=pd.to_timedelta(data["tug"])
    print(data["tug"] / pd.Timedelta(seconds=1))
    data["tug"]=  data["tug"] / pd.Timedelta(seconds=1)
    data_val=data[(data["grupo ae"]!="ae train")& (data["grupo clf"]!="train clf")].copy()
    df=data[["Clasificacion", "stable","sampleid","tug"]]
    df_val=data_val[["Clasificacion", "stable","sampleid","tug"]]
    #print(df_val)
    bins=50
    #print(df_val[df_val['Clasificacion']==0.0]["stable"])
    #print(df_val[df_val['Clasificacion']==1.0]["stable"])
    plt.title("Sere Index según la etiqueta")
    plt.hist(df_val[df_val['Clasificacion']==0.0]["stable"],label='Estable', range=(0,100),bins=bins)
    plt.hist(df_val[df_val['Clasificacion']==1.0]["stable"],label='Inestable', range=(0,100),bins=bins)
    plt.axvline(df_val[df_val['Clasificacion']==0.0]["stable"].mean(), linestyle='dotted', color ="k", label="Media Estables")
    plt.axvline(df_val[df_val['Clasificacion']==1.0]["stable"].mean(), linestyle="dashed",  color ="k",label="Media Inestables")
    plt.xlabel('Sere Index')
    plt.ylabel('Cantidad de muestras')
    plt.legend(title="Etiquetas")
    plt.show()

    bins=50
    #print(df_val[df_val['Clasificacion']==0.0]["stable"])
    #print(df_val[df_val['Clasificacion']==1.0]["stable"])
    plt.title("TUG según la etiqueta")
    plt.hist(df[df['Clasificacion']==0.0]["tug"],label='Estable',bins=bins)
    plt.hist(df[df['Clasificacion']==1.0]["tug"],label='Inestable', bins=bins)
    plt.axvline(df[df['Clasificacion']==0.0]["tug"].mean(), linestyle='dotted', color ="k", label="Media Estables")
    plt.axvline(df[df['Clasificacion']==1.0]["tug"].mean(), linestyle="dashed",  color ="k",label="Media Inestables")
    plt.xlabel('TUG')
    plt.ylabel('Cantidad de muestras')
    plt.legend(title="Etiquetas")
    plt.show()

    plt.title("Sere Index según la TUG")
    plt.hist(df_val[df_val['tug']<12]["stable"],label='Tug Estable', range=(0,100),bins=bins)
    plt.hist(df_val[df_val['tug']>=12]["stable"],label='Tug Inestable', range=(0,100),bins=bins)
    plt.axvline(df_val[df_val['tug']<12]["stable"].mean(), linestyle='dotted', color ="k", label="Media Tug Estables")
    plt.axvline(df_val[df_val[df_val['tug']>=12]==1.0]["stable"].mean(), linestyle="dashed",  color ="k",label="Media Tug Inestables")
    plt.xlabel('Sere Index')
    plt.ylabel('Cantidad de muestras')
    plt.legend(title="Tug (umbral=12)")
    plt.show()




def write_group_info(data, grupo):
    labels=data["Clasificacion"]
    ground_truth= data["Crit1"]
    print(labels)
    print(ground_truth)


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


    if not os.path.exists(results_train_path):   #Crear directorio de salida de datos
        os.makedirs(results_train_path)

        file1=open(results_path+"Resultados_Generales_"+clasificador+'_'+date+'_'+extra+".txt","w")
    else:
        file1 = open(results_path+"Resultados_Generales_"+clasificador+'_'+date+'_'+extra+".txt","a")
    print("Guardado en:", results_path+"Resultados_Generales_"+clasificador+'_'+date+'_'+extra+".txt")

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

    #%%