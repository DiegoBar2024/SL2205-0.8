#%%
# from sereTestLib.webservice.consultas import muestras_institucion, obtener_json, return_resultado
# from sklearn.metrics import f1_score
# import json
import pandas as pd
import matplotlib.pyplot as plt
from sereTestLib.Preprocesamiento.preprocesamiento_train import *
from termcolor import colored
from sereTestLib.clasificador.entrenamiento_clasificador import BinaryModelStats
import wandb
import numpy as np
from sereTestLib.Inferencia import Inferencia

# #print(len(pacientes_institucion("Clinica Auditiva", mayor_65=True)))
# muestras=muestras_institucion("Clinica Auditiva", True)

# #armar datos
# dicts=[]
# for muestra in muestras:
#     print(muestra)
#     muestra_dict, path=obtener_json(muestra[0])
#     resultado=json.loads(return_resultado(muestra[0])['RESULTADOJSON'])
#     #print(resultado)
#     sere_index=resultado["sere_index"]
#     etiqueta=1 if (muestra_dict[1]["caida"]=='No' and muestra_dict[1]["inestabilidad"]=='No') else 0
#     dicts.append({
#         'numero_muestra':muestra[0],
#         'intervencion':muestra_dict[1]["intervencion_terapeutica"],
#         'cedula':muestra_dict[1]["cedula"],
#         'sere_index':sere_index,
#         'etiqueta':etiqueta,
#         'clasificacion_del_tecnico':muestra_dict[1]["clasificacion_del_tecnico"],
#         'fecha nacimiento':muestra_dict[1]["fecha_nac"],
#         'sicofarmacos':muestra_dict[1]["sicofarmacos"],
#         "nombre fármaco":muestra_dict[1]["nombre_farmaco"],
#         'autopercepcion de inestabilidad':muestra_dict[1]["inestabilidad"],
#         'caidas': muestra_dict[1]["caida"],
#         'nombre': muestra_dict[1]['nombre'],
#         'apellido':muestra_dict[1]['apellido']})
# df = pd.DataFrame(dicts)
# df.to_csv("pacientes.csv",index=False)

# dict_remap={ v: index[0] for index, v in np.ndenumerate(df["cedula"].unique()) }
# #print(dict_remap)
# df=df.replace({"cedula": dict_remap})
# #print(df)
df=pd.read_csv(dir_etiquetas+"pacientes_CA_etiquetados.csv")

run=wandb.init(project="SereTest-autoencoder",reinit=True,job_type="load ae")
modelo_artifact=run.use_artifact(autoencoder_name+':latest')
modelo_dir=modelo_artifact.download(model_path_ae)
run.finish()

run=wandb.init(project="SereTest-clasificador",reinit=True,job_type="load clf")
modelo_artifact=run.use_artifact(clasificador_name+':latest')
run.finish()
data = pd.read_csv ("pacientes_CA_etiquetados.csv")

inferir=data["numero_muestra"]
sere_index=[]
for i in inferir:
    result=Inferencia(i)
    print(result.stable_percentage["Caminando"])
    sere_index.append(result.stable_percentage["Caminando"])
data["sere_index"]=sere_index
df=data



print("Cantidad de muestras con audifonos: ",len(df[df['intervencion']=='Audifonos']))
print("Cantidad de muestras sin intevención: ",len(df[df['intervencion']=='Ninguno']))
print("Cantidad de pacientes", len(df["cedula"].unique()))


# bins=40
# plt.title("Sere Index según la intervención terapéutica")
# plt.hist(df[df['intervencion']=='Ninguno']["sere_index"],label='Ninguno', range=(0,100),bins=bins)
# plt.hist(df[df['intervencion']=='Audifonos']["sere_index"],label='Audifonos', range=(0,100),bins=bins)
# plt.axvline(df[df['intervencion']=='Ninguno']["sere_index"].mean(), linestyle='dotted', color ="k", label="Media Ninguno")
# plt.axvline(df[df['intervencion']=='Audifonos']["sere_index"].mean(), linestyle="dashed",  color ="k",label="Media Audifonos")
# plt.xlabel('Sere Index')
# plt.ylabel('Cantidad de muestras')
# plt.legend(title="Intervención Terapéutica")
# plt.show()

bins=40
plt.title("Sere Index según la etiqueta")
plt.hist(df[df['etiqueta']==1]["sere_index"],label='Estable', range=(0,100),bins=bins)
plt.hist(df[df['etiqueta']==0]["sere_index"],label='Inestable', range=(0,100),bins=bins)
plt.axvline(df[df['etiqueta']==1]["sere_index"].mean(), linestyle='dotted', color ="k", label="Media Estables")
plt.axvline(df[df['etiqueta']==0]["sere_index"].mean(), linestyle="dashed",  color ="k",label="Media Inestables")
plt.xlabel('Sere Index')
plt.ylabel('Cantidad de muestras')
plt.legend(title="Etiquetas")
plt.savefig("Clinica_Auditiva_modelo_nuevo")


# bins=40
# plt.title("Sere Index según la clasificación del técnico")
# plt.hist(df[df["clasificacion_del_tecnico"]=="Estable"]["sere_index"],label='Estable', range=(0,100),bins=bins)
# plt.hist(df[df["clasificacion_del_tecnico"]!="Estable"]["sere_index"],label='Inestable', range=(0,100),bins=bins)
# plt.axvline(df[df["clasificacion_del_tecnico"]=="Estable"]["sere_index"].mean(), linestyle='dotted', color ="k", label="Media Estables")
# plt.axvline(df[df["clasificacion_del_tecnico"]!="Estable"]["sere_index"].mean(), linestyle="dashed",  color ="k",label="Media Inestables")
# plt.xlabel('Sere Index')
# plt.ylabel('Cantidad de muestras')
# plt.legend(title="Etiquetas")
# plt.show()


# bins=40
# plt.title("Sere Index según el paciente")
# plt.scatter(df[df['intervencion']=='Ninguno']["cedula"],df[df['intervencion']=='Ninguno']["sere_index"],label='Ninguno')
# plt.scatter(df[df['intervencion']=='Audifonos']["cedula"],df[df['intervencion']=='Audifonos']["sere_index"],label='Audifonos')
# plt.xlabel('Paciente')
# plt.ylabel('Sere Index')
# plt.legend(title="Intervención Terapéutica")
# plt.show()


# bins=5
# plt.title("Etiqueta utilizada vs Etiqueta HS")
# plt.hist(df[df["etiqueta"]==1]["Clasificacion HS"],label='Estable', alpha=0.5,range=(1,5),bins=bins)
# plt.hist(df[df["etiqueta"]==0]["Clasificacion HS"],label='Inestable',alpha=0.5, range=(1,5),bins=bins)
# plt.xlabel('Etiqueta HS')
# plt.ylabel('Cantidad de muestras')
# plt.legend(title="Etiquetas")
# plt.show()

# bins=40
# plt.title("Sere Index según la etiqueta de HS")
# plt.hist(df[df['Clasificacion HS']==1]["sere_index"],label='1', range=(0,100),bins=bins)
# plt.hist(df[df['Clasificacion HS']==2]["sere_index"],label='2', range=(0,100),bins=bins)
# plt.hist(df[df['Clasificacion HS']==3]["sere_index"],label='3', range=(0,100),bins=bins)

# plt.xlabel('Sere Index')
# plt.ylabel('Cantidad de muestras')
# plt.legend(title="Etiquetas HS")
# plt.show()

lda_model_stats_train= BinaryModelStats(df['etiqueta'],df['sere_index']>49,0, 0, 0, 0,0)


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


