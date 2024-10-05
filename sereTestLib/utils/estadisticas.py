
#%%
from sereTestLib.autoencoder.ae_train_save_model import  autoencoder_model_name_creation
from tensorflow import keras
from sereTestLib.parameters import *
from joblib import  load
from sereTestLib.clasificador.extras import patient_group_aelda, clasificador_name_creation
from sereTestLib.autoencoder.DataClassAuto import DataGeneratorAuto
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import os
import pylab as py
from sereTestLib.utils.ingesta_etiquetas import ingesta_etiquetas
import pandas as pd

plot_path=results_path + "Estadisticas/"
if not os.path.exists(plot_path):   #TODO: Agregar en un log del usuario
    os.makedirs(plot_path)

params= {'data_dir' : scalogram_path,
                            'dim': inDim,
                            'batch_size': 1,
                            'shuffle': False,
                            'activities':act_ae}

autoencoder_name = autoencoder_model_name_creation(act_ae) + '.h5'
modelo_autoencoder = keras.models.load_model(model_path + autoencoder_name)
modelo_autoencoder.summary()


paramsV= {'data_dir' : scalogram_path,
                           'dim': inDim,
                           'batch_size': batch_size,
                           'shuffle': False,
                           'activities':act_ae,
                           'long_sample':False}

clf_name = clasificador_name_creation(act_clf, clasificador)+'.joblib'
clf_model = load(model_path + clf_name)


data = pd.read_csv ("/home/sere/Dropbox/SereNoTocar/sereData/results/20220510/20220510_1155_dense_base_classificator/clasificaciones.csv")   
nsamples=  data.shape[0]  
TN= [data.loc[j, "sampleid"] for j in range(nsamples) if (data.loc[j, "Crit1"]==0) and (data.loc[j, "Clasificacion"]==0 )]
print("True Positive", TN)
TP= [data.loc[j, "sampleid"] for j in range(nsamples) if (data.loc[j, "Crit1"]==1) and (data.loc[j, "Clasificacion"]==1 )]
print("True Negative", TP)
FP= [data.loc[j, "sampleid"] for j in range(nsamples) if (data.loc[j, "Crit1"]==0) and (data.loc[j, "Clasificacion"]==1 )]
print("False Positive", FP)
FN= [data.loc[j, "sampleid"] for j in range(nsamples) if (data.loc[j, "Crit1"]==1) and (data.loc[j, "Clasificacion"]==0 )]
print("False Negative", FN)

#%%
#TN=[113,177]
#TP=[240,161]
#FP=[162,216]
#FN=[101,57]

pacientes=[TN, TP, FP, FN]
eval_TN=[]
eval_TP=[]
eval_FP=[]
eval_FN=[]
loss_bien_TN=[]
loss_mal_TN=[]
loss_bien_TP=[]
loss_mal_TP=[]
loss_bien_FP=[]
loss_mal_FP=[]
loss_bien_FN=[]
loss_mal_FN=[]

dict_bien={}
dict_mal={}
dict_clasificacion={}


for k in range (np.shape(pacientes)[0]):
    for j in range(np.shape(pacientes[k])[0]):
        print(pacientes[k][j])
        scalograms = DataGeneratorAuto(list_IDs=[pacientes[k][j]], **params)
        eval=[]

        for i in range (np.shape(scalograms)[0]):
            eval.append(modelo_autoencoder.evaluate(scalograms[i][0] ,scalograms[i][0]))
        if k==0:
            y_true= 0
            y_pred=0
            etiqueta="True_Negative"
            eval_TN=np.concatenate((eval_TN, eval))
        elif k==1:
            y_true=1
            y_pred=1
            etiqueta="True_Positive"
            eval_TP=np.concatenate((eval_TP, eval))
        elif k==2:
            y_true=0
            y_pred=1
            etiqueta="False_Positive"
            eval_FP=np.concatenate((eval_FP, eval))
        elif k==3:
            y_true=1
            y_pred=0
            etiqueta="False_Negative"

            eval_FN=np.concatenate((eval_FN, eval))
        pat_intermediate = patient_group_aelda([pacientes[k][j]],modelo_autoencoder,layer_name, **paramsV)
        if pat_intermediate.any():
            if clasificador=="hierarchical":
                pat_predictions = clf_model.fit_predict(pat_intermediate)
            else:
        
                pat_predictions = clf_model.predict(pat_intermediate)  
                if clasificador=="perceptron":
                    pat_predictions=pat_predictions>0.5
                if clasificador=="NN":
                    pat_predictions=pat_predictions>0.5
        dict_clasificacion[pacientes[k][j]]=pat_predictions
        muestra_bien_clasificada=[]
        muestra_mal_clasificada=[]
        loss_bien=[]
        loss_mal=[]

        for i, pat_prediction in enumerate(pat_predictions):
            
            if pat_prediction==y_true:
                muestra_bien_clasificada.append(i)
                loss_bien.append(eval[i])

            else:
                muestra_mal_clasificada.append(i)
                loss_mal.append(eval[i])

        loss=[loss_bien,loss_mal]
        dict_bien[pacientes[k][j]]=muestra_bien_clasificada
        dict_mal[pacientes[k][j]]=muestra_mal_clasificada



        if k==0:
            loss_bien_TN=np.concatenate((loss_bien_TN, loss_bien))
            loss_mal_TN=np.concatenate((loss_mal_TN, loss_mal))
        elif k==1:
            loss_bien_TP=np.concatenate((loss_bien_TP, loss_bien))
            loss_mal_TP=np.concatenate((loss_mal_TP, loss_mal))
        elif k==2:
            loss_bien_FP=np.concatenate((loss_bien_FP, loss_bien))
            loss_mal_FP=np.concatenate((loss_mal_FP, loss_mal))
        elif k==3:
            loss_bien_FN=np.concatenate((loss_bien_FN, loss_bien))
            loss_mal_FN=np.concatenate((loss_mal_FN, loss_mal))
        # try:
        #     minimo = np.min([np.min(loss[0]),np.min(loss[1])])
        #     maximo = np.max([np.max(loss[0]),np.max(loss[1])])
        #     plt.hist(loss ,range = (minimo,maximo),bins=50,label=["Muestras bien clasificadas", "Muestras Mal Clasificadas"])
        #     plt.title("Loss según clasificación para paciente "+str(pacientes[k][j])+ " ("+ etiqueta+")")
        #     plt.legend()
        #     plt.show()

        #     plt.savefig(plot_path+"Loss según clasificación para paciente "+str(pacientes[k][j])+ " ("+ etiqueta+")")
        #     plt.close()
        # except ValueError:  #raised if `y` is empty.
        #     print("Error para el ploteo de Loss según clasificación para paciente "+ str(pacientes[k][j]))

        # plt.plot( muestra_bien_clasificada, loss_bien, "o",label="Bien clasificadas" )
        # plt.plot(muestra_mal_clasificada, loss_mal,"o", label="Mal Clasificadas",  )

        # plt.title("Loss para paciente "+str(pacientes[k][j])+ " ("+ etiqueta+")")
        # plt.legend()
        # plt.show()
        # plt.savefig(plot_path+"Loss para paciente "+str(pacientes[k][j])+ " ("+ etiqueta+")")
        # plt.close()

loss_total_TN=[loss_bien_TN,loss_mal_TN]
minimo = np.min([np.min(loss_total_TN[0]),np.min(loss_total_TN[1])])
maximo = np.max([np.max(loss_total_TN[0]),np.max(loss_total_TN[1])])
plt.hist(loss_total_TN, range = (minimo,maximo),bins=50,label=["Muestras bien clasificadas", "Muestras Mal Clasificadas"])
plt.title("Loss True Negative")
plt.legend()

#plt.show()
plt.savefig(plot_path+"Loss True Negative")
plt.close()


loss_total_TP=[loss_bien_TP,loss_mal_TP]
minimo = np.min([np.min(loss_total_TP[0]),np.min(loss_total_TP[1])])
maximo = np.max([np.max(loss_total_TP[0]),np.max(loss_total_TP[1])])
plt.hist(loss_total_TP, range = (minimo,maximo),bins=50,label=["Muestras bien clasificadas", "Muestras Mal Clasificadas"])
plt.title("Loss True Positive")
plt.legend()

#plt.show()
plt.savefig(plot_path+"Loss True Positive")
plt.close()

loss_total_FP=[loss_bien_FP,loss_mal_FP]
minimo = np.min([np.min(loss_total_FP[0]),np.min(loss_total_FP[1])])
maximo = np.max([np.max(loss_total_FP[0]),np.max(loss_total_FP[1])])
plt.hist(loss_total_FP, range = (minimo,maximo),bins=50,label=["Muestras bien clasificadas", "Muestras Mal Clasificadas"])
plt.title("Loss False Positive")
plt.legend()

#plt.show()
plt.savefig(plot_path+"Loss False Positive")
plt.close()

loss_total_FN=[loss_bien_FN,loss_mal_FN]
minimo = np.min([np.min(loss_total_FN[0]),np.min(loss_total_FN[1])])
maximo = np.max([np.max(loss_total_FN[0]),np.max(loss_total_FN[1])])
plt.hist(loss_total_FN, range = (minimo,maximo),bins=50,label=["Muestras bien clasificadas", "Muestras Mal Clasificadas"])
plt.title("Loss False Negative")
plt.legend()

#plt.show()
plt.savefig(plot_path+"Loss False Negative")
plt.close()

v_val=0.1
h_val=1000
verts = list(zip([-h_val,h_val,h_val,-h_val],[-v_val,-v_val,v_val,v_val]))

cmap = clrs.ListedColormap(['red', 'green'])
for k in dict_clasificacion:
    scatter=py.scatter(list(range(len(dict_clasificacion[k]))) ,[k]*len(dict_clasificacion[k]),marker=verts, c=(dict_clasificacion[k] != True), cmap=cmap )
plt.title("Muestras clasificadas como inestables y estables")
plt.legend(*scatter.legend_elements(),bbox_to_anchor =(0.5, -0.05), ncol = 2)
#plt.show()
plt.savefig(plot_path+"Muestras clasificadas como inestables y estables")
plt.close()

#%%

######Ploteo de histogramas Loss estables vs inestables

x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf,  x_inestables_test_clf,x_estables_test_clf, x_ae_train, x_ae_val=ingesta_etiquetas()
todos=np.unique(np.concatenate((x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf,  x_ae_train, x_ae_val), axis=None))
estables_entrada=[]
inestables_entrada=[]
loss_estable_todos=[]
loss_inestable_todos=[]
for pac in todos:
    print(pac)
    scalograms = DataGeneratorAuto(list_IDs=[pac], **params)
    #print(np.shape(scalograms))
    eval=[]
    loss_estable=[]
    loss_inestable=[]
    for i in range (np.shape(scalograms)[0]):
        eval.append(modelo_autoencoder.evaluate(scalograms[i][0] ,scalograms[i][0] ))
    pat_intermediate = patient_group_aelda([pac],modelo_autoencoder,layer_name, **paramsV)

    if pat_intermediate.any():
        if clasificador=="hierarchical":
            pat_predictions = clf_model.fit_predict(pat_intermediate)
        else:  
            pat_predictions = clf_model.predict(pat_intermediate)  
            if clasificador=="perceptron":
                pat_predictions=pat_predictions>0.5
            if clasificador=="NN":
                pat_predictions=pat_predictions>0.5
    eval=np.array(eval)
    pat_predictions=np.array(pat_predictions)

    std=np.std(eval)
    mean=np.mean(eval)
    eval_std=(eval-mean)/std

    largo=len(pat_predictions)
    for i in range(largo):
        if pat_predictions[i]!=True:
            loss_estable.append(eval[i])
        else:
            loss_inestable.append(eval[i])
    
    loss_estable_todos=np.concatenate((loss_estable_todos,loss_estable))
    loss_inestable_todos=np.concatenate((loss_inestable_todos,loss_inestable))

    data = pd.read_csv (dir_etiquetas+'clasificaciones_antropometricos.csv')   
    encontre=False
    j=0
    nsamples=data.shape[0]
    

    while not encontre and nsamples>0: 
        print(data.loc[j, "sampleid"])
        if(data.loc[j, "sampleid"]==pac):
            if data.loc[j, "Crit1"]:
                print("inestable")
                inestables_entrada=np.concatenate((inestables_entrada, eval))
            else:
                print("estable")
                estables_entrada= np.concatenate((estables_entrada, eval))
            encontre=True

        j=j+1
        nsamples=nsamples-1    

    
#%%

loss_total=[loss_estable_todos,loss_inestable_todos]
print("estables salida", len(loss_estable_todos))
print("inestables salida", len(loss_inestable_todos))
minimo = np.min([np.min(loss_total[0]),np.min(loss_total[1])])
maximo = np.max([np.max(loss_total[0]),np.max(loss_total[1])])
plt.hist(loss_total, range = (minimo,maximo),bins=30,label=["Clasificadas Estable", "Clasificadas Inestable"])
plt.title("Loss estables vs inestables a la salida")
plt.legend()

#plt.show()
plt.savefig(plot_path+"Loss estables vs inestables a la salida")
plt.close()

loss_total_entrada=[estables_entrada, inestables_entrada]
print("estables entrada", len(estables_entrada))
print("inestables entrada", len(inestables_entrada))
minimo = np.min([np.min(loss_total_entrada[0]),np.min(loss_total_entrada[1])])
maximo = np.max([np.max(loss_total_entrada[0]),np.max(loss_total_entrada[1])])
plt.hist(loss_total_entrada, range = (minimo,maximo),bins=30,label=["Clasificadas Estable", "Clasificadas Inestable"])
plt.title("Loss estables vs inestables a la entrada")
plt.legend()

#plt.show()
plt.savefig(plot_path+"Loss estables vs inestables a la entrada")
plt.close()

# %%
