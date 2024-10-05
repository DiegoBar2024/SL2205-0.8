

#%%
import os
from keras.layers import TimeDistributed, Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Dropout, BatchNormalization
from keras import models
from keras import optimizers
from sereTestLib.autoencoder.ae_train_save_model import ae_train_save_model, ae_load_model

from sereTestLib.autoencoder.DataClassAuto import  DataGeneratorSeqAE
from sereTestLib.utils.ingesta_etiquetas import  ingesta_etiquetas_concat
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sereTestLib.parameters import *
import wandb
from wandb.keras import WandbCallback
from sereTestLib.clasificador.entrenamiento_clasificador import print_write_classifier_stats

from sereTestLib.clasificador.extras import patient_group_aelda

def cargar_datos(patient_list, ground_truth,patient_list_val, ground_truth_val):
    """
    Dada una lista de muestras para entrenamiento, otra para muestras para validación y
    y sus respectivas etiquetas, devuelve sus DataLoaders.

    Args:
        patient_list (ndarray): lista de muestras para entrenamiento
        ground_truth (_type_): _description_
        patient_list_val (ndarray): lista de muestras para validación
        ground_truth_val (_type_): _description_
    """
    paramsT= {'data_dir' : dir_ae_train,
                           'dim': inDim,
                           'batch_size': batch_size,
                           'shuffle': True,
                           'activities':act_ae,
                           'dim':(256,),
                           'labels':ground_truth}

    paramsV= {'data_dir' : dir_ae_test,
                           'dim': inDim,
                           'batch_size': batch_size,
                           'shuffle': False,
                           'activities':act_ae,
                           'dim':(256,),
                           'labels':ground_truth_val}
    generator = DataGeneratorSeqAE(list_IDs=patient_list, **paramsT)
    generator_val = DataGeneratorSeqAE(list_IDs=patient_list_val, **paramsV)

    return(generator,generator_val)
# def inferencia_secuencialidad(generator, modelo_entrenado):
#     #print(modelo_entrenado.evaluate(generator))

#     predict_val= modelo_entrenado.predict(generator)
#     y_est = predict_val >= 0.5

#     labels_val=[]
#     nro_muestra=[]
#     for indice in generator.indices :
#         labels_val.append(generator.labels[np.where(generator.list_IDs==indice[0])][0])
#         nro_muestra.append(indice[0])

#     print("predicciones por vector de segmentos")
#     print_write_classifier_stats([], [], labels_val,y_est,0, 0, 0, 0,0,clasificador,act_ae,extra)

#     predicciones_paciente=[]
#     for pac in generator.list_IDs:
#         indices=[i for i, j in enumerate(nro_muestra) if j == pac]
#         promedio=0
#         for i in indices:
#             promedio+=predict_val[i][0]/len(indices)
#         predicciones_paciente.append(promedio)
#     y_est_pac = np.array(predicciones_paciente) > 0.5
#     print("predicciones por muestra entera")
#     print_write_classifier_stats([], [], generator.labels,y_est_pac,0, 0, 0, 0,0,clasificador,act_ae,extra)
#     d={"sere_index":predicciones_paciente,"etiqueta":generator.labels}
#     df=pd.DataFrame(data=d)

#     bins=40
#     plt.title("Sere Index según la etiqueta")
#     plt.hist(df[df['etiqueta']==1]["sere_index"]*100,label='Estable', range=(0,100),bins=bins)
#     plt.hist(df[df['etiqueta']==0]["sere_index"]*100,label='Inestable', range=(0,100),bins=bins)
#     plt.axvline((df[df['etiqueta']==1]["sere_index"]*100).mean(), linestyle='dotted', color ="k", label="Media Estables")
#     plt.axvline((df[df['etiqueta']==0]["sere_index"]*100).mean(), linestyle="dashed",  color ="k",label="Media Inestables")
#     plt.xlabel('Sere Index')
#     plt.ylabel('Cantidad de muestras')
#     plt.legend(title="Etiquetas")
#     plt.savefig("Histograma_val.png")



# def crear_grupos():
#     x_estables_train, x_inestables_train,x_estables_val,x_inestables_val =ingesta_etiquetas_concat()


#     patient_list=np.concatenate((x_estables_train,x_inestables_train), axis=None)
#     ground_truth = np.concatenate([np.zeros(len(x_estables_train)),np.ones(len(x_inestables_train))]) # 0=stable 1=unstable

#     patient_list_val=np.concatenate((x_estables_val,x_inestables_val), axis=None)
#     ground_truth_val = np.concatenate([np.zeros(len(x_estables_val)),np.ones(len(x_inestables_val))]) # 0=stable 1=unstable

#     return(patient_list, ground_truth,patient_list_val, ground_truth_val)




model_cnlst = models.Sequential()
model_cnlst.add(Dense(128,activation='relu', input_shape=(largo_vector,256)))

model_cnlst.add(Dense(64,activation='relu'))

model_cnlst.add(Dropout(0.2))

model_cnlst.add(LSTM(32,return_sequences=False,dropout=0.2)) # used 32 units

model_cnlst.add(Dense(64,activation='relu'))
model_cnlst.add(Dense(32,activation='relu'))
model_cnlst.add(Dropout(0.2))
model_cnlst.add(Dense(1, activation='sigmoid'))
#model_cnlst.summary()

callbacks_list_cnlst=[keras.callbacks.EarlyStopping(
monitor='acc',patience=3),

                keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 3)
               ]




#%%

def crear_grupos():
    x_estables_train, x_inestables_train,x_estables_val,x_inestables_val =ingesta_etiquetas_concat()


    patient_list=np.concatenate((x_estables_train,x_inestables_train), axis=None)
    ground_truth = np.concatenate([np.zeros(len(x_estables_train)),np.ones(len(x_inestables_train))]) # 0=stable 1=unstable

    patient_list_val=np.concatenate((x_estables_val,x_inestables_val), axis=None)
    ground_truth_val = np.concatenate([np.zeros(len(x_estables_val)),np.ones(len(x_inestables_val))]) # 0=stable 1=unstable

    return(patient_list, ground_truth,patient_list_val, ground_truth_val)

def guardar_intermediate(patient_list,patient_list_val):
    for pat in patient_list:
        intermediate = patient_group_aelda([pat],modelo_autoencoder,layer_name=layer_name, **paramsT)
        dir_pat=dir_ae_train+"S"+str(pat)+'/'
        if not os.path.exists(dir_pat):
            os.makedirs(dir_pat)
        #print(np.shape(intermediate))
        for i in range(np.shape(intermediate)[0]):
            np.savez_compressed(dir_pat+"3S"+str(pat)+"_"+str(i) , X=intermediate[i])
    for pat in patient_list_val:
        intermediate = patient_group_aelda([pat],modelo_autoencoder,layer_name=layer_name, **paramsV)
        dir_pat=dir_ae_test+"S"+str(pat)+'/'
        if not os.path.exists(dir_pat):
            os.makedirs(dir_pat)
        for i in range(np.shape(intermediate)[0]):
            np.savez_compressed(dir_pat+"3S"+str(pat)+"_"+str(i) , X=intermediate[i])

paramsT= {'data_dir' : dir_preprocessed_data_train,
                        'dim': inDim,
                        'batch_size': batch_size,
                        'shuffle': True,
                        'activities':act_clf}

paramsV= {'data_dir' : dir_preprocessed_data_test,
                        'dim': inDim,
                        'batch_size': batch_size,
                        'shuffle': False,
                        'activities':act_clf}
modelo_autoencoder=ae_load_model(model_path_ae )

patient_list, ground_truth,patient_list_val, ground_truth_val=crear_grupos()
#guardar_intermediate(patient_list,patient_list_val)

generator, generator_val=cargar_datos(patient_list, ground_truth,patient_list_val, ground_truth_val)




opt="sgd" #Adam o sgd
config = {"giro x": girox, "giro z": giroz, "Escalado":escalado, "Clasificador":clasificador,"Actividad":act_clf,"Preprocesamiento":preprocesamiento,"lr":base_learning_rate, "optimizer":opt,"largo_secuencia":largo_vector}

run=wandb.init(project="SereTest-clasificador",reinit=True,config=config,job_type="train clf",name=clasificador_name)


if (opt=="Adam"):
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
elif (opt=="sgd"):
    optimizer=tf.keras.optimizers.SGD(learning_rate=base_learning_rate)

model_cnlst.compile(loss="mse", optimizer=optimizer,metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC()])
#callbacks=[WandbCallback(save_model=False),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=2)]
callbacks=[WandbCallback(save_model=False)]

model_cnlst.fit(generator, epochs=100, validation_data=(generator_val),callbacks=callbacks)
trained_model_artifact=wandb.Artifact(clasificador_name, type="model")
trained_model_artifact.add_dir(model_path_clf)
run.log_artifact(trained_model_artifact)





# unstable_train_intermediate = patient_group_aelda([34],modelo_autoencoder,layer_name=layer_name, **paramsT)
# print(unstable_train_intermediate)
# print(np.shape(unstable_train_intermediate))
# np.savez_compressed(static_path+"prueba" , X=unstable_train_intermediate)
# loaded = np.load(static_path+"prueba.npz")
# print(np.shape(loaded['X']))
#stable_train_intermediate = patient_group_aelda(stable_train,autoencoder_model,layer_name=layer_name, **paramsT)
#unstable_validate_intermediate = patient_group_aelda(unstable_validation ,autoencoder_model,layer_name=layer_name, **paramsV)
#stable_validate_intermediate = patient_group_aelda(stable_validation ,autoencoder_model,layer_name=layer_name, **paramsV)

