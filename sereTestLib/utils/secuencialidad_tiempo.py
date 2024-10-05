#Cargo los datos
from keras.layers import TimeDistributed, Conv1D, Dense, MaxPooling1D, Flatten, LSTM, Dropout, BatchNormalization
from keras import models
from keras import optimizers
from keras.utils.vis_utils import plot_model
from sereTestLib.autoencoder.DataClassAuto import  DataGeneratorSeqTime
from sereTestLib.utils.ingesta_etiquetas import  ingesta_etiquetas_concat

import tensorflow as tf
from tensorflow import keras
from sereTestLib.parameters import *
import wandb
from wandb.keras import WandbCallback
from sereTestLib.clasificador.entrenamiento_clasificador import print_write_classifier_stats
from sereTestLib.Preprocesamiento.Preprocesamiento import Preprocesamiento

def crear_grupos():
    x_estables_train, x_inestables_train,x_estables_val,x_inestables_val =ingesta_etiquetas_concat()


    patient_list=np.concatenate((x_estables_train,x_inestables_train), axis=None)
    ground_truth = np.concatenate([np.zeros(len(x_estables_train)),np.ones(len(x_inestables_train))]) # 0=stable 1=unstable

    patient_list_val=np.concatenate((x_estables_val,x_inestables_val), axis=None)
    ground_truth_val = np.concatenate([np.zeros(len(x_estables_val)),np.ones(len(x_inestables_val))]) # 0=stable 1=unstable

    return(patient_list, ground_truth,patient_list_val, ground_truth_val)

patient_list, ground_truth,patient_list_val, ground_truth_val=crear_grupos()
# preprocesar=True
# if preprocesar:
#     Preprocesamiento(patient_list, patient_list_val)
# else:
#     run = wandb.init(project=project_wandb,job_type="load dataset")
#     data_artifact = run.use_artifact(extra+str(preprocesamiento)+':ae+mlp')
#     data_artifact.checkout(path_dataset)
#     run.finish()
paramsT= {'data_dir' : dir_out_ov_split_train,
                        'dim': inDim,
                        'batch_size': batch_size,
                        'shuffle': True,
                        'activities':act_ae,
                        'labels':ground_truth}

paramsV= {'data_dir' : dir_out_ov_split_test,
                        'dim': inDim,
                        'batch_size': batch_size,
                        'shuffle': False,
                        'activities':act_ae,
                        'labels':ground_truth_val}
generator = DataGeneratorSeqTime(list_IDs=patient_list, **paramsT)
generator_val = DataGeneratorSeqTime(list_IDs=patient_list_val, **paramsV)
entrenar=False
dropout=0.15

opt="sgd" #Adam o sgd
if entrenar:
    dropout=0.15
    model = models.Sequential()
    model.add(LSTM(1024, input_shape=(secuencia,6)))

    #model.add(Dropout(dropout))
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(dropout))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(256, activation='elu'))
    model.add(Dropout(dropout))

    model.add(Dense(128, activation='elu'))
    model.add(Dropout(dropout))

    model.add(Dense(64, activation='elu'))
    model.add(Dense(32, activation='elu'))

    model.add(Dense(1, activation='sigmoid'))

    #plot_model(model, to_file=static_path+'modelo_secuencialidad_temporal.png', show_shapes=True, show_layer_names=True)



    if (opt=="Adam"):
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
    elif (opt=="sgd"):
        optimizer=tf.keras.optimizers.SGD(learning_rate=base_learning_rate)


    model.compile(loss=loss_name, optimizer=optimizer,metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC()])



    config = {"giro x": girox, "giro z": giroz, "Escalado":escalado, "Clasificador":clasificador,"Actividad":act_clf,"Preprocesamiento":preprocesamiento,"lr":base_learning_rate, "optimizer":opt,"largo_secuencia":largo_vector,  "loss":loss_name}

    run=wandb.init(project="SereTest-clasificador",reinit=True,config=config,job_type="train clf",name=clasificador_name)


    callbacks=[WandbCallback(save_model=False),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=2,restore_best_weights=True)]
    #callbacks=[WandbCallback(save_model=False)]

    model.fit(generator, epochs=100, validation_data=(generator_val),callbacks=callbacks)
    model.save(model_path_clf+clasificador_name+'.h5')
run=wandb.init(project="SereTest-clasificador",reinit=True,job_type="load clf",name=clasificador_name)

trained_model_artifact=wandb.Artifact(clasificador_name, type="model")
trained_model_artifact.add_dir(model_path_clf)
run.log_artifact(trained_model_artifact)

modelo_entrenado=keras.models.load_model(model_path_clf+clasificador_name+'.h5')
plot_model(modelo_entrenado, to_file=static_path+'modelo_secuencialidad_temporal.png', show_shapes=True, show_layer_names=True)


predict_val= modelo_entrenado.predict(generator_val)
#print(predict_val)
umbral=0.5
y_est = predict_val > umbral
#print(y_est)

#predict= modelo_entrenado.predict(generator)
#print(predict_val)
#y_est_train = predict > umbral

muestras_val=0
labels_val=[]
nro_muestra=[]
for indice in generator_val.indices :
    labels_val.append(generator_val.labels[np.where(generator_val.list_IDs==indice[0])][0])
    nro_muestra.append(indice[0])


#print(labels_val)
#print(y_est)
print("predicciones por vector de segmentos")
print_write_classifier_stats([], [], labels_val,y_est,0, 0, 0, 0,0,clasificador,act_ae,extra)

predicciones_paciente=[]
for pac in generator_val.list_IDs:
    indices=[i for i, j in enumerate(nro_muestra) if j == pac]
    #print(indices)
    promedio=0
    for i in indices:
        promedio+=predict_val[i][0]/len(indices)
    predicciones_paciente.append(promedio)
#print(predicciones_paciente)
y_est_pac = np.array(predicciones_paciente) > 0.5

print("predicciones por muestra entera")
print_write_classifier_stats([], [], generator_val.labels,y_est_pac,0, 0, 0, 0,0,clasificador,act_ae,extra)





