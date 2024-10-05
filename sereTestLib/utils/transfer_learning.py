
import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sereTestLib.utils.ingesta_etiquetas import  ingesta_etiquetas_concat
import wandb
from wandb.keras import WandbCallback

##################################
# Librerias y parametros Sere
##################################
from sereTestLib.parameters import *
from sereTestLib.autoencoder.ae_train_save_model import ae_train_save_model, ae_load_model

from sereTestLib.autoencoder.DataClassAuto import  DataGeneratorTransfer
from sereTestLib.clasificador.entrenamiento_clasificador import print_write_classifier_stats
from sereTestLib.Preprocesamiento.Preprocesamiento import Preprocesamiento


def concatenar_modelos(modelo='Xception', entrenar_base=False):

    inDims = (height,width,3)

    if modelo=='Xception':
        base_model = keras.applications.Xception(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=inDims,
        include_top=False)
    elif modelo=="InceptionResNetV2":
        base_model = keras.applications.InceptionResNetV2(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=inDims,
        include_top=False)
    elif modelo=="EfficientNetB0":
        base_model = keras.applications.EfficientNetB0(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=inDims,
        include_top=False)
    base_model.trainable = entrenar_base

    inputs_ac = keras.Input(shape=inDims)
    x = base_model(inputs_ac, training=False)
    avg1 = keras.layers.GlobalAveragePooling2D()(x)

    inputs_gy = keras.Input(shape=inDims)
    x = base_model(inputs_gy, training=False)
    avg2 = keras.layers.GlobalAveragePooling2D()(x)

    x= tf.keras.layers.Concatenate()([avg1, avg2])


    x=keras.layers.Dense(128)(x)
    if batchnorm:
        x=keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    x=keras.layers.Dropout(d)(x)
    x=keras.layers.Dense(64)(x)
    if batchnorm:
        x=keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    x=keras.layers.Dropout(d)(x)
    x=keras.layers.Dense(32)(x)
    if batchnorm:
        x=keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    x=keras.layers.Dropout(d)(x)
    x=keras.layers.Dense(16)(x)
    if batchnorm:
        x=keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation('relu')(x)
    x=keras.layers.Dropout(d)(x)

    output = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=[inputs_ac, inputs_gy], outputs=output)
    return(model)

def cargar_datos(patient_list, ground_truth,patient_list_val, ground_truth_val,preprocesar):
    paramsT= {'data_dir' : dir_preprocessed_data_train,
                           'dim': inDim,
                           'batch_size': batch_size,
                           'shuffle': True,
                           'activities':act_ae,
                           'labels':ground_truth}

    paramsV= {'data_dir' : dir_preprocessed_data_test,
                           'dim': inDim,
                           'batch_size': batch_size,
                           'shuffle': False,
                           'activities':act_ae,
                           'labels':ground_truth_val}
    if preprocesar:
        Preprocesamiento(patient_list, patient_list_val)
    else:
        run = wandb.init(project=project_wandb,job_type="load dataset")
        data_artifact = run.use_artifact(extra+str(preprocesamiento)+':ae+mlp')
        data_artifact.checkout(path_dataset)
        run.finish()
    generator = DataGeneratorTransfer(list_IDs=patient_list, **paramsT)
    generator_val = DataGeneratorTransfer(list_IDs=patient_list_val, **paramsV)

    return(generator,generator_val)

def entrenar_concatenado(generator, generator_val, modelo_autoencoder,entrenar_ae=False,opt="sgd"):


    modelo=concatenar_modelos(modelo_autoencoder, entrenar_ae)
    if (opt=="Adam"):
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
    elif (opt=="sgd"):
        optimizer=tf.keras.optimizers.SGD(learning_rate=base_learning_rate)

    modelo.compile(loss="mse", optimizer=optimizer,metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC()])
    callbacks=[WandbCallback(save_model=False),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=2)]

    modelo.fit(generator, epochs=100, validation_data=(generator_val),callbacks=callbacks)
    modelo.save(model_path_clf+clasificador_name+'.h5')

def predict_modelo(modelo,list, umbral, modo):
    predicciones= modelo.predict(list)

    y_est = predicciones > umbral
    #print("Predicciones: ",predicciones)
    #print("y estimado: ",y_est)

    # img=plt.scatter(np.arange(len(predicciones)),predicciones)
    # #images = wandb.Image(img, caption="Predicciones")
    # wandb.log({modo: img})
    # #plt.show()

    # hist=plt.hist(predicciones)
    # #images = wandb.Image(hist, caption="Histograma")
    # wandb.log({modo: hist})
    return(y_est)

def crear_grupos():
    x_estables_train, x_inestables_train,x_estables_val,x_inestables_val =ingesta_etiquetas_concat()

    patient_list=np.concatenate((x_estables_train,x_inestables_train), axis=None)
    ground_truth = np.concatenate([np.zeros(len(x_estables_train)),np.ones(len(x_inestables_train))]) # 0=stable 1=unstable

    patient_list_val=np.concatenate((x_estables_val,x_inestables_val), axis=None)
    ground_truth_val = np.concatenate([np.zeros(len(x_estables_val)),np.ones(len(x_inestables_val))]) # 0=stable 1=unstable

    return(patient_list, ground_truth,patient_list_val, ground_truth_val)


if __name__== '__main__':
    modelo="InceptionResNetV2"
    entrenar=True
    entrenar_base=False
    umbral=0.5
    preprocesar=False
    estadisticas=True
    batchnorm=True
    opt="Adam" #Adam o sgd
    d=0.4

    ##Cargar muestras y etiquetas
    patient_list, ground_truth,patient_list_val, ground_truth_val=crear_grupos()

    generator, generator_val,=cargar_datos(patient_list, ground_truth,patient_list_val, ground_truth_val,preprocesar)

    run=wandb.init(project="SereTest-autoencoder",reinit=True,job_type="load ae")
    modelo_artifact=run.use_artifact(autoencoder_name+':latest')
    modelo_dir=modelo_artifact.download(model_path_ae)
    run.finish()
    modelo_autoencoder=ae_load_model(modelo_dir)
    config = {"giro x": girox, "giro z": giroz, "Escalado":escalado, "Clasificador":clasificador,"Actividad":act_clf,"Modelo":modelo,"Preprocesamiento":preprocesamiento,"lr":base_learning_rate, "optimizer":opt,"batch norm":batchnorm, "Dropout":d}


    ##Entrenar modelo
    if entrenar:
        run=wandb.init(project="SereTest-clasificador",reinit=True,config=config,job_type="train clf",name=clasificador_name)
        entrenar_concatenado(generator, generator_val, modelo, entrenar_base)
        trained_model_artifact=wandb.Artifact(clasificador_name, type="model")
        trained_model_artifact.add_dir(model_path_clf)
        run.log_artifact(trained_model_artifact)
        #run.finish()
    ##Cargar modelo
    else:
        run=wandb.init(project="SereTest-clasificador",reinit=True,job_type="load clf")
        modelo_artifact=run.use_artifact(clasificador_name+':latest')
        modelo_dir=modelo_artifact.download(model_path_clf)
        run.finish()
        run=wandb.init(project="SereTest-clasificador",reinit=True,config=config,job_type="train clf",name=clasificador_name)
    if estadisticas:
        modelo_entrenado=keras.models.load_model(model_path_clf+clasificador_name+'.h5')
        ##Predecir grupo de train
        #print("Predecir grupo de train")
        #labels_train= predict_modelo(modelo_entrenado,generator,umbral,modo="Train")
        ##Predecir grupo de validación
        #print("Predecir grupo de validación")
        #labels_validate=predict_modelo(modelo_entrenado,generator_val,umbral, modo="Validacion")
        ##Estadísticas
        #print_write_classifier_stats(generator.label[35:], labels_train, generator_val.label[35:],labels_validate,0, 0, 0, 0,0,clasificador,act_ae,extra)
    run.finish()
