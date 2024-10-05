#%%
from sereTestLib.utils.ingesta_etiquetas import ingesta_etiquetas
from sereTestLib.parameters  import *
from sereTestLib.clasificador.entrenamiento_clasificador import entrenamiento_clasificador
from sereTestLib.autoencoder.DataClassAuto import DataGeneratorAuto, DataGeneratorPw
from sereTestLib.parameters import *
from sereTestLib.clasificador.entrenamiento_clasificador import train_clasificador,predict_clf,plot_clf_transformation,print_write_classifier_stats
from sereTestLib.clasificador.extras import clasificador_name_creation, patient_group_aelda
from sereTestLib.autoencoder.ae_train_save_model import autoencoder_model_name_creation




from tensorflow import keras
import numpy as np
import mlflow.tensorflow
import mlflow.sklearn
import os

#Obtener las caracteristicas
def intermediate_concat(list, solo_potencia=False,**params):
    generator=DataGeneratorPw(list_IDs=list,**params)
    autoencoder_name=autoencoder_model_name_creation(act_ae)+'.h5'
    modelo=keras.models.load_model(model_path + autoencoder_name)


    intermediate_layer_model = keras.Model(inputs=modelo.input,
                                    outputs=modelo.get_layer(layer_name).output)
    intermediate = intermediate_layer_model.predict(generator)
    print(np.shape(intermediate)[0])
    l=np.shape(intermediate)[0]
    #print(generator.pow)
    print(np.shape(generator.pow[:l]))
    print(np.shape(intermediate))
    if solo_potencia:
        intermediate_pow=generator.pow[:l]
    else:
        intermediate_pow = np.concatenate([generator.pow[:l], intermediate],axis=1)
    print(np.shape(intermediate_pow))
    return(intermediate_pow)
    #return(0)

mlflow.set_tracking_uri("https://dagshub.com/victoria.tournier/Sere_etapa2.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME']="victoria.tournier"
os.environ['MLFLOW_TRACKING_PASSWORD']="vtournier"


mlflow.start_run(run_name="entrenamiento potencia",nested=True)
solo_potencia=True

#%%
x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf,  x_inestables_test_clf,x_estables_test_clf, x_ae_train, x_ae_val,x_estables_ae_train,x_inestables_ae_train,x_estables_ae_val,x_inestables_ae_val=ingesta_etiquetas()
train=np.concatenate((x_inestables_train_clf, x_estables_train_clf, x_ae_train), axis=None)
val=np.concatenate((x_ae_val, x_inestables_val_clf, x_estables_val_clf), axis=None)

#Calcular potencia


paramsT= {'data_dir' : train_scalogram_path,
                            'dim': inDim,
                            'batch_size': 1,
                            'shuffle': True,
                            'activities':act_ae,
                            'data_dir_pow':dir_segundo_clas + "wavelet_aumentados3/",
                            'labels':[]
                            }
                    

paramsV= {'data_dir' : scalogram_path,
                            'dim': inDim,
                            'batch_size': 1,
                            'shuffle': False,
                            'activities':act_ae,
                            'data_dir_pow':dir_segundo_clas + "wavelet_normales/",
                            'labels':[]
                            }


#training_unstable = DataGeneratorPw(list_IDs=x_inestables_train_clf,**paramsT)
intermediate_training_unstable=(intermediate_concat(x_inestables_train_clf,solo_potencia,**paramsT))
intermediate_training_stable=(intermediate_concat(x_estables_train_clf,solo_potencia,**paramsT))
intermediate_validation_unstable=(intermediate_concat(x_inestables_val_clf,solo_potencia,**paramsV))
intermediate_validation_stable=(intermediate_concat(x_estables_val_clf,solo_potencia,**paramsV))

#%%
train=np.concatenate((intermediate_training_unstable,intermediate_training_stable))
print(np.shape(train))
y_train=np.concatenate((np.ones(np.shape(intermediate_training_unstable)[0]),np.zeros(np.shape(intermediate_training_stable)[0])))
print(np.shape(y_train))

val=np.concatenate((intermediate_validation_unstable,intermediate_validation_stable))
print(np.shape(val))
y_val=np.concatenate((np.ones(np.shape(intermediate_validation_unstable)[0]),np.zeros(np.shape(intermediate_validation_stable)[0])))
print(np.shape(y_val))


#print(y_train)
#%%
#clasificador

mlflow.log_param("solo_potencia", solo_potencia)    
mlflow.log_param("girox", girox)
mlflow.log_param("giroz", giroz)
mlflow.log_param("Escalado", escalado)
mlflow.log_param("clasificador", clasificador)
mlflow.log_param("act", "Caminando")
mlflow.sklearn.autolog()
clasificador_name =   clasificador_name_creation(act_clf, clasificador) +'prueba'+'.joblib'

clf_trained = train_clasificador(train,y_train,clasificador_name, clasificador)
    #clf_trained=  load(model_path + lda_name)

    ##Predecir grupo de train
unstable_predictions_train = predict_clf(clf_trained,intermediate_training_unstable, clasificador)
stable_predictions_train = predict_clf(clf_trained,intermediate_training_stable, clasificador)
labels_train = np.concatenate([unstable_predictions_train,stable_predictions_train],axis=0)

unstable_predictions_val = predict_clf(clf_trained,intermediate_validation_unstable, clasificador)
stable_predictions_val = predict_clf(clf_trained,intermediate_validation_stable, clasificador)
labels_val = np.concatenate([unstable_predictions_val,stable_predictions_val],axis=0)

    #print(unstable_predictions)
    #print(stable_predictions)
    

print_write_classifier_stats(y_train, labels_train, y_val,labels_val,0, 0, 0, 0,0,clasificador,act_clf,extra)
mlflow.end_run()

#%%