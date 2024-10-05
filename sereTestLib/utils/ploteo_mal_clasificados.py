#%%

from sereTestLib.autoencoder.ae_train_save_model import  autoencoder_model_name_creation
from tensorflow import keras
from sereTestLib.parameters import *
from keras.models import Model
from joblib import  load
from sereTestLib.clasificador.extras import patient_group_aelda, clasificador_name_creation
from sereTestLib.autoencoder.plot_testgroup_scalograms import plot_testgroup_scalograms



stable_label = 0
unstable_label = 1


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
samples_number=[220]
y_true=1
y_pred=0
pat_intermediate = patient_group_aelda(samples_number,modelo_autoencoder,layer_name, **paramsV)
if pat_intermediate.any():
    if clasificador=="hierarchical":
        pat_predictions = clf_model.fit_predict(pat_intermediate)
    # pat_evaluation = clf_model.transform(pat_intermediate)
    else:
        
        pat_predictions = clf_model.predict(pat_intermediate)  
        if clasificador=="perceptron":
            pat_predictions=pat_predictions>0.5
        if clasificador=="NN":
            pat_predictions=pat_predictions>0.5

muestra_bien_clasificada=[]
muestra_mal_clasificada=[]

for i in range(len(pat_predictions)):
    if pat_predictions[i]==y_true:
        muestra_bien_clasificada.append(i)
    else:
        muestra_mal_clasificada.append(i)

muestras_bien=[muestra_bien_clasificada[0]]
muestras_mal=[muestra_mal_clasificada[0]]
print("Numero de muestras bien clasificadas a utilizar: ", muestras_bien)
print("Numero de muestras mal clasificadas a utilizar: ", muestras_mal)

result = results()

plot_testgroup_scalograms(samples_number,modelo_autoencoder,cant_muestras,result,num_epochs,group=plot_group,is_long_sample=long_sample, muestras=muestras_bien)    
plot_testgroup_scalograms(samples_number,modelo_autoencoder,cant_muestras,result,num_epochs,group=plot_group,is_long_sample=long_sample, muestras=muestras_mal)    


#%%


