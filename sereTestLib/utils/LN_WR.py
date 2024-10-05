#%%

from sereTestLib.parameters  import *
from sereTestLib.webservice.parameters_ws import *
from sereTestLib.autoencoder.ae_train_save_model import   ae_load_model

from sereTestLib.autoencoder.DataClassAuto import DataGeneratorAuto
from sereTestLib.clasificador.extras import patient_group_aelda
from sereTestLib.Preprocesamiento.preprocesamiento_val_test import preprocesamiento_val_test


from matplotlib import pyplot
import tensorflow as tf
from joblib import  load

ln=[252]
wr=[254]
#preprocesamiento_val_test(301)

#preprocesamiento_val_test(255)


params= {'data_dir' : dir_preprocessed_data_test,
                            'dim': inDim,
                            'batch_size': batch_size,
                            'shuffle': False,
                            'activities':act_ae}

scalogramsln = DataGeneratorAuto(list_IDs=ln, **params)
scalogramswr = DataGeneratorAuto(list_IDs=wr, **params)

#%%
#Ploteo primer segmento:

print("Ploteo primer segmento LN")

x=3
y=2
ix = 1
for _ in range(x):
    for _ in range(y):
        ax = pyplot.subplot(x, y, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(scalogramsln[0][0][ 0,:, :, ix-1], cmap='jet')
        ix += 1
pyplot.show()


print("Ploteo primer segmento WR")

x=3
y=2
ix = 1
for _ in range(x):
    for _ in range(y):
        ax = pyplot.subplot(x, y, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(scalogramswr[0][0][ 0,:, :, ix-1], cmap='jet')
        ix += 1
pyplot.show()

#%%
ssim_ac=[]
#Comparo escalogramas
for i in range (len(scalogramsln[0][0])):
    ssim=tf.image.ssim(scalogramsln[0][0][i], scalogramswr[0][0][i],max_val=255)
    ssim_ac.append((ssim.numpy()))

pyplot.plot(ssim_ac)
pyplot.title("SSIM")
pyplot.xlabel("Segmento")
pyplot.show()
#%%

modelo_autoencoder=ae_load_model(model_path_ae)

clf_trained = load(model_path_clf + clasificador_name+'.joblib')


#Comparo capa intermedia
pat_intermediate_ln = patient_group_aelda(ln,modelo_autoencoder,layer_name, **params)
pat_intermediate_wr = patient_group_aelda(wr,modelo_autoencoder,layer_name, **params)
intermedios=[]
for i in range(len(pat_intermediate_ln)):
    intermedios.append(np.linalg.norm(pat_intermediate_ln[i]-pat_intermediate_wr[i]))
pyplot.plot(intermedios)
pyplot.title("Diferencia entre las normas en el espacio de 256")
pyplot.xlabel("Segmento")
pyplot.show()
#%%
#Comparo predicciones
pat_predictions_ln = clf_trained.predict(pat_intermediate_ln)
pat_predictions_wr = clf_trained.predict(pat_intermediate_wr)

#%%

diferencia=np.abs(pat_predictions_ln-pat_predictions_wr)
pyplot.plot(diferencia)
pyplot.title("Diferencia entre predicciones en valor absoluto")
pyplot.xlabel("Segmento")
pyplot.show()

# %%
