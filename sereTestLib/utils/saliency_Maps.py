#%%


from sereTestLib.autoencoder.ae_train_save_model import  autoencoder_model_name_creation
from tensorflow import keras
from sereTestLib.parameters import *
from keras.models import Model
from matplotlib import pyplot
from joblib import  load

from sereTestLib.autoencoder.DataClassAuto import DataGeneratorAuto
import tensorflow as tf
from sereTestLib.clasificador.extras import clasificador_name_creation



def get_saliency_map(model, image, class_idx):
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        
        loss = predictions[:, class_idx]
    
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, image)
    
    # take maximum across channels
    gradient = tf.reduce_max(gradient, axis=-1)
    
    # convert to numpy
    gradient = gradient.numpy()
    
    # normalize between 0 and 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    smap = (gradient - min_val) / (max_val - min_val + keras.backend.epsilon())
    
    return smap

samples_number=[111]
autoencoder_name = autoencoder_model_name_creation(act_ae) + '.h5'


modelo_autoencoder = keras.models.load_model(model_path + autoencoder_name)
modelo_autoencoder.summary()

layer_name='Dense_encoder'
intermediate_layer_model = keras.Model(inputs=modelo_autoencoder.input,
                                    outputs=modelo_autoencoder.get_layer(layer_name).output)


params= {'data_dir' : train_scalogram_path,
                            'dim': inDim,
                            'batch_size': batch_size,
                            'shuffle': False,
                            'activities':act_ae}


scalograms = DataGeneratorAuto(list_IDs=samples_number, **params)
    
    #pat_intermediate = patient_group_aelda(samples_number,model,layer_name, **params)
scalogram= tf.convert_to_tensor(scalograms[0][0][ 0,:, :, :], dtype=tf.float32) 
scalogram= tf.expand_dims(scalogram, 0)

smap_ae=[]
for i in range (0,128):
    smap_ae.append(get_saliency_map(modelo_autoencoder, scalogram, i))

print(len(smap_ae[0]))

# x=3
# y=2
# ix = 1
# for _ in range(x):
# 	for _ in range(y):
# 		ax = pyplot.subplot(x, y, ix)
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		pyplot.imshow(smap[0][0,:,:,ix-1], cmap='gray')
# 		ix += 1
# pyplot.show()


pyplot.imshow(smap_ae[0][0,:,:], cmap='jet')
pyplot.show()


smap_cod=[]
for i in range (0,256):
    smap_cod.append(get_saliency_map(intermediate_layer_model, scalogram, i))

print(len(smap_cod[0]))

# x=3
# y=2
# ix = 1
# for _ in range(x):
# 	for _ in range(y):
# 		ax = pyplot.subplot(x, y, ix)
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		pyplot.imshow(smap[0][0,:,:,ix-1], cmap='gray')
# 		ix += 1
# pyplot.show()


pyplot.imshow(smap_cod[0][0,:,:], cmap='jet')
pyplot.show()


#Devolver pesos
clasificador_name =   clasificador_name_creation(act_clf, clasificador) +'.joblib'
clf_trained = load(model_path + clasificador_name)

#print((len(clf_trained.coef_[0])))

#pesos=clf_trained.coef_[0]
#print(pesos)

#smap_final=smap[0][0,:,:,:]*0
#for i in range (0,256):
#    #print(pesos[i])
#    #print(smap[i][0,:,:,:])
#    smap_final=smap_final+smap[i][0,:,:,:]*pesos[i]



#x=3
#y=2
#ix = 1
#for _ in range(x):
#	for _ in range(y):
#		ax = pyplot.subplot(x, y, ix)
#		ax.set_xticks([])
#		ax.set_yticks([])
#		pyplot.imshow(smap_final[:,:,ix-1], cmap='jet')
#		ix += 1
#pyplot.show()



# %%
# intermediate = intermediate_layer_model.predict(scalograms[0][0][ :,:, :, :])
# print(len(intermediate[0]))
# intermediate= tf.convert_to_tensor(intermediate, dtype=tf.float32) 
# #intermediate= tf.expand_dims(intermediate, 0)

# saliency=get_saliency_map(clf_trained, intermediate, 0)
# print(len(saliency))

# print((len(saliency[0])))


#%%
# %%
