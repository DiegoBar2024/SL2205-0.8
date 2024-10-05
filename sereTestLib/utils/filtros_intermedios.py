#%%


from sereTestLib.autoencoder.ae_train_save_model import  autoencoder_model_name_creation
from tensorflow import keras
from sereTestLib.parameters import *
from keras.models import Model
from matplotlib import pyplot
from sereTestLib.autoencoder.DataClassAuto import DataGeneratorAuto_Tapar_canales



samples_number=[221]

def filtrosIntermedios(samples_number, chan0=True, chan1=True, chan2=True, chan3=True, chan4=True, chan5=True):


	autoencoder_name = autoencoder_model_name_creation(act_ae) + '.h5'
	modelo_autoencoder = keras.models.load_model(model_path + autoencoder_name)
	modelo_autoencoder.summary()

	#ixs = [1,3,5]
	ixs = [1,3,5, 11, 13,15, 17,18]
	outputs = [modelo_autoencoder.layers[i].output for i in ixs]
	model = Model(inputs=modelo_autoencoder.inputs, outputs=outputs)
	print(model)

	params= {'data_dir' : train_scalogram_path,
                            'dim': inDim,
                            'batch_size': batch_size,
                            'shuffle': False,
                            'activities':act_ae}

	#print(train_scalogram_path)


	scalograms = DataGeneratorAuto_Tapar_canales(list_IDs=samples_number, **params, chan0=chan0, chan1=chan1, chan2=chan2, chan3=chan3, chan4=chan4, chan5=chan5)


	feature_maps = model.predict(scalograms)


	x=3
	y=2
	ix = 1
	for _ in range(x):
		for _ in range(y):
			ax = pyplot.subplot(x, y, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			pyplot.imshow(scalograms[0][0][ 0,:, :, ix-1], cmap='jet')
			ix += 1
	pyplot.show()

	
	#x=[8,4,4]
	#y=[4,4,2]
	x=[8,4,4,4,4,8,3, 3]
	y=[4,4,2,2,4,4,2,2]
#	for i in range(3):
	for i in range(8):

		ix = 1
		for _ in range(x[i]):
			for _ in range(y[i]):
				ax = pyplot.subplot(x[i], y[i], ix)
				ax.set_xticks([])
				ax.set_yticks([])
				pyplot.imshow(feature_maps[0][ 0,:, :, ix-1], cmap='jet')
				ix += 1
		pyplot.show()
if __name__== '__main__':
	filtrosIntermedios(samples_number)
# %%
