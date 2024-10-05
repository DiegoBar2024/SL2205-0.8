#%%
import numpy as np
import matplotlib.pyplot as plt

from statistics import mean
from sereTestLib.autoencoder.ae_train_save_model import  autoencoder_model_name_creation
from tensorflow import keras
from sereTestLib.parameters import *
from sereTestLib.autoencoder.DataClassAuto import DataGeneratorAuto, DataGeneratorAuto_Tapar_canales
from sereTestLib.autoencoder.plot_testgroup_scalograms import plot_testgroup_scalograms_tapar_canales, plot_testgroup_scalograms
from sereTestLib.utils.ingesta_etiquetas import ingesta_etiquetas
#%%#%%
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

params= {'data_dir' : scalogram_path,
                            'dim': inDim,
                            'batch_size': 1,
                            'shuffle': False,
                            'activities':act_ae}
def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

autoencoder_name = autoencoder_model_name_creation(act_ae) + '.h5'
print(autoencoder_name)
modelo_autoencoder = keras.models.load_model(model_path + autoencoder_name)
modelo_autoencoder.summary()

x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf,  x_inestables_test_clf,x_estables_test_clf, x_ae_train, x_ae_val=ingesta_etiquetas()
samples=np.unique(np.concatenate((x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf,  x_ae_train, x_ae_val), axis=None))

matriz=np.zeros((6,6))
matrizdiag=np.zeros((6,6))

#%%
for s in samples:
	print(s)
	samples_number=[s]

	result = results()

	scalograms = DataGeneratorAuto(list_IDs=samples_number, **params)
	scalogram=scalograms[0][0]
	scalogram_reconstruido=modelo_autoencoder.predict(scalogram)


	j=0
	#Tener muestra reconstruida tapada
	scalograms_tapados = DataGeneratorAuto_Tapar_canales(list_IDs=samples_number, **params, chan0=False)
	scalogram_tapado=scalograms_tapados[0][0]
	scalogram_reconstruido_tapado=modelo_autoencoder.predict(scalogram_tapado)



	for i in range (6):
		if j==i:
			matrizdiag[j,i]=matrizdiag[j,i]+((mse(scalogram[:,:,:,i],scalogram_reconstruido_tapado[:,:,:,i] )-mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))/mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))
			matriz[j,i]=np.nan
		else:
			matriz[j,i]=matriz[j,i]+((mse(scalogram[:,:,:,i],scalogram_reconstruido_tapado[:,:,:,i] )-mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))/mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))
			matrizdiag[j,i]=np.nan
		#print("MSE scalogram original-reconstruido",mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))
		#print("MSE scalogram original-tapado",mse(scalogram[:,:,:,i],scalogram_reconstruido_tapado[:,:,:,i] ))

	j=1

	#Tener muestra reconstruida tapada
	scalograms_tapados = DataGeneratorAuto_Tapar_canales(list_IDs=samples_number, **params, chan1=False)
	scalogram_tapado=scalograms_tapados[0][0]
	scalogram_reconstruido_tapado=modelo_autoencoder.predict(scalogram_tapado)


	for i in range (6):
		if j==i:
			matrizdiag[j,i]=matrizdiag[j,i]+((mse(scalogram[:,:,:,i],scalogram_reconstruido_tapado[:,:,:,i] )-mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))/mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))
			matriz[j,i]=np.nan
		else:
			matriz[j,i]=matriz[j,i]+((mse(scalogram[:,:,:,i],scalogram_reconstruido_tapado[:,:,:,i] )-mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))/mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))
			matrizdiag[j,i]=np.nan


	j=2
	#Tener muestra reconstruida tapada
	scalograms_tapados = DataGeneratorAuto_Tapar_canales(list_IDs=samples_number, **params, chan2=False)
	scalogram_tapado=scalograms_tapados[0][0]
	scalogram_reconstruido_tapado=modelo_autoencoder.predict(scalogram_tapado)


	for i in range (6):
		if j==i:
			matrizdiag[j,i]=matrizdiag[j,i]+((mse(scalogram[:,:,:,i],scalogram_reconstruido_tapado[:,:,:,i] )-mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))/mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))
			matriz[j,i]=np.nan
		else:
			matriz[j,i]=matriz[j,i]+((mse(scalogram[:,:,:,i],scalogram_reconstruido_tapado[:,:,:,i] )-mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))/mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))
			matrizdiag[j,i]=np.nan



	j=3
	#Tener muestra reconstruida tapada
	scalograms_tapados = DataGeneratorAuto_Tapar_canales(list_IDs=samples_number, **params, chan3=False)
	scalogram_tapado=scalograms_tapados[0][0]
	scalogram_reconstruido_tapado=modelo_autoencoder.predict(scalogram_tapado)


	for i in range (6):
		if j==i:
			matrizdiag[j,i]=matrizdiag[j,i]+((mse(scalogram[:,:,:,i],scalogram_reconstruido_tapado[:,:,:,i] )-mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))/mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))
			matriz[j,i]=np.nan
		else:
			matriz[j,i]=matriz[j,i]+((mse(scalogram[:,:,:,i],scalogram_reconstruido_tapado[:,:,:,i] )-mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))/mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))
			matrizdiag[j,i]=np.nan


	j=4
#Tener muestra reconstruida tapada
	scalograms_tapados = DataGeneratorAuto_Tapar_canales(list_IDs=samples_number, **params, chan4=False)
	scalogram_tapado=scalograms_tapados[0][0]
	scalogram_reconstruido_tapado=modelo_autoencoder.predict(scalogram_tapado)


	for i in range (6):
		if j==i:
			matrizdiag[j,i]=matrizdiag[j,i]+((mse(scalogram[:,:,:,i],scalogram_reconstruido_tapado[:,:,:,i] )-mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))/mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))
			matriz[j,i]=np.nan
		else:
			matriz[j,i]=matriz[j,i]+((mse(scalogram[:,:,:,i],scalogram_reconstruido_tapado[:,:,:,i] )-mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))/mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))
			matrizdiag[j,i]=np.nan


	j=5
	#Tener muestra reconstruida tapada
	scalograms_tapados = DataGeneratorAuto_Tapar_canales(list_IDs=samples_number, **params, chan5=False)
	scalogram_tapado=scalograms_tapados[0][0]
	scalogram_reconstruido_tapado=modelo_autoencoder.predict(scalogram_tapado)


	for i in range (6):
		if j==i:
			matrizdiag[j,i]=matrizdiag[j,i]+((mse(scalogram[:,:,:,i],scalogram_reconstruido_tapado[:,:,:,i] )-mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))/mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))
			matriz[j,i]=np.nan
		else:
			matriz[j,i]=matriz[j,i]+((mse(scalogram[:,:,:,i],scalogram_reconstruido_tapado[:,:,:,i] )-mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))/mse(scalogram[:,:,:,i],scalogram_reconstruido[:,:,:,i] ))
			matrizdiag[j,i]=np.nan
# %%

matrix=matriz/len(samples)
matrixdiag=matrizdiag/len(samples)

#print(matrix)

fig, ax = plt.subplots()
ax.matshow(matrix)

for i in range(6):
    for j in range(6):
        c = trunc(matrix[i][j]*100)/100
        ax.text(j, i, str(c), va='center', ha='center')
plt.xlabel("Tapado")
plt.ylabel("Canal MSE")

fig, ax = plt.subplots()
ax.matshow(trunc((matrixdiag)))

for i in range(6):
    for j in range(6):
        c = trunc(matrixdiag[i][j]*100)/100
        ax.text(i, j, str(c), va='center', ha='center')
plt.xlabel("Tapado")
plt.ylabel("Canal MSE")



#%%