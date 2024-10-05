from sereTestLib.autoencoder.DataClassAuto import DataGeneratorAuto, DataGeneratorAuto_Tapar_canales
from sereTestLib.parameters  import *
import matplotlib.pyplot as plt
import pywt
import os
import numpy as np

def plot_testgroup_scalograms(samples_number,ae_model,cant_muestras,result:results,num_epochs = 1, group = 'default',extra_name='',is_long_sample=long_sample, muestras=None):
    """
    Function that plots and saves some scalograms both original and reconstructed

    Parameters:
    -----------
    samples_number: list    
    ae_model: 
        autoencoder model
    cant_muestras: int
        number of scalograms to plot
    num_epochs: int  
        Defaults to 1.
    group: str
        Defaults to 'default'.
    extra_name: str
        Defaults to ''.
    """

    # TODO: Acá no tiene sentido utilizar el datagenerator. Levantan todos los scalogramas para plotear algunos
    ## pacientes inestables
    for pat in samples_number:
        for actividad in act_ae:
            params= {'data_dir' : dir_preprocessed_data_test,
                                'dim': inDim,
                                'batch_size': batch_size,
                                'shuffle': False,
                                'activities':[actividad],
                                'long_sample':is_long_sample}

            test_generator = DataGeneratorAuto(list_IDs=samples_number, **params)
            if test_generator.n_samples >0:
                results = ae_model.predict(test_generator)
                if muestras == None:
                    rand_muestra = np.sort(np.random.randint(0,high = np.shape(results)[0],size=cant_muestras))
                else:
                    rand_muestra=muestras
                dataclass_index = rand_muestra[0]//batch_size
                originales_batch = np.asarray(test_generator.__getitem__(rand_muestra[0]//batch_size))
                for muestra in range(len(rand_muestra)):
                    i = rand_muestra[muestra]
                    if i//batch_size == dataclass_index:
                        #print(results_path)
                        ploter(results[i,:,:,:],results_path+"reconstrucciones"+extra_name+"/"+str(group)+"/", params['activities'][0] +"_resultado_"+ str(num_epochs) +"_"+str(pat)+"_"+str(i))
                        ploter(originales_batch[0,np.mod(i,batch_size),:,:,:],results_path+"reconstrucciones"+extra_name+"/"+str(group)+"/", params['activities'][0] +"_original_"+ str(num_epochs) +"_"+str(pat)+"_"+str(i))
                    else:
                       # print(results_path)

                        dataclass_index = i//batch_size
                        originales_batch = np.asarray(test_generator.__getitem__(i//batch_size))
                        ploter(results[i,:,:,:],results_path+"reconstrucciones"+extra_name+"/"+str(group)+"/", params['activities'][0] +"_resultado_"+ str(num_epochs) +"_"+str(pat)+"_"+str(i))
                        ploter(originales_batch[0,np.mod(i,batch_size),:,:,:],results_path+"reconstrucciones"+extra_name+"/"+str(group)+"/", params['activities'][0] +"_original_"+ str(num_epochs) +"_"+str(pat)+"_"+str(i))
        result.images_folder_path = results_path+"reconstrucciones"+extra_name+"/"+str(group)




def plot_testgroup_scalograms_tapar_canales(samples_number,ae_model,cant_muestras,result:results,num_epochs = 1, group = 'default',extra_name='',is_long_sample=long_sample, muestras=None, chan0=True, chan1=True, chan2=True, chan3=True, chan4=True, chan5=True):
    """
    Function that plots and saves some scalograms both original and reconstructed

    Parameters:
    -----------

    samples_number: list    
    ae_model: 
        autoencoder model
    cant_muestras: int
        number of scalograms to plot
    num_epochs: int  
        Defaults to 1.
    group: str
        Defaults to 'default'.
    extra_name: str
         Defaults to ''.
    """


    #TODO: Acá no tiene sentido utilziar el datagenerator. Levantan todos los scalogramas para plotear algunos
    ##pacientes inestables
    for pat in samples_number:
        for actividad in act_ae:
            params= {'data_dir' : dir_preprocessed_data_test,
                                'dim': inDim,
                                'batch_size': batch_size,
                                'shuffle': False,
                                'activities':[actividad],
                                'long_sample':is_long_sample}

            test_generator = DataGeneratorAuto_Tapar_canales(list_IDs=samples_number, **params, chan0=chan0, chan1=chan1, chan2=chan2, chan3=chan3, chan4=chan4, chan5=chan5)
            if test_generator.n_samples >0:
                results=ae_model.predict(test_generator)
                if muestras==None:
                    rand_muestra = np.sort(np.random.randint(0,high = np.shape(results)[0],size=cant_muestras))
                else:
                    rand_muestra=muestras
                dataclass_index = rand_muestra[0]//batch_size
                originales_batch = np.asarray(test_generator.__getitem__(rand_muestra[0]//batch_size))
                for muestra in range(len(rand_muestra)):
                    i = rand_muestra[muestra]
                    if i//batch_size == dataclass_index:
                        #print(results_path)
                        ploter(results[i,:,:,:],results_path+"reconstrucciones"+extra_name+"/"+str(group)+"/", params['activities'][0] +"_resultado_tapado_"+ str(num_epochs) +"_"+str(pat)+"_"+str(i))
                        ploter(originales_batch[0,np.mod(i,batch_size),:,:,:],results_path+"reconstrucciones"+extra_name+"/"+str(group)+"/", params['activities'][0] +"_original_tapado_"+ str(num_epochs) +"_"+str(pat)+"_"+str(i))
                    else:
                       # print(results_path)

                        dataclass_index = i//batch_size
                        originales_batch = np.asarray(test_generator.__getitem__(i//batch_size))
                        ploter(results[i,:,:,:],results_path+"reconstrucciones"+extra_name+"/"+str(group)+"/", params['activities'][0] +"_resultado_tapado_"+ str(num_epochs) +"_"+str(pat)+"_"+str(i))
                        ploter(originales_batch[0,np.mod(i,batch_size),:,:,:],results_path+"reconstrucciones"+extra_name+"/"+str(group)+"/", params['activities'][0] +"_original_tapado_"+ str(num_epochs) +"_"+str(pat)+"_"+str(i))
        result.images_folder_path = results_path+"reconstrucciones"+extra_name+"/"+str(group)





def ploter(val:np.array,image_dir:str,name:str):
    """
    Function that plots the scalogram

    Parameters
    ----------
    val: 
        the scalogram to plot
    image_dir: str
        Output path
    name: str
        name of the plots
    """
    imageName = image_dir+name
#esto solo es útil para poder armar los ejes
    if not os.path.exists(image_dir):   #TODO: Agregar en un log del usuario
        os.makedirs(image_dir)
    titles={'0': 'AC_x', '1': 'AC_y','2': 'AC_z',
            '3': 'GY_x', '4': 'GY_y', '5': 'GY_z'}
    dt = 0.005
    scales=np.arange(4,132)
    frequencies = pywt.scale2frequency('cmor1.5-1', scales, 8) / dt
    time200=np.arange(0, 3, dt)
    #Ploter Aceleraciones
    fig, ax = plt.subplots(ncols=3,nrows=2,figsize=(30,20),sharex='all',dpi=150,constrained_layout=True)

    for i in range(val.shape[2]):
        if i < 3:
            scalogram(val[:, :, i], frequencies, time200, ax[0, i], dt)
            ax[0, i].title.set_text(
                'Magnitud ' + titles[str(i)])
            ax[0, i].set_ylabel('Frecuencia [Hz]')

        else:
            scalogram(val[:, :, i], frequencies, time200, ax[1, i - 3], dt)
            ax[1, i - 3].title.set_text(
                'Magnitud ' + titles[str(i)])
            ax[1, i - 3].set_ylabel('Frecuencia [Hz]')

#    print(imageName)
    plt.xlabel('tiempo [s]')
    plt.savefig(imageName)
    plt.close()
#    plt.show()


def scalogram(values:np.array, frec:float, time:np.array, ax,dt:float, clim:bool=False):
    """
    Auxiliar function to plot the scalogram
    """
    #values = coef[0:,:]
    freq=200/frec
    df = freq[-1] / freq[-2]
    #fig, ax = plt.subplots()
    ymesh = freq#np.concatenate([freq, [freq[-1] * df]])
    #ax.set_yscale('linear')
    ylim = ymesh[[-1, 0]]
    ax.set_ylim(*ylim)
    ax.set_yticks(freq)
    #ax.set_yticklabels([frec[0],frec[-1]])
    xmesh = time#np.concatenate([time, [time[-1] + dt]])
    qmesh = ax.pcolormesh(xmesh, ymesh, values, cmap='jet',shading='gouraud')
    if clim:
        qmesh.set_clim(clim)

    #fig.show()
    return ax