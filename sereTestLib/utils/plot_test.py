#%%
import cv2
import numpy as np
import glob



from sereTestLib.parameters import *
from matplotlib import pyplot
from sereTestLib.autoencoder.DataClassAuto import DataGeneratorAuto



samples_number=[222]


params= {'data_dir' : scalogram_path,
                        'dim': inDim,
                        'batch_size': batch_size,
                        'shuffle': False,
                        'activities':act_ae}



scalograms = DataGeneratorAuto(list_IDs=samples_number, **params)


for i in range(np.shape(scalograms[0][0])[0]):
    pyplot.imshow(scalograms[0][0][ i,:, :, 0], cmap='jet')
    pyplot.savefig(static_path+"Imagenes_prueba/"+str(i)+".png")

# %%
img_array = []

for filename in glob.glob(static_path+'/Imagenes_prueba/*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
out = cv2.VideoWriter(static_path+'/Imagenes_prueba/project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

# %%
