#%%
import cv2
import numpy as np
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import natsort





path_video='proyecto_111.avi'
img_array = []

for filename in natsort.natsorted(glob.glob("/home/sere/Dropbox/111_estable/"+'*.png')):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
out = cv2.VideoWriter(path_video ,cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

#%%