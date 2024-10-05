####### IMPORTO LA CARPETA DONDE ESTÁN LOS PARÁMETROS ########
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
##############################################################

import pandas as pd
import shutil
from natsort.natsort import natsorted
import os,re
from parameters import *

def rename_segments(pat, dir = dir_out_ov_split,dir_rename = dir_out_ov_split_rename, dir_out_new_char=dir_out_new_char, long_sample = False):
    """
    Renames segments according to the inferred activities.


    Parameters
    ----------
    pat: int
        sample id number
    dir: str, optional
        Path to the segments to be renamed.
    dir_rename: str, optional
        Output path
    long_sample: bool, optional
        Defaults to False

    Extra
    -----
    dir_out_new_char: str
        It is imported from parameters.py.  Indicates the path to the files containing the inferred activities.



    """
    ## En caso que se trate de una muestra larga, asigno a la variable <<character>> el valor 'L'
    if long_sample:
        character = 'L'

    ## En caso que se trate de una muestra corta, asigno a la variable <<character>> el valor 'S'
    ## Por defecto la variable <<long_sample>> está seteada en FALSE (ver <<parameters.py>>) se trata de una muestra corta
    else:
        character = 'S'
    
    ## En <<dir_out_new_char>> tengo una cadena donde tengo el path donde voy a almacenar el archivo de salida
    ## En <<paciente>> yo voy a guardar aquella ruta que esté dentro de la nueva carpeta
    paciente = dir_out_new_char + '/' + character + str(pat) + '/'

    ## <<archivos_caracteristicas>> va a ser una lista de cadenas que contenga todos los archivos/directorios que se encuentren dentro del path de <<paciente>>
    archivos_caracteristicas = natsorted(os.listdir(paciente))

    ## Especifico cual es la nueva dirección
    new_dir = dir_rename + '/' + character + str(pat) + '/'

    ## Crear directorio de salida de datos
    ## En caso de que el directorio de salida no exista, voy a crearlo
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    ## Itero archivo por archivo en la lista de archivos en <<archivos_caracteristicas>>
    for file in archivos_caracteristicas:

        ## Hago la lectura del segmento del paciente y lo almaceno en una estructura de tipo DataFrame Pandas
        df = pd.read_csv(paciente + file)

        for k in range(df.shape[0]):
            oldname = dir + '/' + character + str(pat) + '/' + df.file_name[k]
            newname = new_dir + df.new_name[k]
            shutil.copy(oldname, newname)