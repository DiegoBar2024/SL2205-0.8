####### IMPORTO LA CARPETA DONDE ESTÁN LOS PARÁMETROS ########
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
##############################################################

import numpy as np
import os,re
import pandas as pd
from natsort.natsort import natsorted
from numpy.core.defchararray import array
from copy import deepcopy
from parameters import time_overlap, dict_actividades
import pywt, scipy
import multiprocessing
from functools import partial
from parameters  import wavelet_library
from matplotlib import pyplot as plt

## Toma como entrada cada una de las submuestras y calcula la transformada de wavelet para cada una de ellas en los seis canales (AC_x, AC_y, AC_z, GY_x, GY_y, GY_z)
def create_segments_scalograms(directorio_segmentos_muestra, directorio_scalogramas,escalas = [8,136], dt = 0.005, girox = [0,0], giroz = [0,0], actividades = None, save_files = True):
    '''
    Each segment is rotated an alpha angle in X and a beta angle in Z axis. 
    Alpha and beta are sorted from uniform distributions based on griox and giroz. The rotations are made by "rotate" function. 
    It generates the scalograms based on sample segments and cuts them to minimize border effects with the function "scalogram_overlap". 
    IMPORTANT: If the folder already exist, this function will not be executed

    Parameters
    ----------
    directorio_segmentos_muestra: str
        Path to original sample segments
    directorio_scalogramas: str
        Destination path
    escalas: list, optional
        The wavelet scales to use in scalogram_overlap. Defaults to [8,136].
    dt: float, optional
        Sampling time. Defaults to 0.005
    girox: list, optional
        Range of the angles (in degrees) of the rotation in X. Defaults to [0,0] (no rotation in X)
     giroz: list, optional
        Range of the angles (in degreeds) of the rotation in Z. Defaults to [0,0] (no rotation in Z)
    actividades: list, optional
        List of activities to process.
        Default to None and in that case all activities are processed
    save_files: bool, optional
        If True, scalograms will be saved in the 'directorio_scalogramas' path as npz files.
        Default is True.
        
    Returns
    -------
    list of dicts
        List that in element 'i' there is a dictionary with the scalogram matrix of segment 'i' and the base name of that segment.
        The format of each dictionary is {'escalograma': array, 'nombre_base_segmento': str}.

    '''
    ## Creo un vector numpy que vaya incrementandose de a una unidad desde escalas[0] hasta escalas[1] - 1
    ## Es decir scales = [escalas[0], escalas[0] + 1, ..., escalas[1] - 1]
    ## Suponiendo <<escalas>> = [8, 136] --> scales = [8, 9, ..., 135]
    scales = np.arange(escalas[0], escalas[1])
    
    ## Se crea el directorio de destino en caso de que aún éste no exista
    ## TODO: Agregar en un log del usuario
    if save_files and (not os.path.exists(directorio_scalogramas)):
        os.makedirs(directorio_scalogramas)

    #TODO: Es ver como hacer para no guardar un archivo por segmento. Si es que ocupan mucho
    #TODO: Guardar solo scalogramas de las actividades dinamicas

    ## Lo que hago acá es listar todos los directorios (archivos, carpetas) que tengo dentro de <<directorio_segmentos_muestra>> y los ordeno de forma natural
    archivos_segmentos = natsorted(os.listdir(directorio_segmentos_muestra))

    ## En caso que el campo actividades (lista de nombres de actividades) sea no nulo
    if actividades is not None:

        ## Creo una tupla cuyos elementos sean los valores asociados a las actividades que pasé como lista de entrada, en el diccionario <<dict_actividades>> (código de numeros)
        actividades = tuple([dict_actividades.get(actividad) for actividad in actividades])

        ## Creo una lista de cadenas cuyos elementos son los nombres de los archivos en el directorio <<directorio_segmentos_muestra>> cuyo nombre comience con alguno de los números en la tupla de actividades que hice antes
        archivos_segmentos = [file for file in archivos_segmentos if file.startswith(actividades)]

    # Busco el tipo de adquisición S para corta y L para larga
    tipo_adquisicion = re.split(r'\d',archivos_segmentos[0])[not""]

    # La info del archivo extra no va a estar más en un archivo. Va a ser deprecado
    if tipo_adquisicion.upper() == 'S':

        nombre_base =  re.split(r'\D',archivos_segmentos[0])
        archivo_info_extra='0S%s' % (nombre_base[1])

    else:
        archivo_info_extra= ""

    ## Verificar que hay muestras de las actividades
    ## Genero el path completo al archivo
    archivo_entrada = '%s%s' % (directorio_segmentos_muestra, archivos_segmentos[0])

    ## <<dt>> es el período de muestreo que va a tomarse como 0.005 segundos
    ## <<time_overlap>> es el tiempo de solapamiento que según <<parameters.py>> es igual a 2
    ## Con estos valores se tiene <<cant_muestras_overlap>> = 400
    cant_muestras_overlap = int(time_overlap / dt)

    # TODO: Loggear si no hay datos de una actividad
    # Tupla de actividades en numerico
    # Creo una tupla cuyos elementos sean los valores asociados a las actividades que pasé como lista de entrada, en el diccionario <<dict_actividades>> (código de numeros)
    actividades = tuple([dict_actividades.get(actividad) for actividad in dict_actividades])

    # Selección de sub caso de actividad
    # Asumo que el primer numero es la actividad (o el texto, veremos)
    ## Creo una lista de cadenas cuyos elementos son los nombres de los archivos en el directorio <<directorio_segmentos_muestra>> cuyo nombre comience con alguno de los números en la tupla de actividades que hice antes
    archivos_segmentos_dinamicos = natsorted([file for file in archivos_segmentos if file.startswith(actividades)])

    ## Paso <<giroz>> a radianes. Por defecto giroz = 0 según la entrada a la función que tengo
    giroz = np.dot(np.pi/180, giroz)

    ## Paso <<girox>> a radianes. Por defecto girox = 0 según la entrada a la función que tengo
    girox = np.dot(np.pi/180, girox)

    ## <<thetaz>> va a ser el vector aleatorio cuyos elementos son V.As de distribución uniforme entre giroz[0] y giroz[1]
    thetaz = np.random.uniform(giroz[0], giroz[1], len(archivos_segmentos_dinamicos))

    ## <<thetax>> va a ser el vector aleatorio cuyos elementos son V.As de distribución uniforme entre girox[0] y girox[1]
    thetax = np.random.uniform(girox[0],girox[1], len(archivos_segmentos_dinamicos))

    # En la entrada 'i' habrá un diccionario que contenga la matriz de
    # escalogramas del segmento 'i' y el nombre base de ese segmento
    ## Se crea una lista vacía
    escalogramas_segmentos = []
    
    ## Se realiza una iteración sobre una estructura enumerate con DOS VARIABLES LOCALES distintas en la iteración i y archivo_segmento
    ## La variable i va a tomar los valores de los índices es decir i = 0, 1, ..., len(archivos_segmentos_dinamicos) - 1
    ## La variable archivo_segmento va a tomar los valores de los elementos de la lista <<archivos_segmentos dinámicos>> que son los nombres de los ficheros .csv que contienen los segmentos a examinar
    for i, archivo_segmento in enumerate(archivos_segmentos_dinamicos):

        ## Se concatenan <<directorio_segmentos_muestra>> y <<archivo_segmento>> para obtener la ruta completa al segmento .csv
        archivo_entrada = '%s%s' % (directorio_segmentos_muestra, archivo_segmento)

        ## Se elimina la extensión .csv del nombre del fichero csv del segmento dejando sólamente el nombre normal. Ej: '2S112.csv' --> '2S112'
        nombre_base_segmento = archivo_segmento.replace('.csv','')

        ## Se hace la lectura completa del .csv y se lo almacena en un DataFrame Pandas
        segmento = pd.read_csv(archivo_entrada, low_memory = False)

    ##########################################
    # Funcion/es de ejecuion: TODO: Separar en funciones para pasar a APP
    #########################################
        #age, pathology = get_age_pathology(archivo_info_extra)
        
        # Como no se utiliza get_age_pathology, seteo pathology a 0.
        pathology = 0

        ## Por defecto no tenemos corrección de ejes por los giros
        segmento = rotate(thetaz = thetaz[i],thetax = thetax[i], xs = segmento)

        ## Acá toma de los datos preprocesados la lista de los valores de aceleraciones y giros en los ejes y los pasa a una numpy array para luego transformarlo
        datosACx = np.array(segmento.AC_x)
        datosACy = np.array(segmento.AC_y)
        datosACz = np.array(segmento.AC_z)
        datosGYx = np.array(segmento.GY_x)
        datosGYy = np.array(segmento.GY_y)
        datosGYz = np.array(segmento.GY_z)

        ## Se realiza la descomposición en wavelet de la señal dada por las aceleraciones en el eje X
        coefACx = scalogram_overlap(dt, datosACx, scales, cant_muestras_overlap, wavelet_library)

        ## Creo un tensor tridimensional vacío el cual tenga las dimensiones espaciales de los coeficientes de las wavelets
        ## <<matriz_scalogramas>> me va a almacenar en cada uno de los 6 niveles las matrices de los coeficientes de la transformada de wavelet para cada una de las señales de aceleración y giros
        ## O sea que los niveles van a ser AC_x, AC_y, AC_z, GY_x, GY_y, GY_z en ese orden
        matriz_scalogramas = np.empty((coefACx.shape[0], coefACx.shape[1], 6))

        ## El primer nivel del tensor va a estar dado por los coeficientes de la señal de aceleración en el eje x
        matriz_scalogramas[:,:,0] = coefACx

        ## En caso de que la <<wavelet_library>> sea 'ssqueezepy' se entra a lo siguiente
        # if wavelet_library == 'ssqueezepy':
        #     matriz_scalogramas[:,:,1] = scalogram_overlap(dt, datosACy, scales, cant_muestras_overlap, wavelet_library=wavelet_library)
        #     matriz_scalogramas[:,:,2] = scalogram_overlap(dt, datosACz, scales, cant_muestras_overlap, wavelet_library=wavelet_library)
        #     matriz_scalogramas[:,:,3] = scalogram_overlap(dt, datosGYx, scales, cant_muestras_overlap, wavelet_library=wavelet_library)
        #     matriz_scalogramas[:,:,4] = scalogram_overlap(dt, datosGYy, scales, cant_muestras_overlap, wavelet_library=wavelet_library)
        #     matriz_scalogramas[:,:,5] = scalogram_overlap(dt, datosGYz, scales, cant_muestras_overlap, wavelet_library=wavelet_library)

        matriz_scalogramas[:,:,1] = scalogram_overlap(dt, datosACy, scales, cant_muestras_overlap, wavelet_library=wavelet_library)
        matriz_scalogramas[:,:,2] = scalogram_overlap(dt, datosACz, scales, cant_muestras_overlap, wavelet_library=wavelet_library)
        matriz_scalogramas[:,:,3] = scalogram_overlap(dt, datosGYx, scales, cant_muestras_overlap, wavelet_library=wavelet_library)
        matriz_scalogramas[:,:,4] = scalogram_overlap(dt, datosGYy, scales, cant_muestras_overlap, wavelet_library=wavelet_library)
        matriz_scalogramas[:,:,5] = scalogram_overlap(dt, datosGYz, scales, cant_muestras_overlap, wavelet_library=wavelet_library)

        ##### CÓDIGO COMO ESTABA ANTES
        ## En otro caso, por ejemplo que se trabaje con la librería 'pywt' se entra a lo siguiente
        ## Se realiza lo mismo que está en el if wavelet_library == 'ssqueezepy' pero 
        ## Las 5 líneas de forma paralela para reducir el tiempo de ejecución.
        ## Con la librería ssqueezepy no se logró hacer funcionar la paralelización
        ## i = 1
        ## datos = [datosACy, datosACz, datosGYx, datosGYy, datosGYz]
        ## Se transforman todos los datos a la vez y se los guarda en la matriz de escalogramas
        ## with multiprocessing.Pool() as pool:

            # print(pool.map(partial(scalogram_overlap, dt, scales = scales, overlapS = cant_muestras_overlap, wavelet_library = wavelet_library), datos))
            # for result in pool.map(partial(scalogram_overlap, dt, scales = scales, overlapS = cant_muestras_overlap, wavelet_library = wavelet_library), datos):
            #     matriz_scalogramas[:,:,i] = result
            #     i += 1
                
        if save_files:
            archivo_salida = '%s%s' % (directorio_scalogramas, nombre_base_segmento)
            for i, dato in zip(range(6), ['ACx', 'ACy', 'ACz', 'GYx', 'GYy', 'GYz']):
                np.savez_compressed(archivo_salida + str(dato) + '.npz', y=pathology, X=matriz_scalogramas[:,:,i])

        escalogramas_segmentos.append({'escalograma': matriz_scalogramas, 'nombre_base_segmento': nombre_base_segmento})
    
    return escalogramas_segmentos

    #        if 3:
    #            np.savez_compressed(directorio_scalogramas + "/Escalera" +  + '.npz', y=pathology, X=x)
    #            cam = True #Haymuestras de caminata
    #        if 4:
    #            np.savez_compressed(directorio_scalogramas + "/Caminando" +  + '.npz', y=pathology, X=x)
    #            np.savez_compressed(directorio_scalogramas + "/Caminando" +  + '.npz', y=pathology, X=x)
    #            esc = True #Hay muestras de escalera

## IMPORTANTE: IMPLEMENTACIÓN DE LA TRANSFORMACIÓN EN ESCALOGRAMAS
## Ésta es la función que genera los escalogramas en base a los segmentos de entrada
## Luego los recorta para minimizar los efectos de borde
## El argumento <<wavelet_library>> me va a decir la librería que se va a usar para hacer la transformada de wavelet. Por defecto en el fichero <<parámeters>> está la librería <<pywt>>
def scalogram_overlap(dt:float, signal:np.ndarray, scales:array, overlapS:int, wavelet_library:str)->np.array:
    """
    This function generates the scalograms based on sample segments and cuts them to minimize border effects.
    
    Parameters
    ----------
    dt : float
        Time difference between two samples

    signal : ndarray
        Signal to process

    scales : array
        Scales range

    overlapS : int
        Overlap seconds in samples
    
    wavelet_library : str
        Library that performs the wavelet transform. 
        Options: 'pywt', 'scipy', 'ssqueezepy'

    Returns
    -------
    ndarray
        Scalogram without border effects, by using overlapped windows

    See Also
    -------
        https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
    """

    ## Dependiendo de la librería que yo haya aclarado como argumento de la función, tengo diferentes formas de hacer la transformada de wavelet
    ## Por defecto la librería que se usa acá es <<pywt>> o PyWavelet pero también hay otras librerías que también implementan ésta transformada como <<scipy>> o <<ssqueezepy>>
    ## Para el caso de <<PyWavelet>> ejecuto el siguiente bloque
    if wavelet_library == 'pywt':

        ## IMPORTANTE
        ## La función <<pywt.cwt>> es la que implementa la transformada CONTÍNUA de Wavelet
        ## ENTRADAS
        ##   - <<data>> es la señal de entrada que se va a transformar. Para nosotros, las señales de aceleración o giroscopios. A nivel de programación es un numpy array que se toma como entrada.
        ##   - <<scales>> va a ser el rango de escalas que se va a usar en las wavelets. A nivel de programación es un vector de números (enteros o tipo flotante)
        ##   - <<wavelet>> es el nombre de la mother wavelet que se va a usar para hacer la descomposición. Hay varias opciones. Acá ya está puesta <<cmor1.5-1>> la cual es la Complex Morlet Wavelet
        ##   - <<sampling_period>> es igual a la diferencia temporal entre dos muestras. Ésto es, el período de muestreo. A nivel de programación es un número flotante. Por defecto son 0.005s que son 200Hz.
        ## SALIDAS
        ##   - <<coefs>> me va a dar un array bidimensional de dimensiones (CANTIDAD DE ESCALAS) x (CANTIDAD DE MUESTRAS TEMPORALES) de los coeficientes COMPLEJOS correspondientes a la wavelet a cada una de las escalas que definí y en cada uno de los instantes de tiempo
        ##   Vale la pena recordar que la transformada que se usa acá es contínua de modo que el algoritmo usa métodos de integración numérica para resolver la integral de la transformada de wavelet
        ##   - <<scales_freq>> me da la lista de frecuencias de cada escala. O sea que éste es un vector unidimensional de largo (CANTIDAD DE ESCALAS)
        coef, scales_freq = pywt.cwt(data = signal, scales = scales, wavelet = 'cmor1.5-1', sampling_period = dt)
    
    ## En caso de usar <<Scipy>> para hacer la transformada de wavelet ejecuto el siguiente bloque
    if wavelet_library == 'scipy':
        coef= scipy.signal.cwt(signal, wavelet=scipy.signal.morlet2, widths=scales)

    ## Graficación del escalograma en el i-ésimo paso
    data = np.abs(coef) ** 2
    cmap = plt.get_cmap('jet', 256)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    t = np.arange(coef.shape[1]) * dt
    ax.pcolormesh(t, scales_freq, data, cmap=cmap, vmin=data.min(), vmax=data.max(), shading='auto')
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")
    plt.title("$|CWT(t,f)|^2$")
    plt.show()

    ## En caso de usar <<Ssqueezepy>> para hacer la transformada de wavelet ejecuto el siguiente bloque
    # if wavelet_library == 'ssqueezepy':
    #     wavelet = ssqueezepy.wavelets.Wavelet(wavelet=('morlet', {'mu': 5}), dtype='float64')
    #     coef, _ = ssqueezepy.cwt(signal, wavelet, scales=scales.astype('float64'), fs=1/dt)

    ## <<np.abs(coef[:, overlapS: -overlapS])>> me selecciona TODAS LAS FILAS de la matriz de coeficientes, es decir todas las escalas, tomando el módulo de cada una de las entradas
    ## Ésta parte es importante porque acá se recorta temporalmente los coeficientes generados. NO SE RECORTA LA ESCALA o sea la escala sigue siendo la misma que la entrada
    ## Lo que hago es RECORTAR una cantidad <<overlapS>> de MUESTRAS TEMPORALES en el BORDE IZQUIERDO y en el BORDE DERECHO POR IGUAL.
    ## O sea si por ejemplo tengo una muestra de 10 segundos y tengo <<overlapS>> = x muestras las cuales me representan 1 segundo, obtengo como resultado los coeficientes de la muestra recortada de 8 segundos en el CENTRO
    ## Luego al aplicarle el método <<astype(np.intc)>> lo que hago es DISCRETIZAR el valor de los módulos de los coeficientes (ya con los recortes temporales) a números ENTEROS POR TRUNCAMIENTO (y no por redondeo)
    values = np.array(np.abs(coef[:, overlapS:-overlapS]) * 100).astype(np.intc)

    return values

def rotate(thetaz,thetax,xs):
    """
    Does the rotations with the functions "rotate_Z" and "rotate_X".
    
    Parameters
    ----------
    thetaz: float
        Angle (in radians) to rotate the segment in Z axis.
    thetax: float
        Angle (in radians) to rotate the segment in X axis.
    xs: Segment to rotate

    Returns
    -------
    dfX: Rotated segment
    """
    ## Aplico rotaciones en los distintos planos de los ángulos correspondientes
    dfZ = rotate_Z(thetaz,xs)
    dfX = rotate_X(thetax,dfZ)

    return(dfX)

def rotate_Z(angle, xs):
    """
    Rotates the segment in Z axis. 

    Parameters
    ----------
    angle: float
        Angle (in radians)
    xs: Segment to rotate

    Returns
    -------
    dfX: Rotated segment in Z axis.

    """
    ## Hago una copia del segmento a rotar (el segmento va a ser un DataFrame Pandas)
    df = deepcopy(xs)

    ## Selecciono las dos aceleraciones en los ejes x e y (columnas 'AC_x' y 'AC_y')
    x = df.AC_x
    y = df.AC_y

    ## Selecciono los dos giros en los ejes x e y (columnas 'GY_x', 'GY_y')
    gx = df.GY_x
    gy = df.GY_y

    ## Construyo las nuevas aceleraciones x e y luego de aplicar la rotación de ángulo <<angle>> a las aceleraciones originales
    new_x = np.cos(angle) * x + np.sin(angle) * y
    new_y = - np.sin(angle) * x + np.cos(angle) * y

    ## Construyo los nuevos giros gx y gy luego de aplicar la rotación de ángulo <<angle>> a los giros originales
    new_gx = np.cos(angle) * gx + np.sin(angle) * gy
    new_gy = - np.sin(angle) * gx + np.cos(angle) * gy

    ## Reasigno las aceleraciones y los giros para construir el nuevo DataFrame
    df.AC_x = new_x
    df.AC_y = new_y
    df.GY_x = new_gx
    df.GY_y = new_gy

    return df

def rotate_X(angle, xs):
    """
    Rotates the segment in X axis. 

    Parameters
    ----------
    angle: float
        Angle (in radians)
    xs: Segment to rotate

    Returns
    -------
    dfX: Rotated segment in X axis.

    """
    ## Hago una copia del segmento a rotar (el segmento va a ser un DataFrame Pandas)
    df = deepcopy(xs)

    ## Selecciono las dos aceleraciones en los ejes z e y (columnas 'AC_z' y 'AC_y')
    z = df.AC_z
    y = df.AC_y
    
    ## Selecciono los dos giros en los ejes z e y (columnas 'GY_z', 'GY_y')
    gz = df.GY_z
    gy = df.GY_y

    ## Construyo las nuevas aceleraciones z e y luego de aplicar la rotación de ángulo <<angle>> a las aceleraciones originales   
    new_z = np.cos(angle) * z + np.sin(angle) * z
    new_y = - np.sin(angle)*z + np.cos(angle) * y

    ## Construyo los nuevos giros z e y luego de aplicar la rotación de ángulo <<angle>> a los giros originales  
    new_gz = np.cos(angle)*gz + np.sin(angle) * gy
    new_gy = -np.sin(angle)*gz + np.cos(angle) * gy

    ## Reasigno las aceleraciones y los giros para construir el nuevo DataFrame
    df.AC_z=new_z
    df.AC_y=new_y
    df.GY_z=new_gz
    df.GY_y=new_gy

    return df