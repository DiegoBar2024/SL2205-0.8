""" Library for process raw data and characteristic extraction

    Author
    --------
    Mariana del Castillo: <mdelcastillo@proyectos.com.uy>

    Functions
    --------

    Characteristic extraction per channel functions:
    'meansAceleration', 'meansGyro', 'varsAceleration', 'varsGyro',
    'stdAceleration', 'stdGyro'

    Sagital plane energy characteristic extraction
    'ACenergy', 'energyGyro'

     Auxiliar functions
    'medianFilter','toFloat', 'getAgePathology', 'splitSamples', 'splitSamplesOverlap',
    'fftFreq1', 'ACxfreq', 'ACyfreq', 'ACzfreq', 'correlations', 'interpolate'

    See also
    --------
    paramHandler/attributes.py

    Examples
    --------
        >>> lst= [ 'meanAC_x', 'meanAC_y', 'meanAC_z',
              'Age', 'State', 'Patology']


        >>> file_in='<route_to_file>/features/<patient>/<state>caracteristicas.hdf5'
        >>> X,y=data_generation(list_atrib=lst,  data_in=file_in)
        >>> print(X)
    """

####### IMPORTO LA CARPETA DONDE ESTÁN LOS PARÁMETROS ########
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
##############################################################

import pandas as pd
import numpy as np
from scipy.interpolate import splev, splrep
import os, re
from natsort.natsort import natsorted
from parameters  import fs

## Agarro muestra entera (un archivo .csv) y lo parto en varias muestras
def split_and_save_samples_overlap(ruta_muestra_paciente, ruta_segmentos_csv, time_frame, time_overlap, dt, step=3):
    """
    Function that reads each fragment of a sample, cuts them with overlap. If the segment
    is valid with respect to the activities, the segment is saved in a csv.
    Each segment is stored as the fragment name and the segment number.
    
    Parameters:
    -----------
    ruta_muestra_paciente : str
        Path to the fragments of the sample
    ruta_segmentos_csv : str
        Output path 
    time_frame : num
        Time duration for the segments
    time_overlap : num
        Overlap time
    dt : num
        Sampling time
    Returns:
    --------
    lista_descarte : list
        List with the csv that were not saved because they were not valid
    n_segmentos_descartados_acel:
        number of discarded segments
    """

    ## Creo una lista vacía llamada <<lista_descarte>>
    lista_descarte = []

    ## En <<files>> voy a guardar la lista de directorios (archivos, carpetas) que se encuentran en la ruta dada por <<ruta_muestra_paciente>>
    ## En <<ruta_muestra_paciente>> yo voy a tener aquellos fragmentos (que comienzan con 0) que voy a procesar
    files = os.listdir(ruta_muestra_paciente)

    ## En <<fragmentos>> voy a guardar aquellos archivos (ordenados de forma natural) cuyo nombre no comience con 0
    fragmentos = natsorted([file for file in files if not file.startswith('0')])

    ## Se crea el directorio de salida de los datos en caso de que no esté creado aún
    ## TODO: Agregar en un log del usuario
    if not os.path.exists(ruta_segmentos_csv):
        os.makedirs(ruta_segmentos_csv)
    
    # TODO: Al archivo de extra data no hay que splitearlo En un futuro no debería existir más
    ## Se usa la función re.split para poder obtner como resultado si la muestra es L (Long) o S (Short)
    ## En <<tipo_adquisición>> voy a guardar un caracter 'L' en caso que la muestra sea LARGA y 'S' en caso que la muestra sea CORTA
    tipo_adquisicion = re.split(r'\d', fragmentos[0])[not""]

    ## Itero para cada uno de los fragmentos en la lista de fragmentos
    ## Recuerdo que <<fragmentos>> guarda una lista de cadenas donde dichas cadenas son los nombres de cada uno de los fragmentos que voy a procesar
    for fragmento in fragmentos:

        ## Hago una lectura del fragmento csv del paciente y lo guardo en un DataFrame
        data = pd.read_csv(ruta_muestra_paciente + fragmento, low_memory = False)

        ## En <<estado>> voy a almacenar el primer caracter del fragmento
        estado = re.split(r'\D', fragmento)[0]

        if tipo_adquisicion.upper() == 'S' and str(estado) =='0':

            ## El archivo de datos extra se guarda sin modificar
            fileobject = open(ruta_segmentos_csv+ fragmento,'w')

        else:
            segmentos,_ = split_samples_overlap(data, time_frame, time_overlap, dt, step)
            for i,segmento in enumerate(segmentos):
                validez, explicacion = chequear_aceleracion(segmento)
                fragmento = fragmento.replace('.csv','')
                if validez:
                    nombre_archivo = ruta_segmentos_csv+ fragmento + "s" +str(i)+'.csv'
                    pd.DataFrame(segmento).to_csv(nombre_archivo, index=False)
                else:
                #TODO: agregarlo en el LOG junto con la explicacion
                    lista_descarte.append(fragmento + "s" +str(i))
                    explicacion #Esto va a ir en el log

    n_descarte = np.shape(lista_descarte)[0]
    return lista_descarte, n_descarte

# Funcion auxiliar para separar las muestras en ventanas de tiempo + overlap
def split_samples_overlap(data, time_frame, time_overlap, dt, step):
    """
    Split the raw data into frames with overlap

    Parameters
    ----------
    data : ndarray
        Original raw data
    time_frame : int
        Frame time duration

    time_overlap : int
        Overlap time duration

    dt : float
        Time difference between two samples en SEGUNDOS

    Returns
    -------
    ndarray
        Splited signal

    """
    # TODO: Corroborar que efectivamente esta es la dt falta decimar en caso de tener una fs superior
    # Tolerancia de error en fs (porcentual)
    tol = 0.1

    ## Especifico la longitud total de la muestra que tengo como entrada
    ## Considerando que <<data>> es un DataFrame Pandas al hacer <<data.shape[0]>> simplemente lo que hago es contar la cantidad de muestras temporales que tengo (longitud temporal de la muestra)
    total_sample = data.shape[0]

    ## Calculo el PERÍODO DE MUESTREO en segundos de los datos del sensor
    ## Para ésto calculo la longitud temporal total de la muestra "data.Time.iloc[-1] - data.Time.iloc[0]" y luego la divido entre la longitud de la muestra
    dt_real = ((data.Time.iloc[-1] - data.Time.iloc[0]) / 1000) / total_sample

    ## <<1 / dt_real>> es igual a la FRECUENCIA DE MUESTREO de la muestra que estoy procesando (inverso del período de muestreo)
    ## Se entra al if en caso de que se cumpla AL MENOS UNA DE LAS DOS CONDICIONES SIGUIENTES:
    ## i) <<1 / dt_real > fs + fs * tol>> me da TRUE en caso de que la frecuencia de muestreo de la muestra que estoy procesando sea mayor a un 10% de la frecuencia de referencia (200Hz)
    ## ii) <<1/dt_real <<fs - fs * tol>> me da TRUE en caso de que la frecuencia de muestreo de la muestra que estoy procesando sea menor a un 10% de la frecuencia de referencia (200Hz)
    ## En resumen, se entrará al bloque if en caso de que la frecuencia de muestreo de la muestra actual no pertenezca a un rango (180, 220) Hz
    if (1 / dt_real > fs + fs * tol) or (1 / dt_real < fs - fs * tol):
        
        ## Hacemos remuestreo de los datos a 200Hz
        data = interpolate(data)

        ## Longitud de la muestra
        total_sample = data.shape[0]

    ## <<time_frame>> va a ser la ventana temporal que voy a usar para recortar los datos
    ## <<dt>> es la diferencia temporal entre dos muestras sucesivas
    ## <<samplesPerSplit>> me va a dar la cantidad de muestras por cada ventana de <<time_frame>>
    ## Tomando <<time_frame>> = 3 y <<dt>> = 0.005 (ver parameters.py) --> <<samplesPerSplit>> = 600 de modo que tenemos 600 muestras cada 3 segundos
    samplesPerSplit = int(time_frame / dt)

    ## <<samplesOverlap>> me va a dar la cantidad de muestras por cada ventana de <<time_overlap>>
    ## Tomando <<time_overlap>> = 2 y <<dt>> = 0.005 (ver parameters.py) --> <<samplesOverlap>> = 400 de modo que tenemos 400 muestras cada 2 segundos
    samplesOverlap = int(time_overlap / dt)

    ## <<samplesStep>> me va a dar la cantidad de muestras por cada ventana de <<step>>
    ## Tomando <<step>> = 3 y <<dt>> = 0.005 (ver parameters.py, step es por defecto 3) --> <<samplesStep>> = 600 de modo que tenemos 600 muestras cada 3 segundos
    samplesStep = int(step / dt)

    ## Defino lista vacía
    x_sub = []

    ## <<total_sample>> me daba la cantidad TOTAL de muestras que tengo
    ## <<maxSplit>> se interpreta como la cantidad de ventanas de tamaño <<samplesStep>> COMPLETAS que se pueden llenar cuando a las muestras totales se descuenta el doble de la cantidad de las muestras de overlap
    maxSplit = int((total_sample - 2 * samplesOverlap ) // samplesStep)

    ## Itero para i en un rango i = 0, 1, ..., maxSplit - 1
    for i in range(maxSplit):

        ## Dentro del append lo que se hace es recortar el DataFrame <<data>> comenzando desde el índice '(i * samplesStep)' hasta '(i  * samplesStep + 2 * samplesOverlap + samplesPerSplit)' sin incluír
        ## Se genera una lista con las muestras cortadas y solapadas
        x_sub.append(data[(i * samplesStep): i  * samplesStep + 2 * samplesOverlap + samplesPerSplit])

    return x_sub, samplesOverlap

## Función que se usa para aumentar la resolución de la señal cuando la frecuencia de muestreo es MENOR a 200Hz
## La entrada <<df>> es el DataFrame Pandas con mis datos
def interpolate(df):
    """
    Used to increase signal resolution when fs < 200Hz
    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    DataFrame
        Complete interpolated dataframe.
    """
    ## Creo un nuevo DataFrame vacío el cual tenga como columnas aquellas columnas del dataFrame de entrada
    X = pd.DataFrame(columns = df.columns)

    ## <<columns>> son las columnas del dataFrame de entrada las cuales fueron seteadas al hacer el procesamiento (sin incluír la columna de Tiempo)
    columns = ['AC_x', 'AC_y', 'AC_z', 'GY_x', 'GY_y', 'GY_z']

    ## <<df['Time'][-1:]>> es de tipo DataFrame Pandas y contiene el índice final y el timestamp final
    ## <<df['Time'][0]>> es de tipo Float Numpy y contiene únicamente el timestamp inicial
    ## <<t_final>> va a ser el resultado de dividir entre 1000 la diferencia entre ambos valores y se queda con el cociente de la división
    t_final = (df['Time'][-1:] - df['Time'][0]) // 1000

    ## <<len(df['Time'])>> me va a dar la cantidad de muestras temporales que saqué, es decir la longitud de la muestra temporal
    f = len(df['Time']) // t_final

    ## Tomo la lista de timestamps que tengo y los copio usando el método <<.copy()>>
    ## Obteno entonces un nuevo DataFrame Pandas el cual contenga los índices y los respectivos timestamps
    time50 = df['Time'].copy(deep = True)

    ## Asigno la variable <<cont>> a 0
    cont = 0

    ## Itero en el rango de los timestamps. Es decir i = 0, 1, ..., n - 1, n
    ## La idea es armar un vector de tiempos de la misma cantidad de elementos que comience en 0 y tenga la frecuencia f que calculé arriba
    for i in range(0, len(df['Time'])):

        ## En <<interval>> almaceno el valor numérico del nuevo período de muestreo generado a partir de la nueva frecuencia de muestreo f
        ## Recuerdo que la f calculada arriba va a ser una frecuencia MAYOR a la frecuencia de muestreo de la señal original
        interval = 1 / f

        ## <<time50.at[i]>> me va a contener el i-ésimo instante temporal de mi nuevo vector de tiempos
        ## La fórmula general me va a quedar time50.at[i] = i / f, para todo i = 0, 1, ..., n - 1, n
        time50.at[i] = cont

        ## Actualizo el valor de <<cont>> con el intervalo según el período de muestreo
        cont = cont + interval

    ## Creo un vector de tiempos de frecuencia 200Hz el cual vaya de 0 hasta el instante <<t_final>>
    time200 = np.arange(0, int(t_final), 0.005)

    ## Asigno a la columna <<Time>> del DataFrame X vector de tiempos <<time200>> a la frecuencia de 200Hz
    X['Time'] = time200

    ## Itero columna por columna para cada una de las columnas
    for elem in columns:

        spl = splrep(time50, df[elem])
        y = splev(time200, spl)
        y = pd.DataFrame(y)
        X[elem] = y

    X['Time'] = X['Time'].apply(lambda x: x * 1000)

    return X

def chequear_aceleracion(segmento):
    """
    Checks if the segments axes are consistent with the expected activities.

    
    Y retorna un booleano indicando el resultaod y un string indicando el motivo
    Parameters
    ----------
    segmento : matrix
        
    Returns
    ------
    inconsistencia: str
        Reurns de reason to the inconsistency.
    coherente : boolean
        Returns True if the accelerations are consistent or False if they are not.
    """
    inconsistencia = 'Funcion sin desarrollar'
    coherente = True

    return coherente, inconsistencia
