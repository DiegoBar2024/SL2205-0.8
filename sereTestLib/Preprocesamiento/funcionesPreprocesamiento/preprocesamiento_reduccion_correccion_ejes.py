####### IMPORTO LA CARPETA DONDE ESTÁN LOS PARÁMETROS ########
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
##############################################################

import pandas as pd 
import os
from statistics import mean 
from parameters import long_sample, path_log_preprocesamiento, dict_actividades
import parameters
from natsort.natsort import natsorted

## Agarra los datos "crudos" del sensor y los preprocesa dejando un archivo .csv
def preprocesamiento_reduccion_correccion_ejes(ruta_muestra_cruda,ruta_muestra_preprocesada,is_long_sample=long_sample, actividades=None):
    '''
    
    Performs the reduction of the corresponding raw sample and, if necessary, corrects the axes.
    Requirements:
        * the raw sample is segmented by size or activity.
        * there are no mixed samples corresponding to long and short samples.
    
    Authors
    ------
    Carla Rodriguez
    Mariana del Castillo - mariana.delcastillo@proyectos.com.uy
    Francisco de Izaguirre - francisco.de.izaguirre@proyectos.com.uy

    Parameters
    ----------
    ruta_muestra_cruda : str
        Path to the raw sample
    ruta_muestra_preprocesada : str
        Output path
    is_long_sample: bool
        True if it is a long sample
    actividades: list, optional
        List of activities to process. 
        The default is None, in which case all existing activities are processed
    '''

    ## <<os.listdir>> lo que hace es listarme los directorios (o sea archivos y carpetas) que tengo en una determinada ruta pasada como parametro
    ## El comando natsorted lo que hace es ordenar un iterable de forma NATURAL
    fragmentos = natsorted(os.listdir(ruta_muestra_cruda))

    # Si se utiliza el argumento 'actividades', solo esas se preprocesan
    # En caso que el campo 'actividades' sea no nulo, yo quiero que únicamente éstas sean procesadas
    if actividades is not None:

        ## Se crea una tupla donde se guardan los valores de dict_actividades asociados a cada una de las actividades de la lista
        ## O sea que <<actividades>> me va a quedar una tupla cuyos elementos son los valores asociados a las actividades de la lista <<actividades>> en el diccionario <<dict_actividades>> 
        actividades = tuple([dict_actividades.get(actividad) for actividad in actividades])

        ## Lo que hago acá es generar una lista cuyos elementos sean el nombre de los fragmentos (ficheros) los cuales contengan datos de la(s) actividad(es) deseada(s)
        ## Según el caracter con el que comienza el nombre de un fragmento se hace referencia a una actividad en específico
        ## Las actividades se indexan de la siguiente manera:
        ## 1 -- Sentado
        ## 2 -- Parado
        ## 3 -- Caminando
        ## 4 -- Escalera
        fragmentos = [fragmento for fragmento in fragmentos if fragmento.startswith(actividades)]

    # En caso que no exista una ruta para la muestra preprocesada, voy a crearla
    # TODO: Agregar en un log del usuario
    if not os.path.exists(ruta_muestra_preprocesada):
        os.makedirs(ruta_muestra_preprocesada)

    ## <<long_sample>> me dice si la muestra con la que estoy trabajando es una muestra larga. Tiene valor False que está especificado en Parametros
    ## Para nuestra primera fase del proyecto vamos a trabajar únicamente con muestras cortas, de modo que ésto en principio no es necesario
    ## Si es muestra larga se reduce y  verifica el primer fragmento.
    if is_long_sample:
        extraer_columnas(ruta_muestra_cruda + fragmentos[0], ruta_muestra_preprocesada + fragmentos[0] )
        correccion = chequear_ejes(ruta_muestra_preprocesada + fragmentos[0])
        if not (correccion == "Ninguna" or correccion == "Incorrecto"):  #TODO: Incorrecto no debería estar, porque deberia lanzarse una excepcion.
                                                                        # Ejecutar "Ninguna" no realiza accion, pero el "if" para evitar ciclos innecesario del "for".
            ejecutar_correccion(correccion,ruta_muestra_preprocesada + fragmentos[0]) # se corrije el primer fragmento
            for fragmento in fragmentos[1::]:  # se reducen y corrigen en resto de los fragmentos de la muestra
                extraer_columnas(ruta_muestra_cruda + fragmento, ruta_muestra_preprocesada + fragmento )
                ejecutar_correccion(correccion,ruta_muestra_preprocesada + fragmento)
        else: # La primer muestra dio que no hay que corregir los ejes, por lo tanto solo se reducen los archivos
            for fragmento in fragmentos[1::]:
                extraer_columnas(ruta_muestra_cruda + fragmento, ruta_muestra_preprocesada + fragmento )

    ## Especifico el procesado en caso de que esté trabajando con muestras cortas
    ## Se esta trabajando con muestras cortas, por lo cual se debe reducir, corroborar y corregir cada una por separado
    else:

        ## Itero para cada uno de los fragmentos de datos en la lista de fragmentos de antes
        ## Recuerdo que <<fragmentos>> va a ser una lista de cadenas cuyos elementos son los nombres de los archivos csv de los datos crudos dentro de las rutas especificadas
        for fragmento in fragmentos:

            ## Se extraen las columnas para obtener los valores de los acelerómetros en los tres ejes y los giroscopios en los tres ejes
            ## Como resultado escribo en un archivo .csv de salida los datos preprocesados. Ésto es, para cada paciente escribo las tres columnas de aceleración y las tres columnas de giros
            extraer_columnas(ruta_muestra_cruda + fragmento, ruta_muestra_preprocesada + fragmento)

            ## Se hace el chequeo de los ejes para dar como salida la corrección
            correccion = chequear_ejes(ruta_muestra_preprocesada + fragmentos[0])

            ## Se hace la corrección de los ejes ejecutando la corrección que corresponda
            ejecutar_correccion(correccion, ruta_muestra_preprocesada + fragmento)

# Extrae las columnas con informacion de sensores de interés
# Se hace una reducción de la correspondiente muestra cruda
# O sea se toma la muestra cruda como entrada y se escribe un nuevo archivo de excel cuyas columnas son las aceleraciones y los giros
def extraer_columnas(ruta_archivo_origen,ruta_archivo_destino):
    '''
    Performs the reduction of the corresponding raw sample.
    It extracts the information of the columns of interest from the original csv and saves it in a new csv file.
    
    Parameters:
    -----------
        ruta_archivo_origen : str
            Path to the original file.

        ruta_archivo_destino:
            Path to the new file.

    Extra:
    ------
        accel_type : str
            It is imported from parameters.py Indicates if the desired acceleration is "wide range" or "low noise"
    ''' 

    ## El valor <<accel_type>> es importado del fichero de parameters.py
    ## Lo que me dice éste parámetro es si la aceleración deseada es del tipo "wide range" (WR) o "low noise" (LN)
    accel_type = parameters.accel_type

    ## Imprimo el parámetro <<accel_type>>
    print(accel_type)

    ## Hago la lectura del csv de origen con el procesamiento correspondiente
    ## <<skiprows=1>> me dice que yo me salteo una línea al inicio de la lectura del archivo
    ## <<sep='\t'>> me dice que se toman las tabulaciones como separadores
    dataframe = pd.read_csv(ruta_archivo_origen, skiprows = 1, sep = '\t')

    ## ACELERACIÓN EN EL EJE X
    ## Creo un dataframe que tenga una columna de nombre "Acc_x" pero que no tenga datos
    Ac_x = pd.DataFrame(columns = ["Acc_x"])

    ## Me quedo únicamente con aquella columna que tenga la lista de datos del acelerómetro en el eje x
    Ac_x = encontrar_columna(dataframe, accel_type + "_X", accel_type + "_X.1")
    
    ## ACELERACIÓN EN EL EJE Y
    ## Creo un dataframe que tenga una columna de nombre "Acc_y" pero que no tenga datos
    Ac_y = pd.DataFrame(columns = ["Acc_y"])

    ## Me quedo únicamente con aquella columna que tenga la lista de datos del acelerómetro en el eje y
    Ac_y = encontrar_columna(dataframe, accel_type + "_Y", accel_type + "_Y.1")
    
    ## ACELERACIÓN EN EL EJE Z
    ## Creo un dataframe que tenga una columna de nombre "Acc_z" pero que no tenga datos
    Ac_z = pd.DataFrame(columns = ["Acc_z"])

    ## Me quedo únicamente con aquella columna que tenga la lista de datos del acelerómetro en el eje z
    Ac_z = encontrar_columna(dataframe, accel_type + "_Z", accel_type + "_Z.1")

    ## GIROS EN EL EJE X
    ## Creo un dataframe que tenga una columna de nombre "Gy_x" pero que no tenga datos
    Gy_x = pd.DataFrame(columns=["Gy_x"])

    ## Me quedo únicamente con aquella columna que tenga la lista de datos del giroscopio en el eje x
    Gy_x = encontrar_columna(dataframe, "Gyro_X", "Gyro_X.1")

    ## GIROS EN EL EJE Y
    ## Creo un dataframe que tenga una columna de nombre "Gy_y" pero que no tenga datos
    Gy_y = pd.DataFrame(columns=["Gy_y"])

    ## Me quedo únicamente con aquella columna que tenga la lista de datos del giroscopio en el eje y
    Gy_y = encontrar_columna(dataframe, "Gyro_Y", "Gyro_Y.1")

    ## GIROS EN EL EJE Z
    ## Creo un dataframe que tenga una columna de nombre "Gy_z" pero que no tenga datos
    Gy_z = pd.DataFrame(columns=["Gy_z"])

    ## Me quedo únicamente con aquella columna que tenga la lista de datos del giroscopio en el eje x
    Gy_z = encontrar_columna(dataframe, "Gyro_Z", "Gyro_Z.1")

    ## INTERVALO TEMPORAL
    ## Creo un dataframe que tenga una columna de nombre "Timestamp" pero que no tenga datos
    Time = pd.DataFrame(columns=["Timestamp"])

    ## Me quedo únicamente con aquella columna que tenga el nombre "Timestamp" o sea que contenga las muestras de tiempo
    Time = encontrar_columna(dataframe, "System_Timestamp_Plot_Zeroed", "Timestamp")

    ## DATOS FINALES
    ## Creo un nuevo dataframe y concateno todas los otros data frames (aceleraciones y giros) para formar la tabla final
    data_final = pd.DataFrame()
    data_final = pd.concat((Time,Ac_x, Ac_y, Ac_z, Gy_x, Gy_y, Gy_z), axis=1, keys=("Time","AC_x", "AC_y", "AC_z", "GY_x", "GY_y", "GY_z"))

    ## Elimino los indices 0 y 1 (filas 0 y 1) de toda la tabla que me queda
    data_final.drop([0,1], axis=0, inplace = True)

    ## Escribo los datos seleccionados en un nuevo archivo csv
    data_final.to_csv(ruta_archivo_destino, index = False)

## Función que me permite encontrar una columna en función de los nombres de columna deseados
## <<dataframe>> es el csv que fue leído usando pd.read_csv
def encontrar_columna(dataframe, nombre1:str, nombre2:str):
    """
    Function to find the calibrated data corresponding to the desired column names.

    Parameters
    ----------
    dataframe: pd.DataFrame
    nombre1: str
    nombre2: str

    """
    ## Creo primero la plantilla tabular para los datos útiles que voy a extraer donde el nombre es una entrada
    ## Es importante que la creación del DataFrame quede por fuera del bucle for
    datos_utiles = pd.DataFrame(columns = [nombre1])

    ## La función <<dataframe.columns>> me da una "lista" del tipo pandas que tiene las cadenas con todos los nombres de mis columnas
    ## Como es una estructura iterable yo puedo ir elemento por elemento leyendo nombre por nombre de mi columna
    ## Ésto implica que el primer valor va a ser ACCEL_LN_X, el segundo ACCEL_LN_Y y así sucesivamente
    for columna in dataframe.columns:

        ## Lo que hago acá es setear una bandera a True en caso que el primer elemento de dicha columna (indice 0) sea CAL
        ## Dicho de otra manera estoy "seleccionando" únicamente aquellas columnas de datos que tengan CAL
        bandera = (dataframe[columna].iloc[0] == "CAL")

        ## Entro al if si se cumplen las siguientes cosas AL MISMO TIEMPO:
        ## i) La columna tiene que tener el valor CAL, lo cual se comprueba mirando el valor de <<bandera>>
        ## ii) Debe ocurrir que el nombre de la columna debe ser igual a <<nombre1>> o a <<nombre2>>
        ##     La comparación de los nombres se hace en minúscula para normalizar los caracteres
        ## En resumen yo me estoy quedando con aquellas columnas CAL y cuyo nombre sea igual a uno de los dos que pasé como parámetro de entrada
        if (bandera) and (columna.lower() == nombre1.lower() or columna.lower() == nombre2.lower()):
                
                ## Selecciono entonces la columna correspondiente en el dataframe que me interesa
                datos_utiles[nombre1] = dataframe[columna]

    ## Retorno un objeto DataFrame de Pandas ÚNICAMENTE con la columna que me interesa
    return datos_utiles[nombre1]

###########################################################################################
#CORRECCION EJES  # TODO: Verificar la correccion de ejes.
##########################################################################################

## Función que se encarga de chequear que los ejes del segmento estén correctos
def chequear_ejes(ruta_archivo):
    '''
    Checks if the axes of the sample are correct. Logs the correction to a log file.
    Parameters
    ----------
    ruta_archivo: str
        Path to the sample

    Returns
    -------
    correccion : str
        The correction to be made: 
            -"Ninguna": no changes. No correction is made.
            -"Incorrecto": data is inconsistent. No correction is made.
            -"ViejosNegativo"
            -"ViejosPositivo"
            -"Nuevos"

    '''
    # TODO: ELIMIAR ESTO JUNTO CON LOS PACIENTES MAL ADQUIRIDOS, ESTO NO ES ROBUSTO Y SOLO VA A TRAER PROBLEMAS

    ## Asigno umbrales de manera arbitraria para las aceleraciones en los ejes X e Y respectivamente
    umbral_x = 8.3
    umbral_y = 8.1

    ## Abro el archivo csv pasado como parámetro y lo guardo en una estructura de tipo DataFrame
    ## Recuerdo que los datos de entrada ya están preprocesados de modo que tengo las columnas de Tiempo, AC_x, AC_y, AC_z, y los giros en los tres ejes también
    df = pd.read_csv(ruta_archivo)

    ## REVISAR: Entiendo que se quieren calcular los valores medios de las columnas de aceleraciones del archivo preprocesado
    ## SIN EMBARGO A MI ME DA ERROR AL PROBAR APLICAR ÉSTO YA QUE NO ME DEJA CALCULAR UNA MEDIA USANDO <<mean>> A UNA ESTRUCTURA DE TIPO DATAFRAME PANDAS
    ## SÍ ME FUNCIONA CUANDO HAGO LA CONVERSIÓN DE LA COLUMNA A UN VECTOR NUMPY Y AHÍ LE CALCULO EL VALOR MEDIO USANDO <<mean>>
    ## ESCRIBO COMO COMENTARIO LAS LÍNEAS QUE SÍ ME FUNCIONAN ACÁ

    ## En caso de que el valor medio de la columna de aceleración en el eje x
    if (mean(df["AC_x"]) > umbral_x):
        correccion = 'ViejosPositivo'

    elif (mean(df["AC_x"])<-umbral_x):
        correccion = 'ViejosNegativo'
    elif (mean(df["AC_y"])<-umbral_y):
        correccion = 'Nuevos'
    elif(mean(df["AC_y"])>umbral_y):
        correccion = 'Ninguna'
    else:
        correccion = 'Incorrecto' #TODO: deberiamos loggear esto en un archivo, y riseError

    f = open(path_log_preprocesamiento, "a")
    f.write("\nArchivo%s" % (ruta_archivo) )
    f.write('\nCorrecion de ejes:%s' % (correccion))
    f.close()

    return correccion

def ejecutar_correccion(correccion,ruta_archivo):
    '''
    Function that makes the correction. If "correcion" is not in the dictionary, returns the same file.
    
    Parameters:
    -----------
    correccion: str
        The correction to be made
    ruta_archivo: str
        Path to the file to correct. It is overriten.
    '''
    diccionario = {'ViejosPositivo':columnas_viejos_pos(ruta_archivo),
            'ViejosNegativo': columnas_viejos_neg(ruta_archivo),
            'Nuevos':columnas_nuevos(ruta_archivo)}

    return diccionario.get(correccion) 

def columnas_viejos_pos(ruta_archivo):
    '''
    Corrects the placement of the sensor with the "x" axis solidary to gravity and the same direction, instead of being the "y" axis.

    Parameters: 
    ----------
    ruta_archivos: str
        Path to the file to correct. It is overriten.
    '''
    df = pd.read_csv(ruta_archivo)
    ds = df.copy(deep=True)
    acel_x = ds.AC_x
    acel_y = ds.AC_y
    acel_z = ds.AC_z
    giro_x = ds.GY_x
    giro_y = ds.GY_y
    giro_z = ds.GY_z

    df["AC_x"]= -acel_y
    df["AC_y"]=  acel_x
    df["AC_z"]= -acel_z

    df["GY_x"]= -giro_y
    df["GY_y"]=  giro_x
    df["GY_z"]= -giro_z

    df.to_csv(ruta_archivo, index=False)

def columnas_viejos_neg(ruta_archivo):
    '''
    Corrects the placement of the sensor with the "x" axis solidary to gravity and the opposite direction, instead of being the "y" axis.
    
    Parameters: 
    ----------
    ruta_archivos: str
        Path to the file to correct. It is overriten.
    '''

    df = pd.read_csv(ruta_archivo)
    ds = df.copy(deep=True)
    acel_x = ds.AC_x
    acel_y = ds.AC_y
    acel_z = ds.AC_z
    giro_x = ds.GY_x
    giro_y = ds.GY_y
    giro_z = ds.GY_z

    df["AC_x"]=  acel_y
    df["AC_y"]= -acel_x
    df["AC_z"]= -acel_z

    df["GY_x"]=  giro_y
    df["GY_y"]= -giro_x
    df["GY_z"]= -giro_z
    df.to_csv(ruta_archivo, index=False)

def columnas_nuevos(ruta_archivo):
    '''
    Corrects the placement of the sensor rotated 180 degrees.    

    Parameters: 
    ----------
    ruta_archivos: str
        Path to the file to correct. It is overriten.
    '''

    ## Leo el csv pasado como parámetro en <<ruta_archivo>> y lo guardo en un DataFrame
    df = pd.read_csv(ruta_archivo)


    df["AC_x"] = -df["AC_x"]
    df["AC_y"] = -df["AC_y"]
    df["AC_z"] =  df["AC_z"]

    df["GY_x"]= -df["GY_x"]
    df["GY_y"]= -df["GY_y"]
    df["GY_z"]=  df["GY_z"]
    df.to_csv(ruta_archivo, index=False)
