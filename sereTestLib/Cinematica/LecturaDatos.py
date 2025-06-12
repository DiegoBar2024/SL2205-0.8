## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import pandas as pd
import numpy as np
from Muestreo import *

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

def LecturaDatos(id_persona = None, lectura_datos_propios = False, ruta = None):

    ## En caso de que esté leyendo datos propios, le doy el procesamiento correspondiente
    if lectura_datos_propios:

        ## --- Sistema de coordenadas usados en las medidas por los estudiantes ----------
        ##          EJE X: MEDIOLATERAL CON EJE POSITIVO A LA IZQUIERDA
        ##          EJE Y: VERTICAL CON EJE POSITIVO HACIA ARRIBA
        ##          EJE Z: ANTEROPOSTERIOR CON EJE POSITIVO HACIA DELANTE
        ## --- Los tres ejes de coordenadas deben formar una terna directa de vectores ---
        ## -------------------------------------------------------------------------------

        ## SENSOR 952D
        ## Pruebas estandarizadas de 50cm por paso
        # ruta = "C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/Pruebas SabriRodri-20250122T214746Z-001/2024-07-30_15.40.33_505_PC_Session5_Rodrigo_1_952D/505_Session5_Shimmer_952D_Calibrated_PC_Rodrigo_1_952D.txt"
        # ruta = "C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/Pruebas SabriRodri-20250122T214746Z-001/2024-07-30_15.40.33_505_PC_Session3_Sabrina_1_952D/505_Session3_Shimmer_952D_Calibrated_PC_Sabrina_1_952D.txt"

        ## Pruebas sin estandarizar (marcha libre)
        # ruta = "C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/Pruebas SabriRodri-20250122T214746Z-001/2024-07-30_15.40.33_505_PC_Session4_Sabrina_2_952D/505_Session4_Shimmer_952D_Calibrated_PC_Sabrina 2_952D.txt"
        # ruta = "C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/Pruebas SabriRodri-20250122T214746Z-001/2024-07-30_15.40.33_505_PC_Session6_Rodrigo_2_952D/505_Session6_Shimmer_952D_Calibrated_PC.txt"

        ## Sesiones con diferentes actividades
        # ruta = "C:/Yo/Tesis/sereData/sereData/Registros/Actividades_Rodrigo.txt"
        # ruta = "C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/Pruebas SabriRodri-20250122T214746Z-001/Actividades_Sabrina.txt"

        ## Abro el fichero correspondiente
        fichero = open(ruta, "r")

        ## Hago la lectura de todas las lineas correspondientes al fichero
        lineas = fichero.readlines()

        ## Creo un array vacío en donde voy a guardar los datos
        data = []

        ## Itero para todas aquellas lineas que tengan información útil
        for linea in lineas[3:]:

            ## Hago la traducción de la línea de datos a una lista de numeros flotantes, segmentando la línea por tabulación
            lista_datos = list(map(float,linea.split("\t")[:-1]))

            ## Agrego la lista de datos como renglón de la matriz de datos
            data.append(lista_datos)

        ## Hago una lista con todos los headers de los datos tomados
        headers = lineas[1].split("\t")[:-1]

        ## Hago el pasaje de los datos en forma de matriz a forma de dataframe
        data = pd.DataFrame(data, columns = headers)

        ## Creo una lista con las columnas deseadas
        columnas_deseadas = ['Time', 'AC_x', 'AC_y', 'AC_z', 'GY_x', 'GY_y', 'GY_z']

        ## Creo un diccionario con los nombres originales de las columnas y sus nombres nuevos
        nombres_columnas = {'Timestamp': 'Time', 'Accel_LN_X_CAL' : 'AC_x', 'Accel_LN_Y_CAL' : 'AC_y', 'Accel_LN_Z_CAL' : 'AC_z'
                            ,'Gyro_X_CAL' : 'GY_x', 'Gyro_Y_CAL' : 'GY_y', 'Gyro_Z_CAL' : 'GY_z'}

        ## Itero para cada una de las columnas del dataframe
        for columna in data.columns:

            ## Itero para cada uno de los nombres posibles
            for nombre in nombres_columnas.keys():

                ## En caso de que un nombre esté en la columna
                if nombre in columna:

                    ## Renombro la columna
                    data = data.rename(columns = {columna : nombres_columnas[nombre]})

        ## Selecciono las columnas deseadas
        data = data[columnas_deseadas]

    ## En caso de que esté leyendo datos de la empresa, le doy el procesamiento correspondiente
    else:

        ## --- Sistema de coordenadas usados en las medidas por los estudiantes ----------
        ##          EJE X: MEDIOLATERAL CON EJE POSITIVO HACIA LA DERECHA
        ##          EJE Y: VERTICAL CON EJE POSITIVO HACIA ARRIBA
        ##          EJE Z: ANTEROPOSTERIOR CON EJE POSITIVO HACIA ATRÁS
        ## --- Los tres ejes de coordenadas deben formar una terna directa de vectores ---
        ## -------------------------------------------------------------------------------

        ## Nombre del paciente
        nombre_persona = "Diego Barboza"

        ## Fecha de nacimiento del paciente "DD/MM/YYYY"
        nacimiento_persona = "02/04/2002"

        ## Ruta del archivo
        ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S{}/3S{}.csv".format(id_persona, id_persona)

        ## Selecciono las columnas deseadas
        data = pd.read_csv(ruta)

    ## ----------------------------------------- PREPROCESAMIENTO ------------------------------------------

    ## Armamos una matriz donde las columnas sean las aceleraciones
    acel = np.array([np.array(data['AC_x']), np.array(data['AC_y']), np.array(data['AC_z'])]).transpose()

    ## Armamos una matriz donde las columnas sean los valores de los giros
    gyro = np.array([np.array(data['GY_x']), np.array(data['GY_y']), np.array(data['GY_z'])]).transpose()

    ## Separo el vector de tiempos del dataframe
    tiempo = np.array(data['Time'])

    ## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
    tiempo = (tiempo - tiempo[0]) / 1000

    ## Cantidad de muestras de la señal
    cant_muestras = len(tiempo)

    ## En caso de que no haya un período de muestreo bien definido debido al vector de tiempos de la entrada
    if all([x == True for x in np.isnan(tiempo)]):

        ## Asigno arbitrariamente una frecuencia de muestreo de 200Hz es decir período de muestreo de 0.005s
        periodoMuestreo = 0.005

    ## En caso de que el vector de tiempos contenga elementos numéricos
    else:

        ## Calculo el período de muestreo en base al vector correspondiente
        periodoMuestreo = PeriodoMuestreo(data)
    
    ## Retorno los resultados al realizar la lectura correspondiente
    return data, acel, gyro, cant_muestras, periodoMuestreo, tiempo