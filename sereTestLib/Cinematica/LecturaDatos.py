## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import pandas as pd

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Especifico la ruta del fichero .txt a abrir
ruta = "C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/DefaultTrial_Session2_Shimmer_B5B6_Calibrated_PC.txt"

## Especifico la ruta en donde voy a guardar el .csv correspondiente
ruta_csv = "C:/Yo/Tesis/Tesis/Excel_leidos/Prueba.csv"

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

## Escribo el dataframe en un .csv
data.to_csv(ruta_csv, index = False)