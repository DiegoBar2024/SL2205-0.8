## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from Optimizacion import optimo_m1, optimo_m2
import pandas as pd
import os

## ----------------------------------- REGISTRO EN ARCHIVO DE TEXTO ------------------------------------

## Especifico la ruta del archivo de optimizacion
ruta_optim = "C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/Archivo_optimización.csv"

## Compruebo que el archivo de optimización exista
if os.path.isfile(ruta_optim):

    ## Leo el dataframe ya existente en el archivo .csv donde están los parámetros de optimización
    dataframe = pd.read_csv(ruta_optim)

    ## Modifico dicho dataframe agregando otra entrada con los nuevos datos
    # dataframe = pd.concat([dataframe, pd.DataFrame([{'ID' : 1, 'Nombre' : 'Rodrigo', 'Parametro_M1' : optimo_m1, 'Parametro_M2' : optimo_m2}])], ignore_index = True)
    dataframe = pd.concat([dataframe, pd.DataFrame([{'ID' : 2, 'Nombre' : 'Sabrina', 'Parametro_M1' : optimo_m1, 'Parametro_M2' : optimo_m2}])], ignore_index = True)

    ## Guardo el dataframe en el .csv sustituyendo el archivo existente
    dataframe.to_csv(ruta_optim)

## En otro caso lo creo y termino la ejecución del programa
else:

    ## Creo el dataframe vacío con sus columnas
    dataframe = pd.DataFrame(columns = ['ID', 'Nombre', 'Parametro_M1', 'Parametro_M2'])

    ## Cargo el dataframe al .csv
    dataframe.to_csv(ruta_optim)