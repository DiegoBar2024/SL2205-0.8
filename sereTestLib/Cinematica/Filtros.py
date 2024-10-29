import pandas as pd
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from Fourier import TransformadaFourier
from numpy import polyfit, arange

## Tengo como entrada un objeto DataFrame y un tamaño de ventana
## Como salida retorno un DataFrame el cual sea la versión filtrada de la entrada usando el filtro de mediana con el tamaño de ventana que pasé como parámetro
def FiltroMediana(data_frame, window):

    ## Creo en primer lugar un DataFrame vacío en principio sin ninguna columna
    dataframe_filtrado = pd.DataFrame()

    ## Itero columna por columna para el DataFrame de entrada
    for elem in data_frame.columns:
        
        ## Se aplica un filtro de mediana al vector de datos de entrada
        ## Éste filtro lo que hace es primero hacer un padding de ceros al vector de entrada
        ## Luego toma una ventana (en éste caso unidimensional) de tamaño <<window>> y la va deslizando por todo el vector
        ## Tomando el índice de salida como el que está parado, calcula la mediana para ese vector en esa ventana
        columna_filtrada = signal.medfilt(data_frame[elem], window)

        ## Creo un nuevo DataFrame el cual contenga la columna filtrada
        columna_filtrada = pd.DataFrame(columna_filtrada, columns = [elem])

        ## Concateno la columna que filtré con el DataFrame de las columnas que ya fui filtrando
        dataframe_filtrado = pd.concat((dataframe_filtrado, columna_filtrada), axis = 1)

    return dataframe_filtrado