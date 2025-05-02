## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from matplotlib import pyplot as plt
import numpy as np

## ------------------------------------- SEGMENTACIÓN DE LA MARCHA -------------------------------------

def Segmentacion(features, periodoMuestreo, acc_VT, plot = False):

    ## ----------------------------------------- CONJUNTO DE PASOS -----------------------------------------

    ## Obtengo un vector numpy con los contactos iniciales (expresadas en milisegundos de sesión)
    features_ICs = np.array(features['IC'])

    ## Otengo un dataframe con los contactos terminales (expresadas en milisegundos de sesión)
    features_FCs = np.array(features['FC'])

    ## Obtengo un vector con los contactos iniciales expresados en términos de muestras
    ICs_muestras = features_ICs / (1000 * periodoMuestreo)

    ## Obtengo un vector con los contactos terminales expresados en términos de muestras
    FCs_muestras = features_FCs / (1000 * periodoMuestreo)

    ## Creo una lista donde voy a almacenar todos los pasos como sus contactos iniciales y terminales
    pasos = []

    ## Itero para cada uno de los contactos iniciales 
    for i in range (1, len(ICs_muestras) - 1):
        
        ## En caso de que el contacto terminal se encuentre entre los dos contactos iniciales correspondientes
        if ICs_muestras[i] <= FCs_muestras[i - 1] <= ICs_muestras[i + 1]:

            ## Me guardo el paso correspondiente en un diccionario. En caso contrario ocurrió una falsa detección
            pasos.append({'IC': (int(ICs_muestras[i]), int(ICs_muestras[i + 1])),'TC': int(FCs_muestras[i - 1])})

    ## ------------------------------------------- GRAFICACIÓN ---------------------------------------------

    ## En caso de que quiera graficar los datos
    if plot:

        ## Graficación de eventos en la marcha con la aceleración vertical
        plt.plot(acc_VT, label = 'Aceleración Vertical')
        plt.plot(ICs_muestras.astype(int), acc_VT[ICs_muestras.astype(int)], "x", label = 'Contactos Iniciales')
        plt.plot(FCs_muestras.astype(int), acc_VT[FCs_muestras.astype(int)], "o", label = 'Contactos Terminales')
        plt.legend()
        plt.show()

    ## Retorno los pasos segmentados
    return pasos