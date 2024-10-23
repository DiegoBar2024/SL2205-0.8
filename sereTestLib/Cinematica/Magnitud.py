import numpy as np

def Magnitud(datos):
    """
    Función que calcula la magnitud de un array de dimensión (N, M) de entrada
    N sería la cantidad de muestras que se tienen. M sería la dimensión del vector para el que se está calculando la magnitud.

    Parameters
    ----------
    datos : ndarray(N,M)
        Matriz de datos de entrada
    """

    ## Defino un vector de ceros cuya dimensión sea igual a la cantidad de muestras que se tienen
    magnitud = np.zeros(datos[:,0].shape[0])

    ## Itero para cada uno de los índices en el vector de magnitud
    for i in range (datos[:,0].shape[0]):

        ## Asigno el i-ésimo valor del vector de magnitud como la magntitud de la i-ésima muestra M-dimensional
        magnitud[i] =  np.linalg.norm(datos[i,:])
    
    return magnitud