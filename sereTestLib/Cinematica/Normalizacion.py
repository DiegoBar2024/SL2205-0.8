import numpy as np

def Normalizacion(señal):
    """
    Función que toma una señal de entrada y la normaliza en amplitud y offset

    Parameters
    ----------
    señal: ndarray(N,)
        Señal de entrada a normalizar
    """

    ## Hago la traslación de la señal al eje horizontal restándole el valor medio
    desplazada = señal - np.mean(señal)

    ## Hago ahora la normalización en amplitud de la señal
    ## És decir, obligo a que ésta señal cumpla |x(t)|<= 1 para todo valor de t
    ## Para hacer ésto divido cada elemento del vector por el máximo elemento en valor absoluto que tenga
    normalizada = desplazada / np.max(np.abs(desplazada))

    ## Retorno el valor de la señal normalizada
    return normalizada