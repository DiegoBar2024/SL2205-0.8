from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

def DeteccionPicos(señal, umbral):
    """
    Dada una señal de entrada, se encarga de realizar la detección de picos para un umbral dado como parámetro de entrada

    Parameters
    ----------
    señal: ndarray(N,)
        Señal de entrada para la cual se detectarán los picos
    umbral: float
        Valor numérico que representa el umbral a partir del cual se considerarán picos
    """

    ## Se usa la función <<find_peaks>> para poder hacer la detección de los picos de la señal de aceleración
    ## Mediante el atributo <<height>> se configura el umbral que se va a tomar para la detección de los picos
    ## <<picos>> me va a dar una lista de valores temporales (o de muestras en éste caso) en donde se encuentren los picos
    picos, _ = signal.find_peaks(señal, height = umbral)

    ## Retorno una lista que me devuelve las POSICIONES de los picos en la señal de entrada
    return picos

def GraficacionPicos(señal, picos):
    """
    Dada una señal de entrada y sus picos calculados, se realiza la gráfica de la señal marcando con una cruz los instantes de los picos

    Parameters
    ----------
    señal: ndarray(N,)
        Señal de entrada para la cual fueron detectados los picos
    picos: ndarray(M,)
        Vector que me da las posiciones de la señal en las cuales fueron detectados los picos
    """

    ## Se especifica que se quiere graficar <<señal>>
    plt.plot(señal)

    ## Se especifica que se quieren mostrar en la gráfica los picos de la señal en los instantes en que ocurren
    ## Ademas se dice que los picos se indiquen con "x"
    plt.plot(picos, señal[picos], "x")

    ## Se despliega la gráfica
    plt.show()

def SeparacionesPicos(picos):
    """
    Dado un vector de entrada que me indica las posiciones de los picos en una señal, retornar un vector cuyos elementos sean
    las diferencias de posición entre los picos sucesivos
    """

    ## Se obtiene la primera diferencia de la señal <<picos>> usando <<np.diff>>
    separaciones = np.diff(picos)

    ## Retorno el vector de separaciones
    return separaciones