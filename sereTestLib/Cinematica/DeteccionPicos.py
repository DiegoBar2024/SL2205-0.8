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

    Parameters
    ----------
    picos: ndarray(M,)
        Vector que me da las posiciones de la señal en las cuales fueron detectados los picos
    """

    ## Se obtiene la primera diferencia de la señal <<picos>> usando <<np.diff>>
    separaciones = np.diff(picos)

    ## Retorno el vector de separaciones
    return separaciones

def CalculoUmbral(señal, Tmin = 0.2, Tmax = 0.7, step = 0.001):
    """
    Dada una señal de entrada se hace el cálculo del umbral óptimo para la detección de picos.
    Se usa para ésto un algoritmo que logre minimizar la desviación estándar del vector de distancias entre picos
    
    Parameters
    ----------
    señal: ndarray(N,)
        Señal de entrada para la cual se calculará el umbral óptimo para la detección de picos
    Tmin: float
        Extremo inferior del rango sobre el cual se va a iterar.
    Tmax: float
        Extremo superior del rango sobre el cual se va a iterar
    step: float
        Es la longitud del paso que se va a utilizar para la iteración desde Tmin hasta Tmax
    """

    ## En base a los parámetros de entrada Tmin, Tmax y step se calcula el vector sobre el cual se realizará la iteración
    ## Se usa la función <<np.arange>> la cual abarque [Tmin, Tmax] de a pasos dados por step.
    vector_T = np.arange(start = Tmin, stop = Tmax + step, step = step)

    ## Declaro una bandera que me distinga si es el primer elemento en la iteracion
    primer_elemento = True

    ## Itero para cada uno de los valores en el vector anterior como posibles umbrales T
    for T in vector_T:

        ## Hago la detección de los picos de la señal de entrada para el umbral T actual
        picos_T = DeteccionPicos(señal = señal, umbral = T)

        ## Genero la señal que me da las separaciones entre picos en el dominio de las muestras
        sepPicos_T = SeparacionesPicos(picos = picos_T)

        ## Desviación estándar de las separaciones de picos tomando un umbral T
        std_T = np.std(sepPicos_T)

        ## En caso de que éste sea el primer elemento
        if primer_elemento:

            ## Asigno a la tupla el valor de T con su correspondiente desviación estándar
            T_desv = (T, std_T)

            ## Seteo la bandera <<primer_elemento>> en False para indicar que ya no es el primero
            primer_elemento = False
        
        ## En caso de que éste no sea el primer elemento pero la desviación estándar sea mayor a la guardada
        elif std_T < T_desv[1]:

            ## Asigno a la tupla el valor de T con su correspondiente desviación estándar
            T_desv = (T, std_T)
    
    ## Retorno la tupla con el valor de umbral óptimo y su desviación estándar
    return T_desv
