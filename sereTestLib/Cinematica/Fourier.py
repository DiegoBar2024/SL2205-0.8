from scipy.fft import fft, ifft, fftfreq
import numpy as np
from matplotlib import pyplot as plt

def TransformadaFourier(señal, dt, plot = True):
    '''
    Función que realiza el cálculo de la FFT de una señal de entrada y su graficación.
    El eje de las frecuencias está por defecto en Hz.

    Parameters
    ----------
    señal : ndarray (N,)
        Vector de la señal x(t) a transformar
    dt : int
        Período de muestreo de la señal
    plot : bool
        En caso de que plot sea True me grafica. Si plot es False no me grafica
    '''

    ## Por medio de la función <<fft>> calculo la transformada de Fourier de la señal de entrada
    transformada = fft(señal)

    ## Por medio de la función <<fftfreq>> calculo el eje de las frecuencias
    frecuencias = fftfreq(señal.shape[0], dt)

    ## En caso que quiera graficar
    if plot:

        ## Hago la gráfica de los coeficientes de la transformada en función de la frecuencia
        ## La gráfica se hace únicamente para frecuencias positivas
        plt.plot(frecuencias[:señal.shape[0]//2], (2 / señal.shape[0]) * np.abs(transformada[0:señal.shape[0]//2]))

        ## Nomenclatura de ejes
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Magnitud")

        ## Despliego la gráfica
        plt.show()

    ## Retorno una tupla con el vector de frecuencias y de los coeficientes de la transformada
    return (frecuencias, transformada)