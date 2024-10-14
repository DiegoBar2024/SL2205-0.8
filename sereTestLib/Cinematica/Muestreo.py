import numpy as np

def PeriodoMuestreo(dataframe):
    """
    Función que permite calcular el período de muestreo de un dataframe de entrada

    Parameters
    ----------
    dataframe : pd.DataFrame
        Éste valor representa el dataframe de entrada
    """

    ## Construyo el vector de tiempos
    time = np.array(dataframe["Time"])

    ## Retorno la diferencia entre el primer y segundo valor temporales en ms
    return (time[1] - time[0]) / 1000