import pandas as pd

def GeneroEdadPaciente(id_paciente):
    """
    Función que dada la ID de un determinado paciente retorna el género (F/M) y la edad del paciente

    Parameters
    ----------
    id_paciente: int
        Número de identificación del paciente

    Returns
    ----------
    genero_edad: tuple
        Tupla cuyos elementos son el género (F/M) y la edad del paciente en ese orden
    """

    ## Ruta del archivo
    ruta = "C:/Yo/Tesis/sereData/sereData/Etiquetas/clasificaciones_antropometricos.csv"

    ## Lectura de datos
    datos = pd.read_csv(ruta)

    ## Filtrado de los datos
    datos_paciente = datos[datos['sampleid'] == id_paciente]

    ## Genero una tupla con el género y la edad del paciente
    genero_edad = (datos_paciente['Sexo'].iloc[0], datos_paciente['Edad'].iloc[0])

    ## Retorno la tupla correspondiente
    return genero_edad