import pandas as pd
import matplotlib.pyplot as plt
from ahrs.filters import EKF

def graficar_histograma_edades(df, columna_edad = 'Edad', mostrar_porcentaje = False):
    """
    Genera un gráfico de barras con la distribución de edades
    usando intervalos personalizados.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene los datos.
        
    columna_edad : str, opcional (default='Edad')
        Nombre de la columna que contiene las edades.
        
    mostrar_porcentaje : bool, opcional (default=False)
        Si es True, el eje Y mostrará porcentajes en lugar de frecuencias absolutas.

    Descripción
    -----------
    La función agrupa las edades en los siguientes intervalos:
        - 0-45
        - 45-60
        - 60-75
        - 75+

    A continuación, calcula cuántos individuos hay en cada grupo
    y genera un gráfico de barras.
    """

    ## Me aseguro que la columna de Edad sea numérica
    df[columna_edad] = pd.to_numeric(df[columna_edad], errors = 'coerce')

    ## Elimino valores faltantes (NaN) del dataframe
    edades = df[columna_edad].dropna()

    ## Defino los intervalos (bins) y las etiquetas
    bins = [0, 45, 60, 75, float('inf')]
    etiquetas = ['0-45', '45-60', '60-75', '75+']

    ## Creo la variable categórica de grupos de edad
    grupos = pd.cut(edades, bins = bins, labels = etiquetas, right = False)

    ## Cuento la cantidad de individuos por grupo
    conteos = grupos.value_counts().sort_index()

    ## En caso de solicitar el porcentaje correspondiente, lo presento
    if mostrar_porcentaje:
        conteos = conteos / conteos.sum() * 100

    ## Creo el gráfico correspondiente
    plt.figure(figsize = (7, 5))
    plt.bar(conteos.index.astype(str), conteos.values, edgecolor = 'black')

    ## Configuro el título y las etiquetas de los ejes
    plt.title('Distribución de edades por grupo', fontsize = 14)
    plt.xlabel('Grupo de edad')
    plt.ylabel('Porcentaje (%)' if mostrar_porcentaje else 'Frecuencia')

    ## Agrego la cuadrícula al gráfico para poder facilitar la interpretación
    plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)

    ## Agrego las etiquetas encima de cada barra
    for i, valor in enumerate(conteos.values):
        texto = f"{valor:.1f}%" if mostrar_porcentaje else str(int(valor))
        plt.text(i, valor, texto, ha='center', va='bottom')

    ## Despliego la gráfica
    plt.tight_layout()
    plt.show()

def estimar_orientacion_ekf(acc, gyro, fs, frame = 'ENU'):
    """
    Estima la orientación usando EKF (AHRS) con opción de sistema de referencia.

    Parámetros
    ----------
    acc : array Nx3
        Aceleración (m/s^2)
    gyro : array Nx3
        Velocidad angular (rad/s)
    fs : float
        Frecuencia de muestreo (Hz)
    frame : str, opcional ('ENU' o 'NED')
        Sistema de referencia inercial:
        - 'ENU' → Z hacia arriba
        - 'NED' → Z hacia abajo

    Retorna
    -------
    Q : array Nx4
        Cuaterniones (orientación en cada instante)
    """

    ## En caso de que el sistema inercial ingresado no sea 'ENU' o 'NED', arrojo error
    if frame not in ['ENU', 'NED']:
        raise ValueError("frame debe ser 'ENU' o 'NED'")

    ## Inicializo un objeto EKF AHRS pasando el sistema inercial de referencia y la frecuencia de muestreo
    ekf = EKF(gyr = gyro, acc = acc, frequency = fs, frame = frame)

    ## Retorno los cuaterniones de orientación correspondientes de la estimación
    return ekf.Q

import numpy as np
from scipy.spatial.transform import Rotation as R

def rotate_body_to_world(v_body, q):
    """
    Rota vectores desde el frame del IMU (Body) al frame inercial (World: ENU/NED).

    Parameters
    ----------
    v_body : array-like
        Vector(es) en el frame del IMU.
        Shape: (3,) o (N, 3)
    q : array-like
        Cuaterniones de orientación en formato [w, x, y, z].
        Representan la rotación Body → World.
        Shape: (4,) o (N, 4)

    Returns
    -------
    np.ndarray
        Vector(es) rotados en el frame World.
        Shape: (3,) o (N, 3)
    """

    ## Compruebo que las entradas estén expresada como arreglos bidimensionales numpy
    v_body = np.asarray(v_body)
    q = np.asarray(q)

    ## Hago la conversión de cuaterniones de formato wxyz a formato Scipy xyzw
    q_scipy = q[:, [1, 2, 3, 0]]

    ## Creo un objeto de rotación asociado a la secuencia de cuaterniones
    r = R.from_quat(q_scipy)

    ## Hago la rotación del sistema de la IMU al sistema de referencia inercial
    v_world = r.apply(v_body)

    ## Retorno la secuencia de vectores expresadas en el sistema de referencia inercial
    return v_world