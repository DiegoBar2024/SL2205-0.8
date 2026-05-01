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

def moving_average(x, fs = 200, window_sec = 1.0):
    """
    Aplica una media móvil a una señal unidimensional.

    Este filtro suaviza la señal eliminando componentes de alta frecuencia
    y conservando tendencias de baja frecuencia.

    Es útil para señales como velocidades angulares, aceleraciones u otras
    señales biomédicas o de sensores inerciales.

    Parámetros:
        x : np.ndarray
            Señal de entrada (1D).
        fs : int
            Frecuencia de muestreo en Hz.
        window_sec : float
            Tamaño de la ventana de la media móvil en segundos.

    Retorna:
        x_suavizada : np.ndarray
            Señal filtrada mediante media móvil.
    """

    ## Convierto la duración de la ventana a número de muestras
    window_size = int(fs * window_sec)

    ## Aseguro ventana impar para mantener simetría temporal
    if window_size % 2 == 0:
        window_size += 1

    ## Creo kernel de media móvil (filtro FIR simple)
    kernel = np.ones(window_size) / window_size

    ## Aplico convolución centrada
    x_suavizada = np.convolve(x, kernel, mode = 'same')

    ## Retorno la señal suavizada luego de aplicar el filtro de promedios móviles
    return x_suavizada

def detect_turns_windowed(wz, fs = 200, window_sec = 0.5, threshold = 30 * np.pi / 180, min_duration = 0.5):
    """
    Detecta giros usando integración por ventanas de la velocidad angular.

    El algoritmo:
        1. Calcula Δθ en ventanas deslizantes
        2. Usa Δθ para detectar eventos de giro

    Parámetros:
        wz : np.ndarray
            Velocidad angular en eje z (ENU).
        fs : int
            Frecuencia de muestreo.
        window_sec : float
            Tamaño de la ventana en segundos.
        threshold : float
            Umbral de detección en radianes.
        min_duration : float
            Duración mínima del giro en segundos.

    Retorna:
        turns : list of dict
            Eventos detectados con inicio y fin
    """

    ## Defino el período de muestreo a partir de la correspondiente frecuencia de muestreo
    dt = 1 / fs

    ## Obtengo la cantidad de muestras que tengo por ventana
    w = int(fs * window_sec)

    ## Inicializo un vector de ceros donde voy a guardar la variación de ángulo
    delta_theta = np.zeros(len(wz))

    ## Inicializo una lista vacía en la que voy a guardar los giros detectados
    turns = []

    ## Inicializo una variable booleana que me indica si estoy actualmente en medio de un giro
    in_turn = False

    ## Inicializo una variable booleana que me indique el comienzo de un giro
    start = 0

    ## Configuro una cantidad mínima de muestras de duración a partir del cual considero que detecté un giro
    min_samples = int(min_duration * fs)

    ## Itero para cada una de las ventanas que tengo
    for i in range(w, len(wz)):

        ## Recorto la ventana de velocidad angular en el eje vertical correspondiente
        segment = wz[i - w: i]

        ## Obtengo el ángulo total de rotación en dicha ventana
        delta_theta[i] = np.trapz(segment, dx = dt)

        ## En caso de que haya detectado el inicio de un giro
        if not in_turn and abs(delta_theta[i]) > threshold:

            ## Indico a <<in_turn>> que estoy en un giro y me guardo el índice de la muestra de giro de inicio
            in_turn = True
            start = i - w

        ## En caso de que se haya detectado la terminación de un giro
        elif in_turn and abs(delta_theta[i]) < 0.5 * threshold:

            ## Indico a <<in_turn>> que finalizó un giro y me guardo el índice de la muestra de giro de fin
            end = i
            in_turn = False

            ## En caso que la duración del giro detectado en muestras sea mayor al umbral mínimo
            if (end - start) >= min_samples:

                ## Calculo la integral de la velocidad angular para todo el giro (no solo para el segmento)
                theta_total = np.trapz(wz[start:end], dx = dt)

                ## En caso que la variación angular sea mayor a un determinado umbral
                if abs(theta_total) > 30 * np.pi / 180:

                    ## Considero el segmento detectado como un giro y me lo guardo den la lista
                    ## junto con sus índices de comienzo y final
                    turns.append({"start_idx": start, "end_idx": end})

    ## Retorno la lista la cual contiene todos los giros con sus inicios y terminaciones
    return turns

def plot_signal_with_events(x, events, fs = 200, title = "Señal y los eventos de giro"):
    """
    Grafica una señal continua y resalta los segmentos correspondientes a eventos.

    La señal completa se representa en color azul, mientras que los intervalos
    definidos como eventos se superponen en color rojo. Esto permite identificar
    visualmente cuándo ocurren los eventos sin perder la continuidad de la señal.

    Parámetros
    ----------
    x : np.ndarray
        Señal unidimensional a visualizar.
        Shape: (N,)

    events : list of dict
        Lista de eventos detectados. Cada evento debe contener:
            - 'start_idx': índice de inicio
            - 'end_idx': índice de fin

    fs : float, opcional (default=1)
        Frecuencia de muestreo en Hz. Se utiliza para construir el eje temporal.

    title : str, opcional (default="Signal with events")
        Título del gráfico.

    Descripción
    -----------
    La función:
        1. Construye un eje temporal a partir de la frecuencia de muestreo.
        2. Genera una máscara booleana para identificar los segmentos de eventos.
        3. Grafica la señal completa en azul.
        4. Superpone los segmentos correspondientes a eventos en rojo utilizando NaN
        para evitar la conexión entre tramos no contiguos.

    Notas
    -----
    - El uso de valores NaN permite mantener la continuidad visual de la señal
    mientras se resaltan únicamente los segmentos de interés.
    - Esta función es útil para visualizar eventos detectados en señales de sensores,
    como giros en datos de IMU.

    Retorna
    -------
    None
        La función no retorna valores. Muestra el gráfico en pantalla.
    """

    ## Obtengo la cantidad total de muestras de la señal
    n = len(x)

    ## Obtengo el vector de tiempos correspondiente
    t = np.arange(n) / fs

    ## Inicializo la máscara en la cual voy a guardar los eventos de giro
    event_mask = np.zeros(n, dtype = bool)

    ## Itero para cada uno de los eventos de giro detectados
    for e in events:

        ## Localizo aquellos intervalos de la señal en las cuales se producen los giros en base
        ## a los indices de inicio y terminación calculados antes
        event_mask[e["start_idx"]:e["end_idx"]] = True

    ## Configuro el tamaño de la gráfica de la señal y los eventos de giro
    plt.figure(figsize = (12, 4))

    ## Primero grafico toda la señal completa en azul
    plt.plot(t, x, color = 'blue', linewidth = 1, label = "No giros")

    ## Me quedo únicamente con aquella parte de la señal en la cual se produzcan eventos de giro
    x_event = np.copy(x)
    x_event[~event_mask] = np.nan

    ## Grafico ahora los segmentos en donde se producen giros de manera superpuesta a la anterior gráfica
    plt.plot(t, x_event, color = 'red', linewidth = 2, label = "Giros")

    ## Configuro los ejes y parámetros adicionales del gráfico
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()