import matplotlib.pyplot as plt
from ahrs.filters import EKF
import numpy as np
from scipy.spatial.transform import Rotation as R

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

    ## En caso de que sólo tenga un cuaternión de orientación de entrada, lo estructuro de modo consistente
    if q.ndim == 1:
        q = q[np.newaxis, :]

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
                if abs(theta_total) > threshold:

                    ## Considero el segmento detectado como un giro y me lo guardo den la lista
                    ## junto con sus índices de comienzo y final
                    turns.append({"start_idx": start, "end_idx": end})

    ## En caso de que el registro termine cuando estoy en medio de un giro, debo procesarlo como tal
    if in_turn:

        ## Configuro el instante final del giro correspondiente al instante final del dataset
        end = len(wz)

        ## En caso que la duración del giro detectado en muestras sea mayor al umbral mínimo
        if (end - start) >= min_samples:

            ## Hago la integración de la señal de velocidad angular en la ventana correspondiente
            theta_total = np.trapz(wz[start:end], dx = dt)

            ## En caso que la variación angular sea mayor a un determinado umbral
            if abs(theta_total) > threshold:

                ## Considero el segmento detectado como un giro y me lo guardo den la lista
                ## junto con sus índices de comienzo y final
                turns.append({"start_idx": start, "end_idx": end - 1})

    ## Retorno la lista la cual contiene todos los giros con sus inicios y terminaciones
    return turns

def extraer_segmentos_giros(acel, gyro, giros):
    """
    Separa los segmentos de la señal correspondientes a cada evento de giro
    detectado.

    Esta función no elimina los giros, sino que los segmenta individualmente
    para su análisis independiente.

    Parámetros
    ----------
    acel : np.ndarray (N, 3)
        Señal del acelerómetro.

    gyro : np.ndarray (N, 3)
        Señal del giroscopio.

    giros : list of dict
        Lista de eventos de giro detectados, donde cada evento contiene:
        - 'start_idx': inicio del giro
        - 'end_idx': fin del giro

    Retorna
    -------
    segmentos : list of dict
        Lista donde cada elemento representa un giro y contiene:
        - 'acel': segmento del acelerómetro (M, 3)
        - 'gyro': segmento del giroscopio (M, 3)
        - 'start_idx': índice original de inicio
        - 'end_idx': índice original de fin

    Descripción
    -----------
    La función recorta la señal original en los intervalos definidos por
    los eventos de giro, preservando la estructura temporal interna de cada
    evento. Esto permite:

    - Analizar giros de forma individual
    - Comparar patrones entre giros
    - Extraer características por evento
    """

    ## Inicializo una lista vacía en donde voy a guardar los segmentos de giro
    segmentos = []

    ## Itero para cada uno de los eventos de giros detectados
    for g in giros:

        ## Obtengo el índice de comienzo del giro
        start = g["start_idx"]

        ## Obtengo el índice de terminación del giro
        end = g["end_idx"]

        ## Construyo un diccionario conteniendo las secuencias de acelerómetros y giroscopios en el giro
        segmento = {"acel": acel[start:end], "gyro": gyro[start:end], 
                    "start_idx": start, "end_idx": end}

        ## Almaceno el segmento correspondiente a la lista de segmentos
        segmentos.append(segmento)

    ## Retorno los segmentos de acelerómetros y giroscopios separados entre giros
    return segmentos

def quaternion_angular_velocity(q, fs):
    """
    Estima la velocidad angular a partir de una secuencia de cuaterniones
    utilizando diferencias finitas en SO(3).

    Incluye:
    - Corrección de continuidad de signo del cuaternión
    - Diferencia hacia adelante (primer punto)
    - Diferencia central (puntos interiores)
    - Diferencia hacia atrás (último punto)

    Parámetros
    ----------
    q : array Nx4 (w, x, y, z)
        Cuaterniones estimados por EKF-AHRS.
    fs : float
        Frecuencia de muestreo (Hz).

    Retorna
    -------
    omega : array Nx3
        Velocidad angular expresada en el sistema inercial (ENU/NED).
    """

    ## Defino el período de muestreo a partir de la correspondiente frecuencia de muestreo
    dt = 1 / fs

    ## Obtengo la cantidad de muestras de la señal
    n = len(q)

    ## Hago la corrección de signos de los cuaterniones para evitar discontinuidades
    q = q.copy()
    for i in range(1, n):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1

    ## Hago la conversión de los cuaterniones a formato Scipy: de wxyz a xyzw
    q_scipy = q[:, [1, 2, 3, 0]]
    rot = R.from_quat(q_scipy)

    ## Inicializo una matriz en la cual voy a almacenar las estimaciones de la velocidad angular
    omega = np.zeros((n, 3))

    ## Implemento una diferenciación hacia delante de primer orden en la primera muestra
    dq = rot[0].inv() * rot[1]
    omega[0] = dq.as_rotvec() / dt

    ## Itero para cada uno de los cuaterniones de orientación intermedios
    for i in range(1, n - 1):

        ## Implemento una diferencia central para los cuaterniones intermedios
        dq = rot[i - 1].inv() * rot[i + 1]
        omega[i] = dq.as_rotvec() / (2 * dt)

    ## Implemento una diferenciación hacia delante de primer orden en la primera muestra
    dq = rot[n - 2].inv() * rot[n - 1]
    omega[n - 1] = dq.as_rotvec() / dt

    ## Retorno la velocidad angular estimada en base a los cuaterniones de orientación
    return omega