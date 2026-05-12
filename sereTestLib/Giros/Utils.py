import pandas as pd
import matplotlib.pyplot as plt
from ahrs.filters import EKF
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

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

def estimar_angulos_giro(imu_quat, segmentos):
    """
    Estima el ángulo de giro usando únicamente el método de endpoints:
    rotación relativa entre el inicio y el fin del segmento.

    Parameters
    ----------
    imu_quat : np.ndarray (N, 4)
        Cuaterniones (w, x, y, z) estimados por EKF.
    segmentos : list of dict
        Lista de segmentos con:
        - 'start_idx'
        - 'end_idx'

    Returns
    -------
    list of float
        Ángulos de giro en radianes.
    """

    ## Hago la conversión de la secuencia de cuaterniones de entrada a secuencia de objetos Scipy Rotation
    R_all = R.from_quat(imu_quat[:, [1, 2, 3, 0]])

    ## Configuro la dirección vertical según el sistema de referencia inercial escogido en el pipeline
    z_world = np.array([0, 0, 1])

    ## Inicializo una lista vacía en la cual voy a almacenar los ángulos
    angulos = []

    ## Itero para cada uno de los segmentos de giro que tengo
    for seg in segmentos:

        ## Obtengo el índice de la muestra que indica el inicio y el final del giro
        s = seg["start_idx"]
        e = seg["end_idx"]

        ## Chequeo: Evito segmentos degenerados (menos de 2 muestras)
        if e <= s + 1:
            angulos.append(0)
            continue

        ## Obtengo la rotación relativa entre la orientación de la IMU en el instante inicial del giro
        ## y la orientación de la IMU en el instante final del giro
        R_delta = R_all[s].inv() * R_all[e]

        ## Hago la conversión de dicha orientación relativa a vector de rotación
        rotvec = R_delta.as_rotvec()

        ## Hago la rotación del versor que determina la dirección vertical en ENU/NED a la orientación
        ## de la IMU al inicio del giro definida por R_all[s]
        z_start = R_all[s].inv().apply(z_world)

        ## Obtengo el ángulo de giro como la proyección del vector de rotación a la coordenada vertical
        theta = rotvec @ z_start

        ## Agrego el ángulo de giro a la lista de ángulos de giro
        angulos.append(theta)

    ## Retorno los ángulos de giro correspondientes a todos los giros detectados
    return angulos

def extraer_features_basicas(imu_quat, segmentos, fs, id, wz_suav):
    """
    Extrae características cinemáticas por evento de giro a partir de una IMU lumbar.

    Cada giro se analiza de forma independiente utilizando:
    - velocidad angular filtrada del giroscopio (wz_suav) para métricas dinámicas
    - cuaterniones estimados por EKF-AHRS para el cálculo del ángulo total de giro

    El enfoque es completamente event-based y está diseñado para análisis poblacional
    (ej. correlación de parámetros de giro con edad), priorizando robustez y consistencia
    en lugar de reconstrucción espacial completa de la trayectoria.

    Parameters
    ----------
    imu_quat : np.ndarray (N, 4)
        Cuaterniones (w, x, y, z) estimados mediante EKF-AHRS.

    segmentos : list of dict
        Lista de eventos de giro con índices:
            - start_idx
            - end_idx

    fs : float
        Frecuencia de muestreo (Hz).

    id : int
        Identificador del sujeto.

    wz_suav : np.ndarray (N,)
        Señal de velocidad angular filtrada utilizada para detección y métricas dinámicas.

    Returns
    -------
    list of dict
        Lista de features por giro:

            - id
            - duration_s
            - angle_deg
            - mean_w_deg_s
            - peak_w_deg_s
            - rms_w_deg_s
            - time_to_peak
            - peak_mean_ratio
    """

    ## Estimo los ángulos de giro (rad) mediante método de endpoints para cada uno de los segmentos de giro
    ang_rad = estimar_angulos_giro(imu_quat, segmentos)

    ## Hago la conversión de radianes a grados
    ang_deg = np.rad2deg(ang_rad)

    ## Inicializo una lista vacía en la cual voy a almacenar las features de los giros
    features = []

    ## Itero para cada segmento de giro detectado
    for i, seg in enumerate(segmentos):

        ## Obtengo el índice de la muestra que indica el inicio y el final del giro
        s = seg["start_idx"]
        e = seg["end_idx"]

        ## Chequeo: Evito segmentos degenerados (menos de 2 muestras)
        if e <= s + 1:
            continue

        ## Obtengo el segmento de giro actual de la señal medida por el giroscopio
        seg_w = wz_suav[s:e]

        ## Obtengo el tiempo correspondiente al segmento de giro como tiempo = muestras / fs
        duration = (e - s) / fs

        ## Obtengo la estimación del ángulo de giro
        angle = ang_deg[i]

        ## Obtengo la velocidad angular media asociada al giro
        mean_w = angle / duration

        ## Obtengo el valor pico de la señal del giroscopio como max{|w|}
        peak_w = np.max(np.abs(seg_w))

        ## Obtengo el valor RMS (Root Mean Square) correspondiente a la señal del giroscopio
        rms_w = np.sqrt(np.mean(seg_w ** 2))

        ## En caso de que el segmento de giro tenga más de una muestra de largo
        if len(seg_w) > 1:

            ## Obtengo la relación del tiempo que transcurre hasta el instante donde se produce el
            ## mayor pico y el tiempo total del segmento de giro
            t_peak = np.argmax(np.abs(seg_w)) / len(seg_w)
        
        ## En caso de que el segmento de giro tenga una única muestra de largo
        else:
            
            ## No tiene sentido en este caso hablar de tiempo de pico en este caso (asigno valor 0)
            t_peak = 0

        ## Obtengo la relación entre el valor pico de la señal y la velocidad angular media de giro
        peak_mean_ratio = peak_w / (np.abs(mean_w) + 1e-8)

        ## Construyo el diccionario de las features asociadas al segmento actual de giro
        features.append({"id": id, "duration_s": duration, "angle_deg": angle, "mean_w_deg_s": mean_w,
                        "peak_w_deg_s": peak_w, "rms_w_deg_s": rms_w, "time_to_peak": t_peak,
                        "peak_mean_ratio": peak_mean_ratio})

    ## Retorno el listado de features extraídas de los giros
    return features

def asignar_grupo_edad(edades):
    """
    Asigna un grupo etario a cada individuo utilizando reglas simples basadas en edad.

    Esta función convierte un vector de edades en una codificación discreta de grupos,
    lo cual facilita el análisis estadístico posterior sin depender de estructuras tipo DataFrame.

    Codificación utilizada:
        - 0 : edad < 60 años
        - 1 : 60 ≤ edad ≤ 75 años
        - 2 : edad > 75 años

    Parámetros
    ----------
    edades : array-like
        Vector de edades de los sujetos. Puede ser lista o array de NumPy.

    Retorna
    -------
    np.ndarray
        Vector de enteros con la misma longitud que `edades`, donde cada valor
        representa el grupo etario asignado a cada individuo.

    """

    ## Convierto el vector de edades a un vector de numpy
    edades = np.asarray(edades)

    ## Inicializo todos los individuos en el grupo 0 (<60 años)
    grupos = np.zeros_like(edades, dtype = int)

    ## Asigno grupo intermedio (60–75 años)
    grupos[(edades >= 60) & (edades <= 75)] = 1

    ## Asigno grupo mayor a 75 años
    grupos[edades > 75] = 2

    ## Retorno el vector de grupos etarios
    return grupos

def agrupar_por_paciente(array_features):
    """
    Agrega features de giros a nivel de paciente.

    Parameters
    ----------
    array_features : pandas.DataFrame
        DataFrame con columnas:
        - id (sujeto)
        - features de cada giro

    Returns
    -------
    list of dict
        Features agregadas por paciente
    """

    ## Obtengo la lista de IDs correspondientes a todos los giros detectados
    ids = array_features["id"].to_numpy()

    ## Obtengo un numpy array conteniendo el resto de las features correspondientes a todos los giros
    ## Las dimensiones de la matriz X van a ser (m,n) donde:
    ## m: Cantidad total de giros detectados para todas las personas
    ## n: Cantidad total de features extraídas por cada giro
    X = array_features.drop(columns=["id"]).to_numpy()

    ## Obtengo una lista conteniendo todos los IDs (sin repetir) de aquellos pacientes asociados a los giros
    pacientes = np.unique(ids)

    ## Inicializo una lista vacía en la cual voy a almacenar una sumarización estadística de todas
    ## las features de los giros pero ahora asociándolo a la persona
    features_por_paciente = []

    ## Itero para cada uno de los IDs de los pacientes para los cuales tengo giros
    for pid in pacientes:

        ## Localizo únicamente aquellos giros y sus correspondientes features asociadas para
        ## la persona actual que estoy analizando
        mask = ids == pid

        ## Análogo a antes, X va a ser una matriz de dimensiones (d,n) donde:
        ## d: Cantidad total de giros detectados para la persona actual que está procesando
        ## n: Cantidad total de features extraídas por cada giro
        data = X[mask]

        ## Chequeo: En caso de que la persona actual no tenga datos de giros
        if len(data) == 0:

            ## Continúo con la ejecución y paso a procesar los datos de giros de la próxima persona
            continue

        ## Construyo un diccionario de 'features' en las cuales almaceno la ID correspondiente a
        ## la persona que estoy procesando, y la cantidad de giros que realizó en el registro de marcha
        features = {"id": pid, "n_turns": len(data)}

        ## Itero para cada una de las features detectadas del giro
        for i in range(data.shape[1]):

            ## Obtengo un vector con los valores estadísticos del i-ésimo feature correspondientes
            ## a todos los giros detectados para la persona actual que estoy procesando
            col = data[:, i]

            ## Lo que hago es hacer un análisis estadístico correspondinete a los valores de la feature
            ## Hago el cálculo del valor medio asociado a las medidas del i-ésimo feature y lo guardo
            features[f"f{i}_mean"] = np.mean(col)

            ## Hago el cálculo de la desviación estándar asociada las medidas del i-ésimo feature y la guardo
            features[f"f{i}_std"] = np.std(col)

            ## Hago el cálculo de la mediana asociada las medidas del i-ésimo feature y la guardo
            features[f"f{i}_median"] = np.median(col)

            ## Hago el cálculo del valor máximo de las medidas del i-ésimo feature y lo guardo
            features[f"f{i}_max"] = np.max(col)

            ## Hago el cálculo del coeficiente de variabilidad asociada a las medidas del feature i
            features[f"f{i}_cv"] = features[f"f{i}_std"] / (features[f"f{i}_mean"] + 1e-8)

            ## El cálculo de los percentiles contribuye en la caracterización de la distribución
            ## a la que corresponden las secuencias de parámetros de los giros
            ## Hago el cálculo del 25th percentil asociado a las medidas del feature i
            features[f"f{i}_p25"] = np.percentile(col, 25)

            ## Hago el cálculo del 75th percentil asociado a las medidas del feature i
            features[f"f{i}_p75"] = np.percentile(col, 75)

            ## Hago el cálculo del IQR (índice intercuartil) de las medidas de la feature i
            features[f"f{i}_iqr"] = features[f"f{i}_p75"] - features[f"f{i}_p25"]

        ## Agrego el diccionario de features especificando la ID del paciente y conteniendo una
        ## sumarización estádistica de todas las features extraídas de los giros
        features_por_paciente.append(features)

    ## Retorno la lista con las sumarizaciones de los estadísticos de los giros para todas las personas
    return features_por_paciente

def plot_feature_distributions_by_age_group(df, feature_cols, age_col, output_dir, feature_names = None):
    """
    Genera y guarda boxplots de features separadas por grupos de edad.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset con features + columna de grupo etario.

    feature_cols : list of str
        Lista de columnas de features a analizar.

    age_col : str
        Nombre de la columna con grupos de edad.

    output_dir : str
        Carpeta donde se guardarán los gráficos.

    feature_names : dict
        Diccionario con los nombres de las features a graficar con sus nombres de base identificadores.
    """

    ## Generar la carpeta de destino en caso de que esta no esté creada
    os.makedirs(output_dir, exist_ok = True)

    ## Defino las etiquetas correspondientes a los grupos etarios
    group_labels = ["<60", "60 - 75", ">75"]

    ## Creo una lista con las etiquetas de los grupos para indexar en el dataframe
    groups = [0, 1, 2]

    ## Itero para cada una de las columnas de features que voy a analizar
    for feature in feature_cols:

        ## Hago la extracción de los datos correspondientes a dicha feature por grupo
        data = [df[df[age_col] == g][feature].dropna().values for g in groups]

        ## En caso de que yo pase un diccionario con los nombres de las features como entrada
        if feature_names is not None:
            
            ## Obtengo el nombre de base identificador para cada una de las features
            base_name = feature.split("_")[0]

            ## Armo el nombre de la feature correctamente
            nice_name = feature_names.get(base_name, feature)
        
        ## En caso de que no pase como entrada ningún diccionario con nombres de features
        else:

            ## Configuro el nombre predeterminado de la feature
            nice_name = feature

        ## Genero la figura del boxplot pasando como parámetros los datos del feature y las etiquetas
        plt.figure(figsize = (6, 4))
        plt.boxplot(data, tick_labels = group_labels)

        ## Configuro la estética del boxplot
        plt.title(f"{nice_name} vs Grupo Etario")
        plt.ylabel(feature)
        plt.xlabel("Grupo Etario")
        plt.grid(True, axis = "y")

        ## Guardo el boxplot correspondiente en la ruta especificada de salida
        save_path = os.path.join(output_dir, f"{feature}_grupos_etarios.png")
        plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
        plt.close()