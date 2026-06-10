import numpy as np
from scipy.stats import skew, kurtosis
from scipy.spatial.transform import Rotation as R

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

def extract_1d_signal_features(seg, fs):
    """
    Extrae características estadísticas, dinámicas y frecuenciales
    de una señal 1D correspondiente a un segmento de movimiento.

    Esta función está diseñada para ser reutilizable sobre cualquier
    señal cinemática (ej. velocidad angular o aceleración) y constituye
    el bloque básico de extracción de features por eje.

    Se utiliza dentro de un pipeline de análisis de movimientos segmentados
    (ej. giros), aplicándose por separado a cada eje del giroscopio y acelerómetro.

    Parameters
    ----------
    seg : np.ndarray
        Segmento de señal 1D (por ejemplo wx, wy, wz o ax, ay, az).

    fs : float
        Frecuencia de muestreo en Hz.

    Returns
    -------
    dict
        Diccionario con características del segmento:

        - mean: valor medio de la señal
        - peak: valor pico absoluto
        - rms: valor RMS (energía efectiva)
        - time_to_peak: instante relativo del pico dentro del segmento (0-1)
        - peak_mean_ratio: relación entre pico y media
        - skew: asimetría de la distribución
        - kurt: curtosis (presencia de outliers o impulsos)
        - zcr: tasa de cruces por cero (oscilación)
        - spec_entropy: entropía espectral (complejidad en frecuencia)
        - spec_cent: centroide espectral (frecuencia media ponderada)
        - dfar: dominancia espectral (inversa de flatness espectral)
        - jerk_energy: energía de la derivada (brusquedad del movimiento)
        - signal_energy: energía total de la señal
    """

    ## Obtengo el valor pico máximo correspondiente al segmento de señal de entrada
    peak = np.max(np.abs(seg))

    ## Obtengo el valor RMS correspondiente al segmento de señal de entrada
    rms = np.sqrt(np.mean(seg ** 2))

    ## Obtengo el valor medio correspondiente al segmento de señal de entrada
    mean = np.mean(seg)

    ## En caso de que el segmento se señal tenga más de una muestra
    if len(seg) > 1:

        ## Obtengo la relación del tiempo que transcurre hasta el instante donde se produce el
        ## mayor pico y el tiempo total del segmento de giro
        t_peak = np.argmax(np.abs(seg)) / len(seg)

    ## En caso que el segmento de señal tenga menos de dos muestras
    else:

        ## Asigno como cero al tiempo de pico
        t_peak = 0

    ## Obtengo la relación entre el valor pico y el valor medio del segmento de señal
    peak_mean_ratio = peak / (np.abs(mean) + 1e-8)

    ## Obtengo el skewness correspondiente al segmento de señal de entrada
    skew_v = skew(seg)

    ## Obtengo el kurtosis correspondiente al segmento de señal de entrada
    kurt_v = kurtosis(seg)

    ## Obtengo la tasa de cruces en cero correspondientes al segmento de señal de entrada
    zcr = np.sum(np.diff(np.sign(seg)) != 0) / len(seg)

    ## Obtengo la entropía espectral correspondiente al segmento de señal de entrada
    fft = np.abs(np.fft.rfft(seg))
    p = fft / (np.sum(fft) + 1e-12)
    spectral_entropy = - np.sum(p * np.log(p + 1e-12))

    ## Obtengo la 'jerk energy' (estimación de la energía de la derivada) del segmento de la señal de entrada
    jerk = np.gradient(seg) * fs
    jerk_energy = np.mean(jerk ** 2)

    ## Obtengo la energía total de la señal (sin normalizar)
    signal_energy = np.sum(seg ** 2)

    ## Obtengo el centroide espectral de la señal
    freqs = np.fft.rfftfreq(len(seg), d = 1/fs)
    spectral_centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-12)

    ## Obtengo la dominancia espectral de la señal
    geometric_mean = np.exp(np.mean(np.log(fft + 1e-12)))
    arithmetic_mean = np.mean(fft)
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-12)
    spectral_dominance = 1 / (spectral_flatness + 1e-12)

    ## Retorno un diccionario con las features extraídas del segmento de señal de entrada
    return {"mean": mean, "peak": peak, "rms": rms, "time_to_peak": t_peak, 
            "peak_mean_ratio": peak_mean_ratio, "skew": skew_v, "kurt": kurt_v, "zcr": zcr,
            "spec_entropy": spectral_entropy, "jerk_energy": jerk_energy, "signal_energy": signal_energy,
            "spec_cent": spectral_centroid, "dfar": spectral_dominance}

def extraer_features_basicas(imu_quat, segmentos, fs, id, gyro, acc):
    """
    Extrae características cinemáticas por evento de giro usando
    giroscopio (wx, wy, wz) y acelerómetro (ax, ay, az).

    Cada segmento se analiza de forma independiente, extrayendo
    el mismo conjunto de features para cada eje.

    Parameters
    ----------
    imu_quat : np.ndarray (N, 4)
        Cuaterniones estimados por EKF-AHRS.

    segmentos : list of dict
        Segmentos con:
            - start_idx
            - end_idx

    fs : float
        Frecuencia de muestreo.

    id : int
        Identificador del sujeto.

    gyro : np.ndarray (N, 3)
        Señal de velocidad angular (en sistema inercial o body consistente).

    acc : np.ndarray (N, 3)
        Señal de aceleración (en mismo sistema que gyro).

    Returns
    -------
    list of dict
        Features por giro y por eje.
    """

    ## Obtengo la estimación del conjunto de todos los ángulos de giro
    ang_rad = estimar_angulos_giro(imu_quat, segmentos)

    ## Hago la conversión de ángulos de giro de radianes a grados
    ang_deg = np.rad2deg(ang_rad)

    ## Inicializo una lista vacía en la cual voy a almacenar todas las features de los giros
    features = []

    ## Construyo un diccionario separando las señales de giroscopios y acelerómetros en cada eje
    signal_map = {"wx": gyro[:, 0], "wy": gyro[:, 1], "wz": gyro[:, 2],
                "ax": acc[:, 0], "ay": acc[:, 1], "az": acc[:, 2]}

    ## Itero para cada uno de los segmentos de giros detectados
    for i, seg in enumerate(segmentos):

        ## Obtengo el índice de la muestra donde comienza el giro
        s = seg["start_idx"]

        ## Obtengo el índice de la muestra donde termina el giro
        e = seg["end_idx"]

        ## En caso de que el giro tenga menos de dos muestras
        if e <= s + 1:

            ## Continúo con el procesamiento del siguiente giro
            continue

        ## Obtengo la duración del giro expresada en segundos
        duration = (e - s) / fs

        ## Obtengo la estimación del ángulo de rotación correspondiente al giro
        angle = ang_deg[i]

        ## Extraigo los segmentos del giro correspondientes al acelerómetro y al giroscopio
        gyro_seg = signal_map["wx"][s:e], signal_map["wy"][s:e], signal_map["wz"][s:e]
        acc_seg = signal_map["ax"][s:e], signal_map["ay"][s:e], signal_map["az"][s:e]

        ## Hago la organización de dichos segmentos de giroscopio y acelerómetro en matrices (M,3)
        gyro_seg = np.vstack(gyro_seg).T
        acc_seg = np.vstack(acc_seg).T

        ## Hago la diferenciación temporal de los segmentos de giroscopio y acelerómetro
        gyro_jerk = np.gradient(gyro_seg, axis = 0) * fs
        acc_jerk = np.gradient(acc_seg, axis = 0) * fs

        ## Obtengo la energía promedio de las diferenciaciones del segmento de giroscopio y el
        ## acelerómetro en los ejes en el plano horizontal
        gyro_horz_jerk_energy = np.mean(gyro_jerk[:, 0] ** 2 + gyro_jerk[:, 1] ** 2)
        acc_horz_jerk_energy = np.mean(acc_jerk[:, 0] ** 2 + acc_jerk[:, 1] ** 2)

        ## Inicializo un diccionario donde guardo características del giro como el ID de la persona,
        ## duración del giro en segundos y ángulo de rotación del giro en grados
        base_record = {"id": id, "start_idx": s, "end_idx": e, "duration_s": duration, "angle_deg": angle,
                "gyro_horz_jerk_energy": gyro_horz_jerk_energy, "acc_horz_jerk_energy": acc_horz_jerk_energy}

        ## Itero para cada una de las señales de los giroscopios y acelerómetros
        for name, signal in signal_map.items():

            ## Me quedo únicamente con el segmento de dicha señal de acelerómetro/giroscopio
            ## correspondiente al intervalo en el cual se produce un giro
            seg_signal = signal[s:e]

            ## Hago la extracción de features del segmento de señal de acelerómetro/giroscopio del giro
            f = extract_1d_signal_features(seg_signal, fs)

            ## Configuro un prefijo combinando el tipo de señal y el giro detectado
            f_prefixed = {f"{name}_{k}": v for k, v in f.items()}

            ## Agrego las features de la señal de acelerómetro/giroscopio al diccionario de features
            ## correspondiente al giro detectado
            base_record.update(f_prefixed)

        ## Agrego el diccionario de features del giro a la lista de features de los giros
        features.append(base_record)

    ## Retorno la lista de diccionarios donde cada diccionario contiene todas las features de un giro
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

def get_segment(signals, seg):
    """
    Extrae un segmento de señales IMU a partir de índices de inicio y fin.

    Description
    -----------
    Recorta las señales de giroscopio, acelerómetro y cuaterniones
    correspondientes a un evento (turno) definido por índices temporales.

    Parameters
    ----------
    signals : dict
        Diccionario con las señales completas del sistema IMU:
        - 'gyro': array (N, 3)
        - 'acc' : array (N, 3)
        - 'quat': array (N, 4)

    seg : dict
        Diccionario con los índices del segmento:
        - 'start_idx': índice inicial del turno
        - 'end_idx'  : índice final del turno
        - opcionalmente 'turn_id'

    Returns
    -------
    dict
        Diccionario con las señales recortadas del segmento:
        - 'gyro': array (M, 3)
        - 'acc' : array (M, 3)
        - 'quat': array (M, 4)

    Raises
    ------
    ValueError
        Si el índice final es menor o igual al inicial.
    """

    ## Obtengo los índices de muestra en los que comienza y termina el giro correspondiente
    s, e = int(seg["start_idx"]), int(seg["end_idx"])

    ## Chequeo: En caso de que los índices del segmento de giro no sean válidos
    if e <= s:

        ## Disparo una excepción indicando el error correspondiente
        raise ValueError(f"Invalid segment: start = {s}, end = {e}")

    ## Retorno un diccionario con las correspondientes segmentaciones de giroscopio, acelerómetro
    ## y cuaternión en el tramo de giro correspondiente
    return {"gyro": signals["gyro"][s:e], "acc": signals["acc"][s:e], "quat": signals["quat"][s:e]}

def add_feature_to_dataset(df, segments, feature_name, feature_fn, signals, fs):
    """
    Añade una nueva feature al dataset de giros sin recomputar el resto de variables.

    Description
    -----------
    Calcula una feature a nivel de cada segmento de giro y la incorpora como una
    nueva columna en el DataFrame existente, preservando todas las features previas.

    El cálculo se realiza de forma independiente por segmento utilizando una función
    de feature personalizada.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset de giros, donde cada fila corresponde a un evento.

    segments : list of dict
        Lista de segmentos de giro con:
        - 'start_idx'
        - 'end_idx'
        - opcionalmente 'turn_id'

    feature_name : str
        Nombre de la nueva feature a agregar como columna en el DataFrame.

    feature_fn : callable
        Función que recibe:
            - data (dict con gyro, acc, quat)
            - fs (float)
        y devuelve un valor escalar por segmento.

    signals : dict
        Señales completas del sistema IMU:
        - 'gyro': array (N, 3)
        - 'acc' : array (N, 3)
        - 'quat': array (N, 4)

    fs : float
        Frecuencia de muestreo en Hz.

    Returns
    -------
    pandas.DataFrame
        DataFrame con la nueva columna añadida.

    Raises
    ------
    ValueError
        Si el número de filas del DataFrame no coincide con el número de segmentos,
        o si se detecta desalineación en los índices de segmentos.
    """

    ## Chequeo: En caso de que el tamaño del dataframe no coincida con la cantidad de giros
    if len(df) != len(segments):

        ## Despliego el mensaje de error correspondiente
        raise ValueError("El tamaño del dataframe debe ser igual a la cantidad de giros")

    new_values = np.empty(len(df))

    ## Itero para cada uno de los segmentos de giro
    for i, seg in enumerate(segments):

        ## En caso de que haya una inconsistencia con el orden de los segmentos
        if "turn_id" in seg and seg["turn_id"] != i:

            ## Despliego la excepción con el mensajede error correspondiente
            raise ValueError("Inconsistencia en el ordenamiento de los segmentos")

        ## Hago la segmentación de los tramos de acelerómetro, giroscopio y cuaterniones en el tramo de giro
        data = get_segment(signals, seg)

        ## Asigno el valor de la nueva feature sobre los tramos de señales en el giro dado
        new_values[i] = feature_fn(data, fs)

    ## Hago una copia del dataframe original para no modificar el original
    df = df.copy()

    ## Agrego el vector conteniendo el cálculo de la nueva feature para todos los giros
    df[feature_name] = new_values

    ## Retorno el dataframe resultante de concatenarlo con los nuevos valores de la feature
    return df

def feature_fn_nueva(data, fs):
    """
    Calcula una o múltiples features a partir de un segmento de señales IMU.

    Description
    -----------
    Función genérica de extracción de características que recibe un segmento
    de señales (giroscopio, acelerómetro y cuaterniones) y devuelve una o más
    features asociadas al mismo.

    La función puede devolver:
        - un valor escalar (una sola feature)
        - un diccionario de features (múltiples features)

    Este diseño permite integrar fácilmente nuevas métricas sin modificar
    el pipeline de procesamiento.

    Parameters
    ----------
    data : dict
        Diccionario con las señales del segmento:
        - 'gyro': array (M, 3)
        - 'acc' : array (M, 3)
        - 'quat': array (M, 4)

    fs : float
        Frecuencia de muestreo en Hz.

    Returns
    -------
    float or dict
        - float: valor único de la feature
        - dict: conjunto de features con nombre como clave y valor escalar
    """

    gyro = data["gyro"]
    acc = data["acc"]

    features = {}

    # ============================================================
    # Horizontal plane signals (XY)
    # ============================================================
    gyro_h = np.sqrt(gyro[:, 0]**2 + gyro[:, 1]**2)
    acc_h  = np.sqrt(acc[:, 0]**2 + acc[:, 1]**2)

    # ============================================================
    # Jerk energy function
    # ============================================================
    def jerk_energy(x):
        jerk = np.gradient(x) * fs
        return np.mean(jerk ** 2)

    # ============================================================
    # Horizontal jerk energy features (NEW CORE IDEA)
    # ============================================================
    features["gyro_horz_jerk_energy"] = jerk_energy(gyro_h)
    features["acc_horz_jerk_energy"] = jerk_energy(acc_h)

    return features