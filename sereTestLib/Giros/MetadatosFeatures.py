## Asocio el identificador de las features con su respectivo nombre en español
BASE_FEATURES = {"mean": "Media", "peak": "Pico", "rms": "RMS", 
        "time_to_peak": "Tiempo hasta el pico (fracción)", "peak_mean_ratio": "Relación pico/media",
        "skew": "Asimetría", "kurt": "Curtosis", "zcr": "Cruces por cero", 
        "spec_entropy": "Entropía espectral", "jerk_energy": "Energía de jerk"}

## Asocio el identificador de las features globales (o sea, las que no se diferencian por eje) 
## con su respectivo nombre en español
GLOBAL_FEATURES = {"angle_deg": "Ángulo de giro (grados)", "duration_s": "Duración del giro (s)"}

## Defino una lista con los nombres de todos los ejes de las señales
## Tenemos ahora los tres ejes del giroscopio y tres ejes del acelerómetro
AXES = ["wx", "wy", "wz", "ax", "ay", "az"]

def build_feature_schema():
    """
    Construye el esquema completo de features del pipeline de giros.

    Incluye:
    - Features dependientes de eje (wx, wy, wz, ax, ay, az)
    - Features globales por evento de giro (angle, duration, etc.)

    Returns
    -------
    dict
        Diccionario con metadatos de todas las features.
    """

    ## Inicializo un diccionario vacío en el cual voy a almacenar la información de las features
    FEATURES_INFO = {}

    ## Itero para cada uno de los ejes de las señales que tengo
    for axis in AXES:

        ## Itero para cada una de las features que tengo definidas
        for feat, label in BASE_FEATURES.items():

            ## Construyo la clave del nuevo diccionario de features que voy a construír
            key = f"{axis}_{feat}"

            ## Asocio a la feature la etiqueta y el nombre del archivo de gráfico correspondiente
            FEATURES_INFO[key] = {"label": f"{label} ({axis})", "file_name": f"{feat}_{axis}"}
        
    ## Itero para cada una de las features globales que extraigo del giro
    for feat, label in GLOBAL_FEATURES.items():

        ## Asocio la feature global con su etiqueta y nombre de archivo correspondiente
        FEATURES_INFO[feat] = {"label": label, "file_name": feat}

    ## Retorno el diccionario con la información de todas las features
    return FEATURES_INFO

## Obtengo la información y los metadatos correspondientes a todas las features que tengo
FEATURES_INFO = build_feature_schema()

## Obtengo una lista con los nombres de las columnas de las features
FEATURE_COLUMNS = list(FEATURES_INFO.keys())

## Obtengo una lista con todos los nombres de las features
FEATURE_NAMES = {k: v["label"] for k, v in FEATURES_INFO.items()}

## Obtengo una lista con todos los nombres de los archivos de gráficos asociados a las features
FEATURE_FILE_NAMES = {k: v["file_name"] for k, v in FEATURES_INFO.items()}