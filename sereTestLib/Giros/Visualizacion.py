import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

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
    plt.ylabel('Porcentaje (%)' if mostrar_porcentaje else 'Número de personas')

    ## Agrego la cuadrícula al gráfico para poder facilitar la interpretación
    plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)

    ## Agrego las etiquetas encima de cada barra
    for i, valor in enumerate(conteos.values):
        texto = f"{valor:.1f}%" if mostrar_porcentaje else str(int(valor))
        plt.text(i, valor, texto, ha = 'center', va = 'bottom')

    ## Despliego la gráfica
    plt.tight_layout()
    plt.show()

def graficar_caidas_por_rango_etario(df, columna_edad = "Edad",
                                    columna_caida = "Caida",
                                    rango_edad = [75, float("inf")]):
    """
    Grafica la distribución absoluta del número de caídas
    para un rango etario determinado.

    La función filtra los sujetos pertenecientes al intervalo de edad
    especificado y construye un gráfico de barras donde:

        - El eje X representa el número de caídas {0, 1, 2}
        - El eje Y representa la frecuencia absoluta de individuos

    Esta representación permite caracterizar poblaciones específicas
    según antecedentes de caídas, manteniendo consistencia visual
    con los histogramas utilizados en el resto del pipeline.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la información de los pacientes.

    columna_edad : str, opcional (default = "Edad")
        Nombre de la columna que contiene la edad de los sujetos.

    columna_caida : str, opcional (default = "Caida")
        Nombre de la columna que contiene el número de caídas.

    rango_edad : list, opcional (default = [75, float("inf")])
        Intervalo etario utilizado para el filtrado.
        Debe especificarse como:
            [edad_min, edad_max]

        Ejemplos:
            [60, 75]
            [75, float("inf")]
    """

    ## Obtengo el límite inferior del rango etario
    edad_min = rango_edad[0]

    ## Obtengo el límite superior del rango etario
    edad_max = rango_edad[1]

    ## Me quedo solamente con aquellas personas dentro del rango eterio especificado
    df_filtrado = df[(df[columna_edad] >= edad_min) & (df[columna_edad] < edad_max)].copy()

    ## Hago la conversión de la columna cuyos valores son el número de caídas
    df_filtrado[columna_caida] = pd.to_numeric(df_filtrado[columna_caida], errors="coerce")

    ## Elimino aquellas filas que no tengan ningún valor como número de caídas
    df_filtrado = df_filtrado.dropna(subset = [columna_caida])

    ## Especifico el conteo absoluto restringido al conjunto {0,1,2} para la nomenclatura del eje x
    conteos = (df_filtrado[columna_caida].value_counts().reindex([0, 1, 2], fill_value = 0).sort_index())

    ## Inicializo el gráfico y configuro sus dimensiones
    plt.figure(figsize = (7,5))

    ## Especifico parámetros del gráfico de barras (colores y contorno)
    plt.bar([0, 1, 2], conteos.values, color = 'green', edgecolor = "black")

    ## Construcción del título del gráfico
    if edad_max == float("inf"):
        titulo = f"Distribución de caídas (Edad ≥ {edad_min})"
    else:
        titulo = f"Distribución de caídas ({edad_min} ≤ Edad < {edad_max})"

    ## Configuración de título del gráfico y nomenclatura de los ejes
    plt.title(titulo, fontsize = 14)
    plt.xlabel("Número de caídas")
    plt.ylabel("Número de personas")

    ## Obligo una restricción implícita a los valores del eje x
    plt.xticks([0, 1, 2])

    ## Configuro la grilla del historgrama
    plt.grid(axis = "y", linestyle = "--", alpha = 0.7)

    ## Despliego las etiquetas sobre las barras con los valores numéricos correspondientes
    for i, valor in enumerate(conteos.values):
        plt.text(i, valor, str(int(valor)),
                ha = "center", va="bottom")

    ## Despliego la gráfica
    plt.tight_layout()
    plt.show()

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

    ## Generar la carpeta/ruta de destino en caso de que esta no esté creada
    os.makedirs(output_dir, exist_ok = True)

    ## Defino las etiquetas correspondientes a los grupos etarios
    group_labels = ["<60", "60 - 75", ">75"]

    ## Creo una lista con las etiquetas de los grupos para indexar en el dataframe
    groups = [0, 1, 2]

    ## Itero para cada una de las columnas de features que voy a analizar
    for feature in feature_cols:

        ## Hago la extracción de los datos correspondientes a dicha feature por grupo etario
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

def plot_turn_features_by_age_group(df, feature_cols, output_dir, feature_names = None,
                                    feature_file_names = None):
    """
    Genera boxplots de features a nivel de giro agrupadas por grupo etario.

    Parameters
    ----------
    df : pandas.DataFrame
        Debe contener columnas de features a nivel de giro y la columna
        'age_group' con valores {0, 1, 2}.

    feature_cols : list of str
        Lista de nombres de las features a graficar.

    output_dir : str
        Directorio base donde se guardarán los gráficos. Dentro de este
        se crea automáticamente una subcarpeta con timestamp para cada ejecución.

    feature_names : dict, opcional
        Diccionario que mapea el nombre técnico de la feature a una etiqueta
        legible (usado para los títulos de los gráficos).

    feature_file_names : dict, opcional
        Diccionario que mapea el nombre técnico de la feature a un nombre
        seguro para archivos (usado para los nombres de los PNG generados).
    """

    ## Construyo el timestamp en el cual indica el instante en el que se generan los gráficos
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ## Construyo la ruta completa de salida concatenando la ruta pasada como parámetro y el timestamp
    run_dir = os.path.join(output_dir, timestamp)

    ## Generar la carpeta/ruta de destino en caso de que esta no esté creada
    os.makedirs(run_dir, exist_ok = True)

    ## Agrupo el dataset de giros según el grupo etario de la persona a la que pertenezcan
    grouped = df.groupby("age_group")

    ## Defino las etiquetas correspondientes a los grupos etarios
    labels = ["<60", "60-75", ">75"]

    ## Itero para cada una de las columnas de features que voy a analizar
    for feature in feature_cols:

        ## Hago la extracción de los datos correspondientes a dicha feature por grupo etario
        data = [grouped.get_group(g)[feature].dropna() if g in grouped.groups else [] for g in [0, 1, 2]]

        ## Genero la figura del boxplot pasando como parámetros los datos del feature y las etiquetas
        plt.figure(figsize = (6, 4))
        plt.boxplot(data, tick_labels = labels)

        ## Configuro los rasgos estéticos y el título del boxplot
        title = feature_names.get(feature, feature) if feature_names else feature
        plt.title(title)
        plt.xlabel("Rango etario")
        plt.ylabel(feature)
        plt.grid(True, axis = "y")
        plt.tight_layout()

        ## Especifico el nombre del boxplot que voy a guardar
        safe_name = feature_file_names.get(feature, feature)

        ## Guardo el boxplot correspondiente en la ruta especificada de salida
        save_path = os.path.join(run_dir, f"{safe_name}_por_grupo_etario.png")
        plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
        plt.close()

def plot_turn_feature_pairs_by_age_group(df, feature_pairs, output_dir,
                                        feature_names = None,
                                        feature_file_names = None):
    """
    Genera scatter plots 2D de pares de features a nivel de giro,
    coloreados por grupo etario.

    Parameters
    ----------
    df : pandas.DataFrame
        Debe contener features a nivel de giro y la columna 'age_group'
        con valores {0, 1, 2}.

    feature_pairs : list of tuple
        Lista de pares (feature_x, feature_y) a graficar.

    output_dir : str
        Directorio base donde se guardarán los gráficos.
        Se crea automáticamente una carpeta con timestamp.

    feature_names : dict, opcional
        Mapeo de nombre técnico → etiqueta legible para ejes y títulos.

    feature_file_names : dict, opcional
        Mapeo de nombre técnico → nombre seguro para archivos.
    """

    ## Construyo el timestamp en el cual indica el instante en el que se generan los gráficos
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    ## Construyo la ruta completa de salida concatenando la ruta pasada como parámetro y el timestamp
    run_dir = os.path.join(output_dir, timestamp)

    ## Generar la carpeta/ruta de destino en caso de que esta no esté creada    
    os.makedirs(run_dir, exist_ok = True)

    ## Agrupo el dataset de giros según el grupo etario de la persona a la que pertenezcan
    grouped = df.groupby("age_group")

    ## Defino las etiquetas correspondientes a los grupos etarios
    labels = {0: "<60", 1: "60-75", 2: ">75"}

    ## Itero para todos los pares de features para los que quiero graficar
    for f1, f2 in feature_pairs:

        ## Inicializo y configuro el tamaño del gráfico
        plt.figure(figsize = (6, 5))

        ## Itero para cada uno de los groupos estarios que tengo
        for g in [0, 1, 2]:

            ## Itero para cada una de las agrupaciones de datos de entrada
            if g in grouped.groups:

                ## Obtengo únicamente los vectores de features de los giros correspondientes al grupo g
                data = grouped.get_group(g)

                ## Hago el gráfico de dispersión correspondiente a uno de los grupos etarios
                plt.scatter(data[f1], data[f2], alpha = 0.5, label = labels[g])

        ## Configuraciones de títulos y ejes del gráfico de dispersión
        title = f"{feature_names.get(f1, f1)} vs {feature_names.get(f2, f2)}" if feature_names else f"{f1} vs {f2}"
        xlabel = feature_names.get(f1, f1) if feature_names else f1
        ylabel = feature_names.get(f2, f2) if feature_names else f2

        ## Configuraciones adicionales del gráfico y despliegue de la gráfica
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        ## Configuro nombre del archivo del gráfico
        f1_safe = feature_file_names.get(f1, f1) if feature_file_names else f1
        f2_safe = feature_file_names.get(f2, f2) if feature_file_names else f2

        ## Hago el guardado del gráfico con el nombre correspondiente
        filename = f"{f1_safe}_vs_{f2_safe}.png"
        save_path = os.path.join(run_dir, filename)
        plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
        plt.close()

def plot_clusters_2d(df, feature_x, feature_y, labels, centroids = None):
    """
    Grafica clusters en 2D usando dos features.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset con las features.
    feature_x : str
        Feature eje X.
    feature_y : str
        Feature eje Y.
    labels : array-like
        Etiquetas de cluster para cada punto.
    centroids : np.ndarray or pd.DataFrame, optional
        Centroides del clustering (K x 2 o K x n_features).
    """

    ## Inicializo la figura y configuro las dimensiones del gráfico correspondiente
    plt.figure(figsize = (7, 6))

    ## Obtengo el conjunto de valores correspondientes al feature x
    x = df[feature_x].values

    ## Obtengo el conjunto de valores correspondientes al feature y
    y = df[feature_y].values

    ## Inicializo el diagrama de dispersión correspondiente
    scatter = plt.scatter(x, y, c = labels, cmap = "viridis", alpha = 0.6)

    ## En caso de que se hayan provisto los centroides de los clústers a la entrada
    if centroids is not None:

        ## Grafico los centroides con una X dentro del gráfico de dispersión
        plt.scatter(centroids[:, 0], centroids[:, 1], c = "red", marker = "X", s = 200, label = "Centroids")

    ## Nomenclatura de ejes, título y despliegue del gráfico
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title("Clusters en espacio de features")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()