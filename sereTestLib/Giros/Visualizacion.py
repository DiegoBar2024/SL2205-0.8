import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib import use
from sklearn.metrics import confusion_matrix
import math

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

def graficar_caidas_por_rango_etario(df, columna_edad = "Edad", columna_caida = "Caida",
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
                ha = "center", va = "bottom")

    ## Despliego la gráfica
    plt.tight_layout()
    plt.show()

def graficar_variable_por_rango_etario(df,
                                       columna_edad = "Edad",
                                       columna_variable = "Caida",
                                       rango_edad = [75, float("inf")],
                                       categorias = None,
                                       titulo_variable = "Distribución",
                                       etiqueta_x = "Valor"):
    """
    Grafica la distribución absoluta de una variable para un rango etario
    específico.

    La función filtra los sujetos pertenecientes al intervalo de edad
    especificado y construye un gráfico de barras donde:

        - El eje X representa las categorías o valores posibles de la variable.
        - El eje Y representa la frecuencia absoluta de individuos para cada
          categoría.

    Esta representación permite caracterizar poblaciones específicas según
    cualquier variable discreta contenida en el DataFrame (por ejemplo,
    número de caídas, uso de bastón, uso de andador, etc.), manteniendo una
    representación gráfica consistente a lo largo del pipeline.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la información de los pacientes.

    columna_edad : str, opcional (default = "Edad")
        Nombre de la columna que contiene la edad de los sujetos.

    columna_variable : str, opcional (default = "Caida")
        Nombre de la variable cuya distribución se desea representar.

    rango_edad : list, opcional (default = [75, float("inf")])
        Intervalo etario utilizado para el filtrado.
        Debe especificarse como:

            [edad_min, edad_max]

        Ejemplos:

            [60, 75]
            [75, float("inf")]

    categorias : list, opcional (default = None)
        Lista de categorías que se desean representar en el eje X.

        Si se especifica, todas las categorías indicadas aparecerán en el
        gráfico, incluso aquellas cuya frecuencia sea cero.

        Si es None, las categorías se obtienen automáticamente a partir de
        los valores presentes en la variable.

    titulo_variable : str, opcional (default = "Distribución")
        Texto utilizado para construir el título del gráfico.

    etiqueta_x : str, opcional (default = "Valor")
        Etiqueta utilizada para nombrar el eje X.
    """

    ## Obtengo el límite inferior del rango etario
    edad_min = rango_edad[0]

    ## Obtengo el límite superior del rango etario
    edad_max = rango_edad[1]

    ## Me quedo solamente con aquellas personas dentro del rango etario especificado
    df_filtrado = df[(df[columna_edad] >= edad_min) & (df[columna_edad] < edad_max)].copy()

    ## Convierto la variable de interés a formato numérico
    df_filtrado[columna_variable] = pd.to_numeric(df_filtrado[columna_variable], errors = "coerce")

    ## Elimino aquellas filas cuyo valor para la variable de interés sea inválido
    df_filtrado = df_filtrado.dropna(subset = [columna_variable])

    ## Si el usuario no especificó las categorías, las obtengo automáticamente
    ## a partir de los valores presentes en la variable
    if categorias is None:
        categorias = sorted(df_filtrado[columna_variable].unique())

    ## Calculo la frecuencia absoluta para cada categoría especificada
    ## completando con cero aquellas categorías que no aparezcan
    conteos = (df_filtrado[columna_variable].value_counts().reindex(categorias, fill_value = 0).sort_index())

    ## Inicializo el gráfico y configuro sus dimensiones
    plt.figure(figsize = (7, 5))

    ## Especifico parámetros del gráfico de barras (colores y contorno)
    plt.bar(categorias, conteos.values, color = "green", edgecolor = "black")

    ## Construyo el título del gráfico en función del rango etario analizado
    if edad_max == float("inf"):
        titulo = f"{titulo_variable} (Edad ≥ {edad_min})"
    else:
        titulo = f"{titulo_variable} ({edad_min} ≤ Edad < {edad_max})"

    ## Configuro el título del gráfico y la nomenclatura de los ejes
    plt.title(titulo, fontsize = 14)
    plt.xlabel(etiqueta_x)
    plt.ylabel("Número de personas")

    ## Obligo a que todas las categorías especificadas aparezcan en el eje X
    plt.xticks(categorias)

    ## Configuro la grilla del histograma
    plt.grid(axis = "y", linestyle = "--", alpha = 0.7)

    ## Despliego las etiquetas numéricas sobre cada barra
    for categoria, valor in zip(categorias, conteos.values):
        plt.text(categoria, valor, str(int(valor)), ha = "center", va = "bottom")

    ## Ajusto automáticamente los márgenes de la figura
    plt.tight_layout()

    ## Despliego el gráfico
    plt.show()

def plot_signal_with_events(x, events, fs = 200, title = "Señal y los eventos de giro", save_path = None):
    """
    Grafica una señal continua y resalta los segmentos correspondientes a eventos.

    La señal completa se representa en color azul, mientras que los intervalos
    definidos como eventos se superponen en color rojo. Esto permite identificar
    visualmente cuándo ocurren los eventos sin perder la continuidad de la señal.

    Parameters
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

    save_path : str or None, opcional (default=None)
        Ruta completa donde se guardará la figura.
        Si es None, la figura solo se muestra en pantalla y no se guarda.

        Si se especifica, la función:
            - Crea automáticamente el directorio contenedor si no existe
            - Guarda la figura en formato PNG (o extensión implícita en el nombre)
            - Usa alta resolución (dpi=300)

    Description
    -----------
    La función:
        1. Construye un eje temporal a partir de la frecuencia de muestreo.
        2. Genera una máscara booleana para identificar los segmentos de eventos.
        3. Grafica la señal completa en azul.
        4. Superpone los segmentos correspondientes a eventos en rojo utilizando NaN
        para evitar la conexión entre tramos no contiguos.
        5. Opcionalmente guarda la figura en disco.

    Notes
    -----
    - El uso de valores NaN permite mantener la continuidad visual de la señal
    mientras se resaltan únicamente los segmentos de interés.
    - Esta función es útil para visualizar eventos detectados en señales de sensores,
    como giros en datos de IMU.
    - El guardado es opcional y controlado mediante `save_path`.

    Returns
    -------
    None
        La función no retorna valores. Muestra el gráfico en pantalla y,
        si corresponde, lo guarda en disco.
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
        event_mask[e["start_idx"]: e["end_idx"]] = True

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

    ## En caso de que se haya definido una ruta de guardado para la figura
    if save_path is not None:

        ## Extraigo el directorio a partir de la ruta completa del archivo
        ## (elimina el nombre del archivo y deja solo la carpeta contenedora)
        dir_name = os.path.dirname(save_path)

        ## Verifico que la ruta del directorio no esté vacía
        ## (esto ocurre cuando save_path es solo un nombre de archivo sin carpeta)
        if dir_name != "":

            ## Creo el directorio en caso de que no exista previamente
            ## exist_ok=True evita errores si la carpeta ya fue creada
            os.makedirs(dir_name, exist_ok = True)

        ## Guardo la figura actual en la ruta especificada
        ## dpi=300 asegura una buena resolución para análisis o publicación
        plt.savefig(save_path, dpi = 300)

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

    ## Agrego configuraciones necesarias para que los gráficos se guarden y no se desplieguen
    ## en la pantall mientras está ejecutándose la función (se vuelve molesto sino)
    use("Agg")
    plt.ioff()

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

def plot_kruskal_results(results_df, top_n = 20):
    """
    Visualiza los resultados del test de Kruskal–Wallis ordenados por tamaño de efecto.

    Esta función genera un gráfico de barras que muestra las features más
    discriminativas según el estadístico epsilon cuadrado (ε²), el cual mide
    el tamaño de efecto del test no paramétrico de Kruskal–Wallis.

    El objetivo es identificar qué variables presentan mayor capacidad de
    separación entre grupos (por ejemplo, grupos etarios o condiciones clínicas).

    Parámetros
    ----------
    results_df : pandas.DataFrame
        DataFrame con los resultados del test de Kruskal–Wallis. Debe contener:
            - feature: nombre de la variable
            - epsilon_sq: tamaño de efecto (ε²)

    top_n : int, opcional (default=20)
        Número de features más relevantes a visualizar, ordenadas por ε².

    Retorna
    -------
    None
        La función genera y muestra un gráfico, sin retornar valores.

    Notas
    -----
    - Un mayor valor de ε² indica mayor capacidad discriminativa de la feature
    entre los grupos analizados.
    - Este gráfico es útil para selección de variables en pipelines de machine learning.
    """

    ## A partir de los datos de entrada obtengo un dataframe ordenando las features de mayor a menor por
    ## el valor de epsilon cuadrado, manteniendo los primeros <<top_n>> valores
    df = results_df.sort_values("epsilon_sq", ascending = False).head(top_n)

    ## Configuro el tamaño de la figura
    plt.figure(figsize = (10,6))

    ## Configuro los datos a graficar en cada uno de los ejes
    sns.barplot(data = df, y = "feature", x = "epsilon_sq")

    ## Configuro título, nomenclatura de ejes y despliego el gráfico
    plt.title("Kruskal-Wallis: Valor de ε² por feature")
    plt.xlabel("ε²")
    plt.ylabel("Feature")
    plt.grid(axis = "x", alpha = 0.3)
    plt.tight_layout()
    plt.show()

def plot_wilcoxon_significance_matrices(results_df, output_dir, use_fdr = True,
                                    feature_names = None, use_timestamp = True, alpha = 0.05):
    """
    Genera matrices visuales de significancia estadística para comparaciones
    por pares entre grupos utilizando el test de Wilcoxon rank-sum (Mann–Whitney U).

    Esta función toma los resultados de análisis estadísticos previamente calculados
    entre grupos etarios y los reorganiza en forma de matrices simétricas
    (grupo × grupo) para cada feature, facilitando la interpretación visual
    de diferencias significativas entre distribuciones.

    Cada matriz representa, para una feature dada, si existe o no diferencia
    estadísticamente significativa entre pares de grupos según un umbral de significancia.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame con los resultados de comparaciones por pares. Debe contener:

        - feature : nombre de la variable analizada
        - group_1 : primer grupo de comparación
        - group_2 : segundo grupo de comparación
        - p_value : p-valor del test (si use_fdr=False)
        - significant_fdr : booleano de significancia corregida (si use_fdr=True)

    output_dir : str
        Directorio base donde se guardarán las figuras generadas.
        Si use_timestamp=True, se crea una subcarpeta con la fecha y hora actual.

    use_fdr : bool, opcional (default=True)
        Define si la significancia se evalúa usando corrección FDR
        (False Discovery Rate) o p-values crudos.

    feature_names : dict, opcional
        Diccionario que mapea nombres técnicos de features a etiquetas
        más legibles para los títulos de los gráficos.

    use_timestamp : bool, opcional (default=True)
        Si es True, crea una carpeta de salida con timestamp para evitar
        sobreescritura de resultados entre ejecuciones.

    alpha : float, opcional (default=0.05)
        Nivel de significancia estadística utilizado cuando use_fdr=False.
        Un valor p < alpha se considera estadísticamente significativo.

        Este parámetro también se utiliza para:
        - Etiquetado de la leyenda del gráfico
        - Inclusión en el título de la figura
        - Nomenclatura de los archivos generados

    Returns
    -------
    None
        La función no retorna valores. Genera y guarda imágenes en disco
        para cada feature analizada.

    Notas
    -----
    - Las matrices son simétricas: (g1, g2) = (g2, g1).
    - La diagonal principal se deja como NaN (sin comparación intra-grupo).
    - Los valores se codifican como:
        - 1.0 → diferencia significativa
        - 0.0 → no significativa
        - NaN → comparación no definida (misma clase)
    - El parámetro alpha controla el umbral de decisión cuando no se usa FDR.
    - Las figuras incluyen una leyenda explícita para interpretación visual.
    - Se utiliza un colormap discreto para facilitar la lectura cualitativa
    de los resultados estadísticos.
    """

    ## Agrego configuraciones necesarias para que los gráficos se guarden y no se desplieguen
    ## en la pantall mientras está ejecutándose la función (se vuelve molesto sino)
    use("Agg")
    plt.ioff()

    ## En caso de que quiera usar el timestamp para guardar los gráficos
    if use_timestamp:

        ## Construyo el timestamp correspondiente al instante actual del gráfico
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        ## Construyo la ruta de salida donde voy a guardar los gráficos
        output_dir = os.path.join(output_dir, timestamp)

    ## En caso de que la ruta de salida no exista, la construyo
    os.makedirs(output_dir, exist_ok = True)

    ## Itero para cada una de las features extraídas de las señales de los giros
    for feat in results_df["feature"].unique():

        ## Selecciono únicamente aquellos resultados del test asociados a la feature que estoy estudiando
        sub = results_df[results_df["feature"] == feat]

        ## Obtengo la lista que contiene todos los grupos etarios
        groups = sorted(set(sub["group_1"]).union(set(sub["group_2"])))

        ## Inicializo la matriz donde voy a representar los resultados de las discriminaciones
        ## por Wilcoxon. Inicialmente todas las entradas de la matriz corresponden a NaN
        mat = pd.DataFrame(np.nan, index = groups, columns = groups, dtype = float)

        ## Configuro la correspondiente matriz de anotación para la escritura de los p-valores
        annot_mat = pd.DataFrame("", index = groups, columns = groups)

        ## Configuro una matriz numérica que me permita almacenar los p-valores
        p_mat = pd.DataFrame(np.nan, index = groups, columns = groups)

        ## Itero para cada uno de los resultados del test de Wilcoxon asociados a la feature de estudio
        for _, row in sub.iterrows():

            ## Obtengo el par de grupos etarios que estoy comparando en el resultado
            g1, g2 = row["group_1"], row["group_2"]

            ## En caso de que quiera usar el indicador FDR
            if use_fdr:

                ## Configuro el resultado correspondiente a dicho indicador
                is_sig = bool(row["significant_fdr"])
            
            ## En caso de que no quiera utilizar el indicador FDR
            else:

                ## Expreso el resultado del test a partir del p-valor y el nivel de significación
                ## correspondiente (alpha, pasado como entrada de la función)
                is_sig = row["p_value"] < alpha

            ## Selecciono el p-valor correspondiente al test de hipótesis realizado
            p = row["p_value"]

            ## Asigno val == 1 en caso de que la diferencia detectada por el test de Wilcoxon sea significativa
            val = 1 if is_sig else 0

            ## Dado que el resultado del test es simétrico (no depende del orden de los grupos etarios)
            ## configuro el mismo valor de resultado de significación del test en las entradas de ambos grupos
            mat.loc[g1, g2] = val
            mat.loc[g2, g1] = val

            ## Configuro la anotación del p-valor del test de hipótesis de manera simétrica en la matriz
            annot_mat.loc[g1, g2] = f"p = {p:.2e}"
            annot_mat.loc[g2, g1] = f"p = {p:.2e}"

            ## Configuro el p-valor en la matriz correspondiente a los p-valores
            p_mat.loc[g1, g2] = p
            p_mat.loc[g2, g1] = p

        ## Construyo un diccionario que contiene las etiquetas asociadas a los grupos etarios
        label_map = {0: "<60", 1: "60-75", 2: ">75"}

        ## Configuro las etiquetas de los grupos etarios en las filas de la matriz
        mat.index = [label_map.get(g, str(g)) for g in mat.index]

        ## Configuro las etiquetas de los grupos etarios en las columnas de la matriz
        mat.columns = [label_map.get(g, str(g)) for g in mat.columns]

        ## Inicializo el gráfico configurando el tamaño y las dimensiones correspondientes
        fig, ax = plt.subplots(figsize = (5, 4))

        ## Inicializo el objeto que representa el mapa de colores que voy a utilizar para las matrices
        cmap = ListedColormap(["lightgray", "darkred"])

        ## Configuro el blanco como el color asociado al NaN
        cmap.set_bad(color = "white")

        ## Configuro los parámetros correspondientes al mapa de calor
        sns.heatmap(mat, cmap = cmap, vmin = 0, vmax = 1, cbar = False, linewidths = 1, 
            linecolor = "black", square = True, ax = ax, annot = annot_mat, fmt = "")

        ## Construyo el título asociado al gráfico correspondiente
        title = feature_names.get(feat, feat) if feature_names else feat

        ## Configuro el título que construí anteriormente para que quede visible en el gráfico
        ax.set_title(f"{title}\nSignificancia Wilcoxon por pares (α = {alpha})", fontsize = 11, pad = 10)

        ## Construyo los contenidos de las leyendas/referencias correspondientes a los gráficos
        legend_elements = [Patch(facecolor = "darkred", edgecolor = "black", 
                                label = f"Diferencia significativa (p-valor < {alpha})"),
                            Patch(facecolor = "lightgray", edgecolor = "black", 
                                label = f"No significativo (p-valor ≥ {alpha})"),
                            Patch(facecolor = "white", edgecolor = "black", 
                                label = "Mismo grupo (NaN)")]

        ## Configuro las referencias/leyendas que construí anteriormente para que queden visibles
        ax.legend(handles = legend_elements, loc = "upper left", bbox_to_anchor = (1.02, 1), frameon = True)

        ## Hago el ajuste de layout correspondiente al gráfico
        plt.subplots_adjust(right = 0.8)

        ## Configuro la nomenclatura del gráfico asociado a la significación de Wilcoxon
        safe_name = feat.replace("/", "_")
        filename = f"{safe_name}_significancia_wilcoxon.png"

        ## Construyo la ruta completa donde voy a guardar el gráfico
        save_path = os.path.join(output_dir, filename)

        ## Guardo el gráfico correspondiente especificando el formato y el tamaño
        plt.savefig(save_path, dpi = 300, bbox_inches = "tight")

        ## Finalizo el procesamiento de la figura (para que los gráficos subsiguientes no se superpongan)
        plt.close(fig)

def plot_features_vs_age(df, feature_cols, output_dir, feature_names = None, x_col = "age",
    results_df = None):
    """
    Genera gráficos de dispersión entre edad y un conjunto de features,
    con soporte opcional para enriquecer la visualización con resultados
    de regresión previamente calculados.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset que contiene la variable independiente (edad) y las features.

    feature_cols : list of str
        Lista de columnas a graficar como variables dependientes.

    output_dir : str
        Carpeta donde se guardarán las figuras generadas.

    feature_names : dict, optional
        Diccionario que mapea nombres técnicos a etiquetas legibles para los gráficos.

    x_col : str, default="age"
        Nombre de la variable independiente (edad).

    results_df : pandas.DataFrame, optional
        Resultados de `regression_analysis`. Si se provee, se usan para
        mostrar información adicional (p-value y R²) en el título del gráfico.

    Returns
    -------
    None
    """

    ## Itero para cada feature seleccionada en feature_cols
    for feat in feature_cols:

        ## En caso de que no tenga ningún valor asociado a dicha feature
        if feat not in df.columns:

            ## Continúo con la ejecución de la siguiente feature
            continue

        ## Obtengo el conjunto de valores de edades a las que corresponden los giros
        x = df[x_col].values

        ## Obtengo el conjunto de valores de las features a las que corresponden los giros
        y = df[feat].values

        ## Inicializo el gráfico y configuro el tamaño y dimensiones del mismo
        plt.figure(figsize = (5, 4))

        ## Hago la graficación del diagrama de dispersión con los pares de valores correspondientes
        plt.scatter(x, y, alpha = 0.5, s = 15)  

        ## En caso de que yo de los resultados de los análisis de regresión como parámetro
        if results_df is not None:

            ## Selecciono la fila correspondiente a la feature en los resultados de regresión
            row = results_df[results_df["feature"] == feat]

            ## En caso de que exista una fila de resultados para esta feature
            if not row.empty:

                ## Selecciono el subconjunto de resultados numéricos más relevante para la ilustración
                row = row.iloc[0]

                ## En caso de que tenga información acerca del slope y el intercept del modelo de regresion
                if "slope" in row and "intercept" in row:

                    ## Selecciono el valor del slope (pendiente)
                    slope = row["slope"]

                    ## Selecciono el valor del intercept (ordenada en el origen)
                    intercept = row["intercept"]

                    ## Me aseguro de ordenar las edades para que la linea sea correcta
                    x_sorted = np.sort(x)

                    ## Obtengo analíticamente la recta del modelo de regresión lineal
                    y_pred = slope * x_sorted + intercept

                    ## Grafico la recta de regresión junto con los datos correspondientes
                    plt.plot(x_sorted, y_pred, color = "red", linewidth = 2, label = "Linear fit")

                ## Configuro el título del gráfico incluyendo métricas de regresión
                plt.title(f"{feature_names.get(feat, feat) if feature_names else feat} "
                        f"(p={row['p_value']:.3g}, R²={row['linear_r2']:.4f})")
            
            ## En caso de que no tenga resultados numéricos de hacer análisis de regresión
            else:

                ## Configuro el título del gráfico correspondiente
                plt.title(feature_names.get(feat, feat) if feature_names else feat)
        
        ## En caso de que no tenga resultados generales de análisis de regresión de la feature
        else:

            ## Configuro el título del gráfico correspondiente
            plt.title(feature_names.get(feat, feat) if feature_names else feat)

        ## Configuro nomenclatura de ejes del gráfico. Agrego la leyenda del gráfico correspondiente
        plt.xlabel(x_col)
        plt.ylabel(feat)
        plt.tight_layout()
        plt.legend()

        ## Configuro la nomenclatura del gráfico correspondiente
        safe_name = feat.replace("/", "_")
        
        ## Construyo la ruta completa donde voy a guardar el gráfico
        save_path = os.path.join(output_dir, f"{safe_name}_vs_{x_col}.png")

        ## Guardo el gráfico correspondiente especificando el formato y el tamaño
        plt.savefig(save_path, dpi = 300, bbox_inches = "tight")

        ## Finalizo el procesamiento de la figura (para que los gráficos subsiguientes no se superpongan)
        plt.close()

def save_regression_plots(df, results_df, feature_cols, output_dir, x_col = "age", 
                        only_significant = False, use_timestamp = True):
    """
    Genera y guarda gráficos de features vs edad usando resultados
    ya computados de regresión.

    Esta función actúa como wrapper de `plot_features_vs_age`, filtrando
    features a partir de los resultados de `regression_analysis` y
    generando un gráfico por feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset original con edad y features.

    results_df : pandas.DataFrame
        Salida de `regression_analysis` con métricas por feature.

    feature_cols : list of str
        Features candidatas a graficar.

    output_dir : str
        Carpeta base donde se guardarán los gráficos.
        Si `use_timestamp=True`, se crea automáticamente una subcarpeta
        con timestamp.

    x_col : str, default="age"
        Variable independiente usada como eje X.

    only_significant : bool, default=False
        Si True, solo grafica features marcadas como significativas
        en `results_df`.

    use_timestamp : bool, default=True
        Si True, crea una subcarpeta con timestamp dentro de `output_dir`
        para evitar sobrescritura de resultados.

    Returns
    -------
    None
    """

    ## En caso de que quiera usar el timestamp para guardar los gráficos
    if use_timestamp:

        ## Construyo el timestamp correspondiente al instante actual del gráfico
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        ## Construyo la ruta de salida donde voy a guardar los gráficos
        output_dir = os.path.join(output_dir, timestamp)

    ## En caso de que la ruta de salida no exista, la construyo
    os.makedirs(output_dir, exist_ok = True)

    ## En caso de que solo quiera graficar aquellas features que son significativas
    if only_significant:

        ## Selecciono únicamente aquellas features que son significativas para la graficación
        feature_cols = results_df.loc[results_df["significant"], "feature"].tolist()

    ## Itero para cada feature seleccionada en feature_cols
    for feat in feature_cols:

        ## Obtengo los resultados de los análisis de regresión correspondientes a la feature <<feat>>
        feat_stats = results_df[results_df["feature"] == feat]

        ## En caso de que para una feature en particular no haya ningún resultado estadístico
        if feat_stats.empty:

            ## Continúo con el procesamiento de la siguiente feature
            continue

        ## Obtengo los resultados relvantes del análisis de regresión correspondientes a la feature
        feat_stats = feat_stats.iloc[0].to_dict()

        ## Genero el gráfico de la feature contra la edad incluyendo métricas de regresión en el título
        plot_features_vs_age(df = df, feature_cols = [feat], output_dir = output_dir,
            feature_names = None, x_col = x_col, results_df = pd.DataFrame([feat_stats]))

def plot_cluster_age_distributions(df, cluster_col = "cluster", age_col = "age_group", show_counts = True):
    """
    Visualiza la relación entre clusters de giros y grupos etarios mediante
    distribuciones condicionales normalizadas.

    Se construyen dos matrices de contingencia:

    1. P(age | cluster)
    Distribución de grupos etarios dentro de cada cluster
    (normalización por filas).

    2. P(cluster | age)
    Distribución de clusters dentro de cada grupo etario
    (normalización por columnas).

    Se incluyen opcionalmente los tamaños de muestra por cluster
    para contextualizar las distribuciones.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset a nivel de giro con columnas de cluster y grupo etario.

    cluster_col : str, default="cluster"
        Columna con etiquetas de cluster.

    age_col : str, default="age_group"
        Columna con grupos etarios.

    Returns
    -------
    tuple (cluster_age, age_cluster)
        cluster_age : P(age | cluster)
        age_cluster : P(cluster | age)
    """

    ## Obtengo la matriz (tabla dinámica) con los conteos absolutos
    counts = pd.crosstab(df[cluster_col], df[age_col])

    ## Hago la agrupación de los estadísticos por clúster
    cluster_age = pd.crosstab(df[cluster_col], df[age_col], normalize = "index")

    ## Hago la agrupación de los estadísticos por rango etario
    age_cluster = pd.crosstab(df[cluster_col], df[age_col], normalize = "columns")

    ## Cuento la cantidad de giros que tengo por cada clúster
    cluster_counts = df[cluster_col].value_counts().sort_index()

    ## Cuento la cantidad de giros que tengo por grupo etario
    age_counts = df[age_col].value_counts().sort_index()

    ## Hago una copia del dataframe que sale de agrupar los giros por clúster
    annot_cluster_age = cluster_age.copy().astype(str)

    ## Hago una copia del dataframe que sale de agrupar los giros por grupo etario
    annot_age_cluster = age_cluster.copy().astype(str)

    ## Itero para cada uno de los índices de los clústers
    for r in cluster_age.index:

        ## Itero para cada uno de los índices de los grupos etarios
        for c in cluster_age.columns:

            ## Obtengo la proporción de giros correspondientes al par grupo etario <<c>> y cluster <<r>>
            ## con respecto al total de puntos en el clúster
            prob = cluster_age.loc[r, c]

            ## Obtengo la cantidad al par grupo etario <<c>> y cluster <<r>>
            n = counts.loc[r, c]

            ## Construyo la anotación correspondiente combinando la proporción y cantidad de puntos dados
            annot_cluster_age.loc[r, c] = f"{prob:.3f}\n({n})"

    ## Itero para cada uno de los índices de los clústers
    for r in age_cluster.index:

        ## Itero para cada uno de los índices de los grupos etarios
        for c in age_cluster.columns:

            ## Obtengo la proporción de giros correspondientes al par grupo etario <<c>> y cluster <<r>>
            ## con respecto al total de puntos en el grupo etario
            prob = age_cluster.loc[r, c]

            ## Obtengo la cantidad al par grupo etario <<c>> y cluster <<r>>
            n = counts.loc[r, c]

            ## Construyo la anotación correspondiente combinando la proporción y cantidad de puntos dados
            annot_age_cluster.loc[r, c] = f"{prob:.3f}\n({n})"

    ## Configuro el tamaño y las dimensiones de la figura correspondiente
    plt.figure(figsize = (7, 4))

    ## Represento la distribución de los rangos etarios dentro de cada clúster
    ax = sns.heatmap(cluster_age, annot = annot_cluster_age, fmt = "", cmap = "viridis", linewidths = 0.5)

    ## En caso de que quiera mostrar la cantidad de elementos por clúster
    if show_counts:

        ## Obtengo las etiquetas que especifican la cantidad de giros por cluster
        y_labels = [f"{idx} (n={cluster_counts.loc[idx]})" if idx in cluster_counts.index else str(idx)
            for idx in cluster_age.index]
    
        ## Agrego las etiquetas de los giros por clúster al gráfico
        ax.set_yticklabels(y_labels, rotation = 0)

        ## Obtengo las etiquetas que especifican la cantidad de giros por grupo etario
        x_labels = [f"{col} (n={age_counts.loc[col]})" if col in age_counts.index else str(col)
            for col in cluster_age.columns]

        ## Agrego las etiquetas de los giros por grupo etario al gráfico
        ax.set_xticklabels(x_labels, rotation = 0)

    ## Configuro nomenclatura de los ejes, el título y despliego el gráfico resultante
    plt.title("Distribución etaria dentro de cada cluster")
    plt.xlabel("Grupo etario")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()

    ## Configuro el tamaño y las dimensiones de la figura correspondiente
    plt.figure(figsize = (7, 4))

    ## Represento la distribución de los clústers dentro de cada rango etario
    ax = sns.heatmap(age_cluster, annot = annot_age_cluster, fmt = "", cmap = "viridis", linewidths = 0.5)

    ## En caso de que quiera mostrar la cantidad de elementos por clúster
    if show_counts:

        ## Obtengo las etiquetas que especifican la cantidad de giros por cluster
        y_labels = [f"{idx} (n={cluster_counts.loc[idx]})" if idx in cluster_counts.index else str(idx)
            for idx in cluster_age.index]
    
        ## Agrego las etiquetas de los giros por clúster al gráfico
        ax.set_yticklabels(y_labels, rotation = 0)

        ## Obtengo las etiquetas que especifican la cantidad de giros por grupo etario
        x_labels = [f"{col} (n={age_counts.loc[col]})" if col in age_counts.index else str(col)
            for col in cluster_age.columns]

        ## Agrego las etiquetas de los giros por grupo etario al gráfico
        ax.set_xticklabels(x_labels, rotation = 0)

    ## Configuro nomenclatura de los ejes, el título y despliego el gráfico resultante
    plt.title("Distribución de clusters dentro de cada grupo etario")
    plt.xlabel("Grupo etario")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()

    ## Retorno las tablas dinámicas al agrupar por rango etario y por clúster
    return cluster_age, age_cluster

def plot_svm_feature_error_ranking(results_svm, top_k = None, annotate = True, title = None,
    C = None, gamma = None, wilcoxon_results = None, group_pair = None, use_fdr = False):
    """
    Visualiza el ranking de features basado en el error de clasificación
    obtenido mediante SVM univariado, incorporando opcionalmente el
    p-valor asociado al test estadístico de Wilcoxon para cada feature.

    La idea del gráfico es combinar dos criterios complementarios de
    discriminación:

        - Capacidad predictiva:
            medida mediante el error de clasificación del SVM univariado.
            Menor error indica una mayor capacidad de separación entre grupos.

        - Evidencia estadística:
            medida mediante el test de Wilcoxon rank-sum, evaluando si las
            distribuciones de una feature son significativamente diferentes
            entre dos grupos determinados.

    Parameters
    ----------
    results_svm : pandas.DataFrame
        DataFrame con resultados del análisis univariado mediante SVM.

        Debe contener al menos:
            - 'feature'
            - 'error' (o 'mean_error')

    top_k : int or None
        Si se especifica, muestra únicamente las top-k features con menor
        error de clasificación SVM.

    annotate : bool
        Si True, muestra sobre cada barra:
            - error de clasificación SVM
            - p-valor de Wilcoxon (si fue proporcionado)

    title : str or None
        Título del gráfico. Si es None, se genera un título automático.
        En caso de proporcionarse los hiperparámetros C y gamma, estos
        son agregados al título.

    C : float or None
        Parámetro de regularización del modelo SVM utilizado.

    gamma : float or str or None
        Parámetro gamma del kernel utilizado en el SVM.

    wilcoxon_results : pandas.DataFrame or None
        Resultados del análisis Wilcoxon rank-sum generado mediante
        `pairwise_wilcoxon_rank_sum()`.

        Debe contener:
            - 'feature'
            - 'group_1'
            - 'group_2'
            - 'p_value'

        Opcionalmente:
            - 'p_fdr'

        Si se proporciona, se agrega el p-valor correspondiente a cada
        feature del ranking SVM.

    group_pair : tuple or None
        Par de grupos utilizados para seleccionar el resultado de Wilcoxon.

        Ejemplo:
            group_pair = (0,2)

        Esto debe coincidir con la separación de grupos utilizada en
        el análisis SVM.

    use_fdr : bool
        Si True y existe la columna 'p_fdr', utiliza el p-valor corregido
        mediante FDR en lugar del p-valor original.

    Returns
    -------
    None
        Genera un gráfico de barras ordenado por error ascendente.

        Cada barra representa una feature y opcionalmente contiene
        información estadística adicional del test Wilcoxon.
    """

    ## Hago una copia del dataframe de entrada que contiene los resultados del SVM
    df = results_svm.copy()

    ## Detecto automáticamente la columna correspondiente al error de clasificación
    ## dependiendo del formato del dataframe de resultados recibido
    error_col = "error" if "error" in df.columns else "mean_error"

    ## Ordeno las features de acuerdo al desempeño del clasificador SVM
    ## dejando primero aquellas features con menor error de predicción
    df = df.sort_values(error_col, ascending = True)

    ## En caso de que se solicite visualizar únicamente las mejores k features,
    ## selecciono aquellas con menor error de clasificación
    if top_k is not None:

        ## Mantengo únicamente las top-k features más discriminativas
        df = df.head(top_k)

    # ==========================================================
    # Incorporación de resultados del test de Wilcoxon
    # ==========================================================

    ## En caso de que se proporcionen resultados estadísticos de Wilcoxon,
    ## agrego los p-valores correspondientes a cada feature
    if wilcoxon_results is not None:

        ## Hago una copia de los resultados de Wilcoxon para evitar modificar
        ## el dataframe original recibido como entrada
        wilcox_df = wilcoxon_results.copy()

        ## Selecciono si utilizar el p-valor original o el corregido mediante FDR
        ## dependiendo del parámetro indicado por el usuario
        p_col = "p_fdr" if use_fdr and "p_fdr" in wilcox_df.columns else "p_value"

        ## En caso de que se indique una separación específica de grupos,
        ## filtro únicamente la comparación de interés
        if group_pair is not None:

            ## Obtengo los dos grupos que quiero comparar mediante Wilcoxon
            g1, g2 = group_pair

            ## Selecciono únicamente la fila correspondiente a dicha comparación,
            ## independientemente del orden en el que aparezcan los grupos
            wilcox_df = wilcox_df[((wilcox_df["group_1"] == g1) & (wilcox_df["group_2"] == g2))|
                ((wilcox_df["group_1"] == g2) & (wilcox_df["group_2"] == g1))]

        ## Mantengo únicamente la información necesaria para hacer el merge
        ## con los resultados del SVM
        wilcox_df = wilcox_df[["feature", p_col]]

        ## Renombro la columna del p-valor para identificar claramente
        ## que corresponde al test de Wilcoxon
        wilcox_df = wilcox_df.rename(columns = {p_col: "wilcoxon_p"})

        ## Uno los resultados del SVM con los resultados estadísticos utilizando
        ## el nombre de la feature como clave común
        df = df.merge(wilcox_df, on = "feature", how = "left")

    ## Extraigo la lista de nombres de features que serán graficadas
    features = df["feature"].values

    ## Extraigo los errores asociados a cada feature
    errors = df[error_col].values

    ## Inicializo el tamaño del gráfico
    plt.figure(figsize = (12, 5))

    ## Construyo el gráfico de barras con el error SVM como métrica principal
    bars = plt.bar(features, errors)

    ## Configuro las etiquetas del eje x rotando los nombres para mejorar
    ## la visualización cuando existen muchas features
    plt.xticks(rotation = 45, ha = "right")

    ## Configuro la etiqueta del eje y indicando la métrica representada
    plt.ylabel("Error de clasificación (SVM univariado)")

    ## En caso de no recibir un título definido por el usuario,
    ## genero uno automáticamente
    if title is None:

        ## Construyo el título estándar del gráfico
        title = ("Ranking de features por capacidad discriminativa (menor error SVM = mejor)")

    ## En caso de que se proporcionen hiperparámetros del SVM,
    ## los agrego al título para dejar registrado el modelo utilizado
    if C is not None or gamma is not None:

        ## Construyo el texto auxiliar con los parámetros del modelo
        hp_text = [f"C = {C}", f"gamma = {gamma}"]

        ## Agrego los hiperparámetros al título principal
        title += " | " + ", ".join(hp_text)

    ## Muestro el título final del gráfico
    plt.title(title)

    ## En caso de querer mostrar anotaciones numéricas sobre las barras
    if annotate:

        ## Itero sobre cada barra y su correspondiente error de clasificación
        for i, (bar, err) in enumerate(zip(bars, errors)):

            ## Inicializo el texto de la anotación indicando explícitamente
            ## que el valor corresponde al error de clasificación SVM
            annotation = f"err={err:.3f}"

            ## En caso de haber resultados Wilcoxon disponibles,
            ## agrego el p-valor correspondiente a la feature actual
            if wilcoxon_results is not None:

                ## Obtengo el p-valor asociado a la feature actual
                p = df.iloc[i]["wilcoxon_p"]

                ## En caso de existir un valor válido de Wilcoxon
                if pd.notna(p):

                    ## Agrego la anotación correspondiente a la barra de la feature
                    annotation += f"\np={p:.2e}"

            ## Despliego la etiqueta encima de cada barra
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), annotation, ha = "center",
                va = "bottom", fontsize = 10)

    ## Ajusto automáticamente los márgenes del gráfico para evitar cortes
    plt.tight_layout()

    ## Despliego el gráfico final
    plt.show()

def plot_svm_univariate_confusion_matrices(predictions, feature_cols,
    target_values = None, save_dir = None, C = 1, gamma = "scale", cmap = "coolwarm"):
    """
    Genera matrices de confusión por feature a partir de predicciones de un SVM,
    con estilo visual tipo Wilcoxon.

    - Matrices de confusión en conteos absolutos (sin normalización)
    - Escala de color por feature (vmin = 0, vmax = máximo de cada matriz)
    - Anotaciones con valores absolutos en cada celda
    - Guardado opcional en formato PNG con timestamp
    - Estilo gráfico consistente para análisis comparativo entre features

    Parameters
    ----------
    predictions : dict
        Diccionario con predicciones por feature. Cada entrada debe contener:
        - "y_true": etiquetas reales
        - "y_pred": etiquetas predichas

    feature_cols : list of str
        Lista de features evaluadas.

    target_values : array-like or None, default=None
        Valores posibles de la variable objetivo. Si es None, se infieren
        desde la primera feature en `predictions`.

    save_dir : str or None, default=None
        Directorio donde se guardan las imágenes. Si es None, no se guardan archivos.

    C : float, default=1
        Parámetro de regularización del SVM (solo para anotación en el título).

    gamma : str or float, default="scale"
        Parámetro del kernel RBF del SVM (solo para anotación en el título).

    cmap : str, default="coolwarm"
        Colormap utilizado para visualizar la matriz de confusión.

    Returns
    -------
    None
        La función genera y guarda las figuras, pero no retorna valores.
    """

    ## Agrego configuraciones necesarias para que los gráficos se guarden y no se desplieguen
    ## en la pantall mientras está ejecutándose la función (se vuelve molesto sino)
    use("Agg")
    plt.ioff()

    ## En caso de que yo pase una ruta de guardado como entrada de la función
    if save_dir is not None:

        ## Configuro el instante de tiempo correspondiente al guardado
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        ## Construyo la ruta de guardado resultante
        save_dir = os.path.join(save_dir, timestamp)

        ## Creo la ruta de guardado en el directorio
        os.makedirs(save_dir, exist_ok = True)

    ## En caso de que yo no de valores objetivos como entrada de la función
    if target_values is None:

        ## Tomo arbitrariamente el primer feature de la lista a procesar
        any_feat = feature_cols[0]

        ## Configuro la lista de posibles valores objetivos
        target_values = np.unique(predictions[any_feat]["y_true"])

    ## Construyo una lista vacía en la cual voy a almacenar las matrices de confusión
    matrices = []

    ## Itero para cada una de las features que voy a analizar
    for feat in feature_cols:

        ## Selecciono la lista correspondiente a las asignaciones verdaderas de cada giro
        y_true = predictions[feat]["y_true"]

        ## Selecciono la lista correspondiente a las asignaciones predichas de cada giro
        y_pred = predictions[feat]["y_pred"]

        ## Construyo la matriz de confusión correspondiente para dicha feature
        cm = confusion_matrix(y_true, y_pred, labels = target_values)

        ## Agrego la matriz de confusión asociada a la feature a la lista correspondiente
        matrices.append(cm)

    ## Itero para cada una de las features con sus correspondientes matrices de confusión
    for feat, cm in zip(feature_cols, matrices):

        ## Configuro el tamaño del gráfico correspondiente
        plt.figure(figsize = (5, 4))

        ## Despliego la matriz de confusión en el gráfico correspondiente
        im = plt.imshow(cm, interpolation = "nearest", cmap = cmap, vmin = 0, vmax = cm.max())

        ## Configuro el título del gráfico
        plt.title(f"Matriz de Confusión SVM\n{feat} | C = {C}, gamma = {gamma}")

        ## Configuro las etiquetas horizontales de la matriz de confusión
        plt.xticks(range(len(target_values)), target_values)

        ## Configuro las etiquetas verticales de la matriz de confusión
        plt.yticks(range(len(target_values)), target_values)

        ## Configuro la leyenda horizontal de la matriz de confusión
        plt.xlabel("Predicho")

        ## Configuro la leyenda vertical de la matriz de confusión
        plt.ylabel("Real")

        ## Obtengo el valor máximo de la matriz de confusión asociada a la feature correspondiente
        vmax_feat = cm.max()

        ## Itero para cada una de las filas de la matriz de confusión
        for i in range(cm.shape[0]):

            ## Itero para cada una de las columnas de la matriz de confusión
            for j in range(cm.shape[1]):

                ## Obtengo el valor de la matriz de confusión en la i-ésima fila y j-ésima columna
                val = cm[i, j]

                ## Configuro el texto asociado a la entrada correspondiente de la matriz de confusión
                plt.text(j, i, f"{val}", ha = "center", va = "center",
                    color = "white" if val > vmax_feat * 0.5 else "black", fontsize = 9)

        ## Hago el ajuste de padding entre las subgráficas
        plt.tight_layout()

        ## Configuro el nombre del archivo de la matriz de confusión correspondiente
        filename = f"{feat}_SVM_C{C}_gamma{gamma}.png".replace(" ", "_")

        ## En caso de que yo haya pasado como argumento una ruta de entrada
        if save_dir is not None:

            ## Guardo la matriz de confusión como imagen en la ruta correspondiente
            plt.savefig(os.path.join(save_dir, filename), dpi = 300)

        ## Finalizo el procesamiento y cierro el gráfico de la matriz de confusión
        plt.close()

def plot_feature_space_2d(df, feature_x, feature_y, target_col, class_labels = None, 
    title = None, alpha = 0.6, figsize = (7, 6), save_path = None):
    """
    Proyecta el espacio de características en 2D y colorea los puntos según la clase.

    Esta función permite visualizar la relación entre dos features arbitrarias
    dentro del dataset, facilitando el análisis exploratorio de separabilidad
    entre clases en el espacio de características.

    Es especialmente útil para:
    - Validar visualmente resultados de selección de features (SFFS)
    - Analizar separabilidad entre grupos etarios
    - Detectar solapamiento entre clases
    - Evaluar estructuras no lineales antes de aplicar SVM

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las features y la variable objetivo.

    feature_x : str
        Nombre de la feature que se usará en el eje X.

    feature_y : str
        Nombre de la feature que se usará en el eje Y.

    target_col : str
        Nombre de la columna que contiene las etiquetas de clase.

    class_labels : dict or None, default=None
        Diccionario opcional para mapear las clases a etiquetas legibles.
        Ejemplo: {0: "≤75", 1: ">75"}

    title : str or None, default=None
        Título del gráfico. Si es None, se genera automáticamente.

    alpha : float, default=0.6
        Nivel de transparencia de los puntos.

    figsize : tuple, default=(7, 6)
        Tamaño de la figura.

    save_path : str or None, default=None
        Ruta donde se guarda la figura. Si es None, no se guarda.

    Returns
    -------
    None
        La función genera y muestra el gráfico, sin retornar valores.
    """

    ## Inicializo y configuro las dimensiones de la figura
    plt.figure(figsize = figsize)

    ## Obtengo un listado con todas las posibles clases a las que pertenecen los giros
    classes = df[target_col].unique()

    ## Itero para cada una de las clases
    for c in classes:

        ## Selecciono únicamente aquellos puntos correspondientes a la clase actual <<c>>
        subset = df[df[target_col] == c]

        ## Obtengo las etiquetas correspondientes a los puntos de la clase
        label = class_labels[c] if class_labels is not None else f"Clase {c}"

        ## Hago el diagrama de dispersión con los puntos correspondientes a la clase <<c>> dada
        plt.scatter(subset[feature_x], subset[feature_y], label = label, alpha = alpha)

    ## Configuro la nomenclatura del eje x
    plt.xlabel(feature_x)

    ## Configuro la nomenclatura del eje y
    plt.ylabel(feature_y)

    ## En caso de que no exista un título como entrada a la función
    if title is None:

        ## Construyo el título del gráfico correspondiente
        title = f"{feature_x} vs {feature_y} (coloreado por {target_col})"

    ## Despliego el título en el gráfico
    plt.title(title)

    ## Despliego las leyendas en el gráfico
    plt.legend()
    plt.tight_layout()

    ## En caso de que yo de una ruta de guardado de entrada
    if save_path is not None:

        ## Hago el guardado del gráfico en la ruta correspondiente
        plt.savefig(save_path, dpi = 300)

    ## Despliego el gráfico
    plt.show()

def plot_error_space(df, x_feat, y_feat, title = ""):
    """
    Visualiza el espacio de características en 2D diferenciando tipos de aciertos y errores de clasificación.

    Descripción
    -----------
    Esta función genera un gráfico de dispersión en el plano definido por dos variables
    (features), donde cada punto representa una observación del dataset. Los puntos se
    colorean y estilizan según el tipo de resultado obtenido por el clasificador, definido
    en la columna "error_type".

    Se distinguen cuatro categorías:
        - young_correct: muestras de la clase joven correctamente clasificadas.
        - old_correct: muestras de la clase mayor correctamente clasificadas.
        - young_to_old: muestras de la clase joven clasificadas erróneamente como mayores.
        - old_to_young: muestras de la clase mayor clasificadas erróneamente como jóvenes.

    Este tipo de visualización permite analizar la estructura espacial de los errores del
    modelo, identificar regiones de confusión y evaluar la dirección del sesgo de clasificación
    en el espacio de features. Además, facilita la comparación entre distintos subconjuntos
    de datos (por ejemplo, personas con y sin caídas).

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las features y la columna "error_type" previamente calculada.

    x_feat : str
        Nombre de la feature utilizada para el eje X.

    y_feat : str
        Nombre de la feature utilizada para el eje Y.

    title : str, default=""
        Título del gráfico.

    Retorna
    -------
    None
        La función genera y muestra un gráfico de dispersión, sin retornar valores.
    """

    ## Inicializo la figura correspondiente al diagrama de dispersión
    fig, ax = plt.subplots()

    ## Construyo un diccionario en el cual asocio cada tipo de resultado con un color específico
    styles = {"young_correct": {"color": "green", "marker": "o"}, "old_correct": {"color": "blue",
        "marker": "o"}, "young_to_old": {"color": "red", "marker": "x"}, "old_to_young": {
        "color": "orange", "marker": "x"}}

    ## Itero para cada uno de los tipos de error que tengo
    for label, style in styles.items():

        ## Construyo un subdataframe seleccionando únicamente las observaciones que tienen como
        ## resultado de clasificación el tipo de error actual
        sub = df[df["error_type"] == label]

        ## Grafico los puntos correspondientes a los objetos asociados con el tipo de error actual
        ax.scatter(sub[x_feat], sub[y_feat], c = style["color"], marker = style["marker"],
            label = label, alpha = 0.7)

    ## Configuro nomenclatura del eje de las abscisas del gráfico de dispersión
    ax.set_xlabel(x_feat)

    ## Configuro nomenclatura del eje de las ordenadas del gráfico de dispersión
    ax.set_ylabel(y_feat)

    ## Configuro el título del gráfico de dispersión
    ax.set_title(title)

    ## Despliego las leyendas del gráfico de dispersión
    ax.legend()

    ## Despliego el gráfico de dispersión
    plt.show()

def plot_turn_3x2(turn_signal, title = "Giro", normalize = False):
    """
    Visualiza un segmento de giro en una disposición 3x2.

    Esta función genera una figura con seis subgráficos organizados en una grilla
    de 2 filas por 3 columnas, permitiendo analizar simultáneamente las señales
    del giroscopio y del acelerómetro durante un giro detectado.

    Estructura de la visualización:
        - Fila 1: componentes del giroscopio (x, y, z)
        - Fila 2: componentes del acelerómetro (x, y, z)

    Parameters
    ----------
    turn_signal : dict
        Diccionario que contiene las señales del giro. Debe incluir:
            - "gyro": array (N x 3) con las componentes del giroscopio
            - "acc" : array (N x 3) con las componentes del acelerómetro

    title : str, opcional
        Título global de la figura. Por defecto es "Turn".

    normalize : bool, opcional
        Si es True, normaliza cada eje de las señales (media 0, desviación estándar 1)
        antes de la visualización. Esto facilita la comparación de la forma de las señales
        entre distintos giros, independientemente de su magnitud.

    Returns
    -------
    None
        La función no retorna valores; únicamente genera una figura en pantalla.
    """

    ## Obtengo el segmento de la señal de giroscopio correspondiente al giro actual
    gyro = turn_signal["gyro"]

    ## Obtengo el segmento de la señal de acelerómetro correspondiente al giro actual
    acc = turn_signal["acc"]

    ## En caso de que yo quiera hacer una normalización de las señales del segmento de giro
    if normalize:

        ## Hago la normalización por componente (z-score) del segmento de giroscopio asociada al giro 
        ## usando la media y desviación estándar
        gyro = (gyro - gyro.mean(axis = 0)) / (gyro.std(axis = 0) + 1e-8)

        ## Hago la normalización por compontente (z-score) del segmento de acelerómetro asociada al giro 
        ## usando la media y desviación estándar
        acc = (acc - acc.mean(axis = 0)) / (acc.std(axis = 0) + 1e-8)

    ## Construyo el vector de índices temporales en el cual voy a graficar los segmentos de las señales en el giro
    t = np.arange(len(gyro))

    ## Defino la grilla donde voy a desplegar los gráficos de los segmentos en cada uno de los ejes para cada sensor
    fig, axs = plt.subplots(2, 3, figsize = (14, 6), sharex = True)

    ## Grafico el segmento de la señal del giroscopio en el giro en el eje x
    axs[0, 0].plot(t, gyro[:, 0])
    axs[0, 0].set_title("Gyro X")
    axs[0, 0].grid()

    ## Grafico el segmento de la señal del giroscopio en el giro en el eje y
    axs[0, 1].plot(t, gyro[:, 1])
    axs[0, 1].set_title("Gyro Y")
    axs[0, 1].grid()

    ## Grafico el segmento de la señal del giroscopio en el giro en el eje z
    axs[0, 2].plot(t, gyro[:, 2])
    axs[0, 2].set_title("Gyro Z")
    axs[0, 2].grid()

    ## Grafico el segmento de la señal del acelerómetro en el giro en el eje x
    axs[1, 0].plot(t, acc[:, 0])
    axs[1, 0].set_title("Acc X")
    axs[1, 0].grid()

    ## Grafico el segmento de la señal del acelerómetro en el giro en el eje y
    axs[1, 1].plot(t, acc[:, 1])
    axs[1, 1].set_title("Acc Y")
    axs[1, 1].grid()

    ## Grafico el segmento de la señal del acelerómetro en el giro en el eje z
    axs[1, 2].plot(t, acc[:, 2])
    axs[1, 2].set_title("Acc Z")
    axs[1, 2].grid()

    ## Configuro el título global del conjunto de gráficos
    fig.suptitle(title, fontsize = 14)

    ## Despliego el gráfico correspondiente
    plt.tight_layout()
    plt.show()

def sample_and_plot_turns(df, condition, turns_map, title, n = 2, seed = 42, normalize = True):
    """
    Filtra, muestrea y visualiza giros representativos a partir de un conjunto de datos.

    Esta función aplica una condición lógica sobre un DataFrame de giros, selecciona
    una muestra aleatoria de los casos que cumplen dicha condición, recupera los
    segmentos de señal correspondientes desde un diccionario de referencia (`turns_map`)
    y genera visualizaciones comparativas de las señales de giroscopio y acelerómetro
    para cada giro seleccionado.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene los giros y sus características asociadas. Debe incluir
        al menos las columnas:
        - "id": identificador del sujeto
        - "start_idx": índice inicial del giro
        - "end_idx": índice final del giro

    condition : callable
        Función que recibe el DataFrame `df` y devuelve una máscara booleana
        para filtrar los giros de interés.

    turns_map : dict
        Diccionario que mapea cada giro mediante la clave:
        (id, start_idx, end_idx)
        hacia un diccionario con las señales asociadas, típicamente:
        - "gyro": señal del giroscopio (N x 3)
        - "acc" : señal del acelerómetro (N x 3)

    title : str
        Título base utilizado en las figuras generadas.

    n : int, opcional
        Número máximo de giros a muestrear y visualizar. Por defecto es 2.

    seed : int, opcional
        Semilla aleatoria utilizada para garantizar reproducibilidad del muestreo.

    normalize : bool, opcional
        Si es True, aplica normalización tipo z-score por componente a las señales
        antes de la visualización, facilitando la comparación de la forma de los giros
        independientemente de su magnitud.

    Returns
    -------
    None
        La función no retorna valores. Su salida consiste en la generación de figuras
        para cada giro seleccionado.
    """

    ## En base al conjunto de todos los giros, hago el filtrado de los giros que cumplen
    ## con la condición que yo paso como parámetro
    candidates = df[condition(df)]

    ## Hago un muestreo aleatorio de hasta n giros del conjunto de candidatos
    sampled = candidates.sample(n = min(n, len(candidates)), random_state = seed)

    ## Itero para cada uno de los giros muestreados
    for _, row in sampled.iterrows():

        ## Obtengo la clave única asociada a dicho giro, la cual está compuesta por el ID de la persona
        ## a la que pertenece, el índice de muestra de comienzo de giro, y el índice de muestra de fin de giro
        key = (int(row["id"]), row["start_idx"], row["end_idx"])

        ## Obtengo (en caso de existir) los segmentos de las señales de acelerómetro y giroscopio asociados a dicho giro
        turn = turns_map.get(key)

        ## En caso de que el giro exista en el diccionario
        if turn is not None:

            ## Grafico los segmentos de señales de acelerómetro y giroscopio del giro
            plot_turn_3x2(turn, title = title, normalize = normalize)

def plot_id_distribution(df, group_col, id_col = "id", normalize = False, title = None,
                        figsize = (10, 5), save_path = None):
    """
    Visualiza la distribución de IDs de pacientes por categoría de agrupamiento
    utilizando un estilo de barras explícito (tipo histograma categórico).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con los datos.

    group_col : str
        Columna de agrupamiento.

    id_col : str
        Columna de identificador de paciente.

    normalize : bool
        Si True, muestra porcentajes en lugar de conteos.

    title : str or None
        Título del gráfico.

    figsize : tuple
        Tamaño de la figura.

    save_path : str or None
        Ruta opcional para guardar la figura.

    Returns
    -------
    None
    """

    ## Defino el conjunto de columnas requeridas para el análisis
    required = {id_col, group_col}

    ## Verifico qué columnas están ausentes en el DataFrame
    missing = required - set(df.columns)

    ## En caso de faltar columnas necesarias, lanzo un error explícito
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    ## Calculo la distribución de frecuencias por grupo
    conteos = df[group_col].value_counts().sort_index()

    ## En caso de activar normalización, convierto los conteos a porcentaje
    if normalize:
        conteos = conteos / conteos.sum() * 100

    ## Creo la figura con el tamaño especificado
    plt.figure(figsize = figsize)

    ## Dibujo barras tipo histograma categórico para cada grupo
    plt.bar(conteos.index.astype(str), conteos.values, edgecolor = "black")

    ## Configuro el título del gráfico
    if title is None:
        title = f"Distribución de IDs por {group_col}"
    plt.title(title)

    ## Configuro etiqueta del eje X (variable de agrupamiento)
    plt.xlabel(group_col)

    ## Configuro etiqueta del eje Y según si está normalizado o no
    plt.ylabel("Porcentaje (%)" if normalize else "Número de muestras")

    ## Activo grilla horizontal para facilitar lectura visual
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    ## Agrego etiquetas numéricas sobre cada barra
    for i, v in enumerate(conteos.values):

        ## Formateo el valor según sea porcentaje o conteo absoluto
        label = f"{v:.1f}%" if normalize else str(int(v))

        ## Dibujo el texto centrado encima de cada barra
        plt.text(i, v, label, ha = "center", va = "bottom")

    ## Ajusto automáticamente el layout para evitar solapamientos
    plt.tight_layout()

    ## Si se especifica una ruta, guardo la figura en disco
    if save_path is not None:
        plt.savefig(save_path, dpi = 300)

    ## Muestro la figura final
    plt.show()

def plot_pca_loading_stacked(load_df, K = 10, title = "Distribución de loadings del PCA"):
    """
    Visualiza la distribución de contribuciones de las features originales
    en los primeros K componentes principales mediante un gráfico de barras apiladas.

    Este gráfico permite analizar cómo se distribuye la influencia de cada feature
    del espacio original a lo largo de las direcciones principales del PCA.

    Cada barra representa una feature, y está descompuesta en segmentos
    correspondientes a los K primeros componentes principales.

    Parameters
    ----------
    load_df : pandas.DataFrame
        DataFrame en formato largo generado por `pc_loading_distributions`, con columnas:
            - feature : nombre de la feature original
            - loading : contribución normalizada
            - pc      : índice del componente principal

    K : int, optional (default=10)
        Número de componentes principales a visualizar.

    title : str, optional
        Título del gráfico.

    Returns
    -------
    None
        La función genera una visualización y no retorna valores.
    """

    ## Filtro únicamente los primeros K componentes principales
    df = load_df[load_df["pc"] < K].copy()

    ## Reorganizo el DataFrame a formato matricial:
    ## filas = features originales
    ## columnas = componentes principales (PCs)
    ## valores = contribuciones normalizadas (loadings)
    pivot = df.pivot(index = "feature", columns = "pc", values = "loading").fillna(0)

    ## Ordeno las features según su contribución total acumulada
    ## (esto facilita la interpretación visual priorizando features más relevantes)
    pivot = pivot.loc[pivot.sum(axis = 1).sort_values(ascending = False).index]

    ## Creo la figura y los ejes del gráfico
    fig, ax = plt.subplots(figsize = (14, 6))

    ## Inicializo la base acumulada para el gráfico de barras apiladas
    bottom = None

    ## Itero sobre los primeros K componentes principales
    for k in range(K):

        ## Extraigo los valores de loading correspondientes al PC actual
        values = pivot[k].values

        ## Dibujo la barra correspondiente a este componente principal
        ## Si 'bottom' es None, dibujo desde cero
        ax.bar(pivot.index, values, bottom = bottom, label = f"PC{k+1}")

        ## Inicializo o actualizo la base acumulada para el apilamiento
        if bottom is None:
            bottom = values.copy()
        else:
            bottom += values

    ## Configuro el título del gráfico
    ax.set_title(title)

    ## Configuro la etiqueta del eje Y (contribución normalizada)
    ax.set_ylabel("Contribución normalizada del loading")

    ## Configuro la etiqueta del eje X (features originales)
    ax.set_xlabel("Features")

    ## Roto las etiquetas del eje X para mejorar la legibilidad
    ax.tick_params(axis = 'x', rotation = 90)

    ## Agrego la leyenda indicando cada componente principal
    ax.legend(ncol = 2, fontsize = 9)

    ## Ajusto el layout para evitar solapamientos
    plt.tight_layout()

    ## Muestro el gráfico
    plt.show()

def plot_pca_feature_contributions(load_df, K = 10, N = 10,
        title = "Contribución de features por componente principal"):
    """
    Visualiza la distribución de contribuciones de las features originales
    en los primeros K componentes principales del PCA.

    Para cada componente principal, se seleccionan únicamente las N features
    con mayor contribución (loading normalizado), y se representa su magnitud
    mediante un heatmap con anotaciones numéricas.

    Este gráfico permite interpretar directamente qué variables dominan cada
    dirección principal del espacio PCA.

    Parameters
    ----------
    load_df : pandas.DataFrame
        DataFrame en formato largo generado por `pc_loading_distributions`,
        con columnas:
            - feature
            - loading
            - pc

    K : int
        Número de componentes principales a analizar.

    N : int
        Número de features más relevantes a mostrar por cada componente.

    title : str
        Título del gráfico.

    Returns
    -------
    None
    """

    ## Filtro únicamente los primeros K componentes principales del análisis PCA
    df = load_df[load_df["pc"] < K].copy()

    ## Ordeno los datos por componente principal y por magnitud de contribución
    ## para asegurar que las features más relevantes aparezcan primero
    df = df.sort_values(["pc", "loading"], ascending = [True, False])

    ## Selecciono únicamente las N features más importantes dentro de cada PC
    df = df.groupby("pc").head(N)

    ## Reorganizo los datos a formato matricial donde:
    ## - filas representan componentes principales (PCs)
    ## - columnas representan features originales
    ## - valores representan loadings normalizados
    pivot = df.pivot(index = "pc", columns = "feature", values = "loading").fillna(0)

    ## Ordeno los componentes principales de menor a mayor (PC1 → PCK)
    pivot = pivot.sort_index()

    ## Inicializo la figura y el eje donde voy a representar el heatmap
    fig, ax = plt.subplots(figsize = (14, 6))

    ## Represento la matriz de loadings como un heatmap
    im = ax.imshow(pivot.values, aspect = "auto")

    ## Configuro las posiciones de los ticks del eje Y según el número de PCs
    ax.set_yticks(np.arange(len(pivot.index)))

    ## Etiqueto cada fila como PC1, PC2, ..., PCK para facilitar interpretación
    ax.set_yticklabels([f"PC{k+1}" for k in pivot.index])

    ## Configuro las posiciones de los ticks del eje X según el número de features
    ax.set_xticks(np.arange(len(pivot.columns)))

    ## Etiqueto cada columna con el nombre de la feature original
    ax.set_xticklabels(pivot.columns, rotation = 90)

    ## Asigno el título del gráfico
    ax.set_title(title)

    ## Recorro todas las celdas del heatmap para añadir valores numéricos
    ## Itero para cada una de los componentes principales PCs (filas)
    for i in range(pivot.shape[0]):

        ## Itero para cada una de las features (columnas)
        for j in range(pivot.shape[1]):

            ## Obtengo el valor de loading normalizado en la celda actual
            val = pivot.values[i, j]

            ## Solo muestro valores relevantes para evitar saturación visual
            if val > 1e-3:

                ## Escribo el valor dentro de la celda correspondiente
                ax.text(j, i, f"{val:.2g}", ha = "center", va = "center", fontsize = 7, color = "black")

    ## Agrego barra de color para interpretar magnitudes globales
    plt.colorbar(im, ax = ax, label = "Loading normalizado")

    ## Ajusto el layout para evitar solapamiento de etiquetas
    plt.tight_layout()

    ## Despliego el gráfico final
    plt.show()

def plot_regression_dependencies(results_df, top_n = 20, 
                    title = "Ranking de dependencias lineales entre features"):
    """
    Visualiza las relaciones de dependencia lineal más fuertes entre pares
    de features según el coeficiente de determinación R².

    La función genera un gráfico de barras horizontal con las combinaciones
    de features que presentan mayor capacidad de explicación lineal.

    Cada barra representa una regresión entre:
        - x_feature: feature utilizada como variable predictora.
        - feature: feature explicada por el modelo.

    Además del valor de R² se indica el valor de ajuste de cada regresión,
    permitiendo identificar las relaciones feature-feature con mayor
    dependencia lineal.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame con los resultados del análisis de regresión.
        Debe contener:
            - x_feature: feature independiente.
            - feature: feature dependiente.
            - linear_r2: coeficiente de determinación del modelo lineal.
            - slope: pendiente de la regresión.

    top_n : int, default=20
        Número de relaciones feature-feature a visualizar.

    title : str, default="Ranking de dependencias lineales entre features"
        Título mostrado en la figura generada.

    Returns
    -------
    None
        La función genera y muestra un gráfico, sin retornar valores.

    Notes
    -----
    - Un mayor valor de R² indica una relación lineal más fuerte entre
    ambas features.
    - El análisis permite identificar posibles relaciones biomecánicas
    entre variables cinemáticas dentro del grupo evaluado.
    """

    ## Ordeno las regresiones según el valor de R² lineal y selecciono únicamente las relaciones
    ## con mayor dependencia explicativa
    df = (results_df.sort_values("linear_r2", ascending = False).head(top_n).copy())

    ## Construyo una etiqueta combinando la feature predictora y la feature
    ## dependiente para identificar cada relación en el eje vertical
    df["pair"] = df["x_feature"] + " → " + df["feature"]

    ## Configuro el tamaño de la figura y obtengo el eje activo para agregar anotaciones
    plt.figure(figsize = (10, 6))
    ax = plt.gca()

    ## Genero el gráfico de barras horizontales donde cada barra representa
    ## el valor de R² correspondiente a una relación feature-feature
    sns.barplot(data = df, y = "pair", x = "linear_r2", ax = ax)

    ## Agrego el valor numérico de R² al final de cada barra
    for bar, r2 in zip(ax.patches, df["linear_r2"]):

        ## Obtengo la posición final de cada barra para ubicar la etiqueta
        width = bar.get_width()

        ## Escribo el valor de R² con tres cifras decimales
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{r2:.3f}", va = "center",
                fontsize = 10)

    ## Agrego información descriptiva del gráfico utilizando el título recibido como parámetro
    plt.title(title)
    plt.xlabel("R² regresión lineal")
    plt.ylabel("Relación entre features")

    ## Agrego una grilla horizontal para facilitar la comparación visual
    plt.grid(axis = "x", alpha = 0.3)

    ## Ajusto automáticamente los márgenes para evitar solapamiento de etiquetas
    plt.tight_layout()

    ## Muestro el gráfico generado
    plt.show()