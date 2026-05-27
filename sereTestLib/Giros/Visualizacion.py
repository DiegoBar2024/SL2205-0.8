import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib import use

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

            ## Asigno val == 1 en caso de que la diferencia detectada por el test de Wilcoxon sea significativa
            val = 1 if is_sig else 0

            ## Dado que el resultado del test es simétrico (no depende del orden de los grupos etarios)
            ## configuro el mismo valor de resultado de significación del test en las entradas de ambos grupos
            mat.loc[g1, g2] = val
            mat.loc[g2, g1] = val

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
                linecolor = "black", square = True, ax = ax)

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