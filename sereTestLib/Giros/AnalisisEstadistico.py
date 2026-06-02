import pandas as pd
import numpy as np
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.metrics import accuracy_score

def kruskal_wallis_features(df, feature_cols, group_col = "age_group"):
    """
    Aplica el test de Kruskal-Wallis a múltiples features para evaluar
    diferencias entre grupos.

    Además calcula tamaño de efecto (epsilon squared) y corrección FDR.

    Parámetros
    ----------
    df : pandas.DataFrame
        Dataset con features a nivel de evento (ej. giro) y columna de grupo.

    feature_cols : list of str
        Lista de columnas numéricas (features) a evaluar.

    group_col : str, opcional (default="age_group")
        Columna que define los grupos (ej: 0, 1, 2).

    Retorna
    -------
    pandas.DataFrame
        Resultados por feature con:
        - feature
        - H : estadístico Kruskal-Wallis
        - p_value : p-value original
        - epsilon_sq : tamaño de efecto
        - p_fdr : p-value corregido (FDR)
        - significant_fdr : significancia tras FDR
    """

    ## Inicializo una lista vacía en la cual voy a almacenar los resultados de Kruskal Wallis
    results = []

    ## Construyo una lista conteniendo únicamente los índices de los grupos etarios
    groups = df[group_col].dropna().unique()

    ## Ordeno los índices de los grupos según su magnitud
    groups = np.sort(groups)

    ## Itero para cada una de las columnas de las features que tengo
    for feat in feature_cols:

        ## Inicializo un vector vacío en donde voy guardando los valores de las features de cada uno
        ## de los grupos etarios que tengo
        samples = []

        ## Itero para cada uno de los grupos etarios que tengo
        for g in groups:

            ## Selecciono la lista con los valores de la feature para el grupo etario correspondiente
            values = df[df[group_col] == g][feat].dropna().values

            ## En caso de que tenga más de un valor en la lista
            if len(values) > 0:

                ## Almaceno los valores del feature del grupo etario en la lista correspondiente
                samples.append(values)

        ## Para evitar tests inválidos, en caso que tenga menos de dos muestras
        if len(samples) < 2:

            ## Continúo con la siguiente feature y no ejecuto el test de hipótesis
            continue

        ## Aplico el test de hipótesis de Kruskal Wallis para todos los valores numéricos de la feature
        ## separados en cada uno de los tres grupos etarios
        H, p = kruskal(*samples)

        ## Obtengo la cantidad total de features (cantidad total de eventos de giro detectados)
        N = sum(len(s) for s in samples)

        ## Obtengo la cantidad total de grupos etarios sobre los cuales estoy dividiendo
        k = len(samples)

        ## Hago el cálculo del Epsilon Cuadrado para Kruskal Wallis: ε² = (H - k + 1) / (N - k)
        eps_sq = (H - k + 1) / (N - k + 1e-12)

        ## Para la feature correspondiente, construyo un diccionario de resultados del test de hipótesis
        ## Kruskal Wallis incluyendo el nombre de la feature, el valor del estadístico de prueba del test
        ## dado por H, el p-valor y el valor de ε²
        results.append({"feature": feat, "H": H, "p_value": p, "epsilon_sq": eps_sq})

    ## Hago la conversión de la lista de diccionarios con los resultados de Kruskal Wallis a dataframe
    results_df = pd.DataFrame(results)

    ## En caso de que haya realizado más de un test de hipótesis
    if len(results_df) > 0:

        ## Dado que estoy haciendo múltiples tests de hipótesis con muchas features, hago el control
        ## de la tasa de falsos positivos (FDR: False Discovery Rate)
        reject, p_fdr, _, _ = multipletests(results_df["p_value"].values, method = "fdr_bh")

        ## Agrego los resultados del FDR al dataframe
        results_df["p_fdr"] = p_fdr

        ## Agrego los resultados de la significación del FDR al dataframe
        results_df["significant_fdr"] = reject

    ## Hago el ordenamiento de los resultados de modo que dejo más arriba los que más me contradicen el
    ## hecho de que todas las distribuciones son iguales (o sea, los que tienen mayor H)
    results_df = results_df.sort_values("epsilon_sq", ascending = False).reset_index(drop = True)

    ## Retorno el dataframe conteniendo todos los resultados de los tests de Kruskal Wallis
    return results_df

def evaluar_clustering_por_edad(clustering_output, group_col = "age_group"):
    """
    Evalúa qué tan bien un clustering no supervisado (K-Means) se alinea con los grupos etarios conocidos.

    Description
    -----------
    Esta función toma la salida de `aplicar_clustering_giros()` y cuantifica qué tan informativos
    son los clusters aprendidos con respecto a las verdaderas etiquetas de grupo etario.

    La idea es medir si los clusters descubiertos de manera no supervisada separan naturalmente
    los grupos etarios, calculando un mapeo desde clusters hacia el grupo etario mayoritario
    y estimando el error de clasificación.

    Esto puede interpretarse como un proxy de la calidad de las features: si las features son informativas,
    los clusters deberían alinearse con la estructura etaria.

    Parameters
    ----------
    clustering_output : dict
        Diccionario de salida de `aplicar_clustering_giros()`. Debe contener:
        - df : pandas.DataFrame con una columna 'cluster' agregada

    group_col : str, default="age_group"
        Columna del dataframe que contiene las etiquetas reales de grupo etario.

    Returns
    -------
    dict
        Diccionario que contiene:

        accuracy : float
            Exactitud del mapeo clusters → grupo etario mayoritario.

        error_rate : float
            1 - accuracy (proxy de la probabilidad de mala clasificación).

        cluster_to_group : dict
            Mapeo de ID de cluster → grupo etario más frecuente.

        confusion_matrix : pandas.DataFrame
            Tabla de contingencia entre clusters y grupos etarios reales.

        cluster_purity : pandas.DataFrame
            Pureza de cada cluster (fracción de la clase dominante dentro del cluster).
    """

    ## Hago una copia en formato dataframe pandas de los resultados del clustering
    df = clustering_output["df"].copy()

    ## Elimino aquellas filas que no tengan etiquetas
    df = df.dropna(subset = [group_col, "cluster"])

    ## Hago el mapeo entre los clusters y los grupos etarios usando voto por mayoría
    cluster_to_group = (df.groupby("cluster")[group_col].agg(lambda x: x.mode().iloc[0]).to_dict())

    ## Obtengo las etiquetas predichas basadas en la asignación de clústers
    df["pred_group"] = df["cluster"].map(cluster_to_group)

    ## Obtengo una medida de la precisión de la relación entre los clústers y los grupos etarios
    acc = accuracy_score(df[group_col], df["pred_group"])

    ## Obtengo una medida del error de la relación entre los clústers y los grupos etarios
    error = 1 - acc

    ## Hago el cálculo de la matriz de confusión correspondiente a la relación de clústers y grupos etarios
    cm = pd.crosstab(df["cluster"], df[group_col])

    ## Retorno un diccionario con todos los resultados estadísticos del agrupamiento entre clusters
    ## y los grupos etarios dados
    return {"accuracy": acc, "error_rate": error, "cluster_to_group": cluster_to_group,
        "confusion_matrix": cm}

def aplicar_clustering_giros(df, feature_cols, k_range = range(2, 7), random_state = 42):
    """
    Aplica clustering no supervisado sobre un conjunto de giros representados
    por features cinemáticas y estadísticas.

    El objetivo es descubrir estructuras latentes en el espacio de características
    de los giros (turn-level analysis), sin utilizar información de edad u otras etiquetas.

    El modelo utiliza K-Means con selección automática de K basada en silhouette score.

    Parámetros
    ----------
    df : pandas.DataFrame
        Dataset de giros ya construido en el pipeline principal.
        Cada fila representa un giro individual.

        Debe contener:
        - features numéricas definidas en `feature_cols`
        - opcionalmente columnas de identificación (ej. id)

    feature_cols : list of str
        Lista de columnas utilizadas como espacio de características para clustering.

    k_range : iterable, default=range(2, 7)
        Rango de valores de K evaluados mediante silhouette score.

    random_state : int, default=42
        Semilla para reproducibilidad del algoritmo K-Means.

    Returns
    -------
    dict
        Diccionario con los siguientes elementos:

        df : pandas.DataFrame
            Dataset original con una nueva columna:
            - cluster: asignación de cluster por giro

        best_k : int
            Número óptimo de clusters seleccionado mediante silhouette score.

        kmeans : sklearn.cluster.KMeans
            Modelo K-Means entrenado con el mejor K.

        scaler : sklearn.preprocessing.StandardScaler
            Escalador ajustado a los datos originales.

        centroids : pandas.DataFrame
            Centroides de los clusters en el espacio original de features.

            Incluye:
            - una fila por cluster
            - columnas = features originales

        cluster_summary : pandas.DataFrame
            Estadísticas descriptivas por cluster:
            - mean
            - std
            - median
            para cada feature

        cluster_counts : dict
            Diccionario con el número de giros en cada cluster.
            Formato: {cluster_id: count}
    """

    ## Obtengo la matriz de datos con el conjunto de todas las features de todos los giros
    ## La i-ésima fila hace referencia al i-ésimo giro
    ## La j-ésima columna hace referencia al j-ésimo giro
    X = df[feature_cols].values

    ## Instancio un objeto escalador de la clase StandardScaler()
    scaler = StandardScaler()

    ## Hago la estandarización de la matriz de datos como etapa previa al clústering
    X_scaled = scaler.fit_transform(X)

    ## Inicializo una lista vacía en donde voy a guardar los silhouette scores asociados a cada
    ## una de las ejecuciones de los clusterings (en el rango deseado)
    sil_scores = []

    ## Itero para cada uno de los clústerings en el rango deseado (en la k-ésima iteración se
    ## ejecuta K-Means especificando un total de k clústers como objetivo)
    for k in k_range:

        ## Instancio un modelo de K-Means especificando la cantidad de clústers deseada
        model = KMeans(n_clusters = k, random_state = random_state, n_init = 10)

        ## Ejecuto K-Means sobre la matriz de datos estandarizada y me quedo con las asignaciones
        labels = model.fit_predict(X_scaled)

        ## Agrego la silhouette score correspondiente al clustering a la lista correspondiente
        sil_scores.append(silhouette_score(X_scaled, labels))

    ## Obtengo la cantidad de clústers óptima dentro del rango especificado como entrada, tomando como
    ## 'óptimo' aquella cantidad de clústers la cual me maximice el silhouette score
    best_k = list(k_range)[np.argmax(sil_scores)]

    ## Instancio un modelo de clustering tomando la cantidad de clústers óptima en base al análisis de
    ## la maximización de silohouette score
    kmeans = KMeans(n_clusters = best_k, random_state = random_state, n_init = 10)

    ## Ejecuto el clustering y agrego al dataframe de features el índice del clúster al que pertenece
    ## cada feature vector de cada giro
    df["cluster"] = kmeans.fit_predict(X_scaled)

    ## Obtengo los centroides correspondientes a cada uno de los clústers
    centroids_scaled = kmeans.cluster_centers_

    ## Invierto la estandarización para expresar los centroides en términos de las features originales
    centroids_original = scaler.inverse_transform(centroids_scaled)

    ## Construyo un dataframe que contenga los centroides y los nombres de las features, y expreso las
    ## coordenadas de los centroides en términos de las features correspondientes
    centroids = pd.DataFrame(centroids_original, columns = feature_cols)

    ## Agrego una columna asignando el centroide al índice del clúster al que corresponde
    centroids["cluster"] = range(best_k)

    ## Obtengo un resumen estadístico de cada una de las features en cada uno de los clusters
    cluster_summary = df.groupby("cluster")[feature_cols].agg(["mean", "std", "median"])

    ## Hago el aplanamiento de las columnas y configuro los nombres correspondientes
    cluster_summary.columns = [f"{feat}_{stat}" for feat, stat in cluster_summary.columns]

    ## Hago el reset del índice correspondiente al dataframe del resumen estadístico de los clústers
    cluster_summary = cluster_summary.reset_index()

    ## Obtengo la cantidad de puntos correspondientes a los diferentes clusters
    cluster_counts = df["cluster"].value_counts().sort_index().to_dict()

    ## Retorno los resultados de hacer el clustering en el dataframe de giros
    return {"df": df, "best_k": best_k, "kmeans": kmeans, "scaler": scaler,
        "centroids": centroids, "cluster_summary": cluster_summary, "cluster_counts": cluster_counts}

def pairwise_wilcoxon_rank_sum(df, feature_cols, group_col = "age_group"):
    """
    Realiza comparaciones estadísticas dos a dos entre grupos etarios utilizando el test de Mann–Whitney U
    (Wilcoxon rank-sum) para evaluar diferencias entre distribuciones de features.

    El test contrasta la hipótesis nula de igualdad de distribuciones entre dos muestras independientes,
    siendo especialmente sensible a diferencias en la localización (mediana/rangos).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset que contiene las features y la variable de agrupamiento.

    feature_cols : list of str
        Lista de columnas numéricas (features) a evaluar.

    group_col : str, default="age_group"
        Nombre de la columna que define los grupos etarios.

    Returns
    -------
    pandas.DataFrame
        Tabla con los resultados por feature y par de grupos, incluyendo:
        - feature: nombre de la variable evaluada
        - group_1, group_2: grupos comparados
        - U: estadístico del test Mann–Whitney U
        - p_value: valor p original
        - p_fdr: valor p corregido por FDR (Benjamini–Hochberg)
        - rank_biserial_corr: tamaño de efecto (correlación biserial por rangos)
        - n1, n2: tamaños muestrales de cada grupo
    """

    ## Construyo una lista conteniendo únicamente los índices de los grupos etarios
    groups = sorted(df[group_col].dropna().unique())

    ## Obtengo una lista de todos los posibles pares de grupos etarios que tengo
    pairs = list(combinations(groups, 2))

    ## Inicializo una lista vacía en la cual voy a almacenar los resultados de Kruskal Wallis
    results = []

    ## Itero para cada una de las columnas de features que voy a analizar
    for feat in feature_cols:

        ## Selecciono g1 y g2 como aquel par de grupos al que corresponde esta iteración 
        for g1, g2 in pairs:

            ## Para la feature estudiada, selecciono la lista de todos los valores estadísticos de dicha feature
            ## para el grupo etario g1
            x1 = df[df[group_col] == g1][feat].dropna().values

            ## Para la feature estudiada, selecciono la lista de todos los valores estadísticos de dicha feature
            ## para el grupo etario g2
            x2 = df[df[group_col] == g2][feat].dropna().values

            ## En caso que la cantidad de valores de la feature sea menor a 2 para cualquier grupo
            if len(x1) < 2 or len(x2) < 2:

                ## Continúo ejecutando con la siguiente feature
                continue

            ## Ejecuto el test de suma de rangos de Wilcoxon para los conjuntos de valores de la feature
            ## estudiada correspondientes a los grupos etarios g1 y g2
            U, p = mannwhitneyu(x1, x2, alternative = "two-sided")

            ## Obtengo la cantidad de valores de la feature estudiada para los grupos g1 y g2
            n1, n2 = len(x1), len(x2)

            ## Hago el cálculo de la correlación biserial de rangos
            rank_biserial = 1 - (2 * U) / (n1 * n2)

            ## Construyo un diccionario correspondientes a todos los tests de hipótesis
            results.append({"feature": feat, "group_1": g1, "group_2": g2, "U": U, "p_value": p,
                "rank_biserial_corr": rank_biserial, "n1": n1, "n2": n2})

    ## Hago la conversión de los resultados a formato dataframe
    results_df = pd.DataFrame(results)

    ## En caso de que haya realizado más de un test de hipótesis
    if len(results_df) > 0:
        
        ## Dado que estoy haciendo múltiples tests de hipótesis con muchas features, hago el control
        ## de la tasa de falsos positivos (FDR: False Discovery Rate)
        reject, p_fdr, _, _ = multipletests(results_df["p_value"].values, method = "fdr_bh")

        ## Agrego los resultados del FDR al dataframe
        results_df["p_fdr"] = p_fdr

        ## Agrego los resultados de la significación del FDR al dataframe
        results_df["significant_fdr"] = reject

    ## Ordeno los resultados según el valor del FDR y el rango de correlación biserial
    results_df = results_df.sort_values(by = ["p_fdr", "rank_biserial_corr"], ascending = [True, False]
                                        ).reset_index(drop = True)

    ## Retorno el dataframe correspondiente con los resultados
    return results_df

def wilcoxon_pairwise_matrices(df_results, alpha = 0.05, use_fdr = False, use_significance = True):
    """
    Convierte resultados de tests Wilcoxon por pares en matrices por feature.

    Description
    -----------
    Toma la salida ya procesada del pipeline (pairwise Wilcoxon rank-sum)
    y la reorganiza en matrices simétricas grupo × grupo para cada feature.

    Permite usar p-values crudos o corregidos por FDR, y opcionalmente
    transformar los resultados en decisiones de significancia.

    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame con columnas:
            - feature
            - group_1
            - group_2
            - p_value
            - p_fdr (opcional)
            - significant_fdr (opcional)

    alpha : float
        Nivel de significación para decidir rechazo de H0.

    use_fdr : bool
        Si True, usa p_fdr en lugar de p_value.

    use_significance : bool
        Si True, devuelve matriz booleana (rechazo H0).
        Si False, devuelve p-values.

    Returns
    -------
    dict
        Diccionario:
            key   = feature
            value = DataFrame (matriz grupo × grupo)
    """

    ## Hago la copia del dataframe que paso a la entrada
    df = df_results.copy()

    ## En caso de que quiera usar el parámetro FDR hago las configuraciones necesarias
    p_col = "p_fdr" if use_fdr and "p_fdr" in df.columns else "p_value"

    ## Obtengo el conjunto de todas las features de los giros que extraje
    features = df["feature"].unique()

    ## Obtengo la lista con los índices correspondientes a todos los grupos etarios
    groups = sorted(set(df["group_1"]).union(set(df["group_2"])))

    ## Inicializo un diccionario vacío donde voy a guardar las matrices con las métricas
    matrices = {}

    ## Itero para cada una de las features que tengo
    for feat in features:

        ## Selecciono únicamente aquellos resultados que correspondan al feature <<feat>> actual
        sub = df[df["feature"] == feat]

        ## Inicializo la matriz cuyas filas y columnas están dadas por los índices de los grupos etarios
        mat = pd.DataFrame(index = groups, columns = groups, dtype = object)

        ## Itero para cada uno de los índices de grupos etarios que tengo
        for g in groups:

            ## Configuro los elementos de la diagonal principal de la matriz de resultados como NaN, para
            ## indicar que no tiene sentido testear un conjunto contra sí mismo
            mat.loc[g, g] = np.nan

        ## Itero para cada uno de los resultados del test de hipótesis para la feature <<feat>> en cuestión
        for _, row in sub.iterrows():

            ## Obtengo el índice correspondiente al primer grupo etario a analizar
            g1 = row["group_1"]

            ## Obtengo el índice correspondiente al segundo grupo etario a analizar
            g2 = row["group_2"]

            ## Obtengo el p-valor correspondiente al test de Wilcoxon de suma de rangos para la feature
            ## <<feat>> estudiada entre los grupos etarios g1 y g2
            p = row[p_col]

            ## En caso de que quiera usar el nivel de significación para dar información de resultados
            if use_significance:

                ## Asigno en <<value>> el resultado del test de hipótesis de Wilcoxon. Recuerdo que si
                ## p-valor < alpha entonces rechazo la hipótesis nula H0 al nivel de significacion alpha
                value = (p <= alpha)
            
            ## En caso de que no quiera usar el nivel de significación para representar el resultado
            else:

                ## Escribo el p-valor correspondiente en la entrada de la matriz
                value = p

            ## Asigno el resultado de rechazar H0 o el p-valor según corresponda en las componentes
            ## simétricas de la matriz de resultados para los grupos g1 y g2
            mat.loc[g1, g2] = value
            mat.loc[g2, g1] = value

        ## Asigno al diccionario la matriz de resultados correspondiente a la feature <<feat>>
        matrices[feat] = mat

    ## Retorno el diccionario conteniendo las matrices de resultados del test de Wilcoxon para cada feature
    return matrices

def regression_analysis(df, feature_cols, x_col = "age", poly_degree = 2, alpha = 0.05):
    """
    Analiza la relación entre edad y features mediante regresión lineal,
    regresión polinómica e inferencia estadística.

    Para cada feature se ajustan dos modelos:
        1. Regresión lineal (sklearn)
        2. Regresión polinómica (sklearn)

    Además, se evalúa la significancia estadística del coeficiente lineal
    de la edad mediante un modelo OLS.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset que contiene la variable independiente y las features.
    feature_cols : list of str
        Columnas a analizar como variables dependientes.
    x_col : str, default="age"
        Variable independiente (edad).
    poly_degree : int, default=2
        Grado del modelo polinómico.
    alpha : float, default=0.05
        Nivel de significación para el test del coeficiente lineal.

    Returns
    -------
    pandas.DataFrame
        Resultados por feature con:

        - feature: nombre de la variable analizada
        - slope: coeficiente lineal (edad)
        - intercept: término independiente
        - linear_r2: R² del modelo lineal
        - poly_r2: R² del modelo polinómico
        - p_value: p-valor del coeficiente de edad (OLS)
        - significant: True si p_value < alpha
    """

    ## Construyo una lista vacía en la cual yo voy a almacenar los resultados de la regresión
    results = []

    ## Obtengo el conjunto de valores de la edad (variable independiente en el análisis de regresión)
    X = df[[x_col]].values

    ## Construyo la matriz de diseño para el modelo OLS, incluyendo el término constante (intercept)
    X_ols = sm.add_constant(df[[x_col]])

    ## Itero para cada una de las features que tengo
    for feat in feature_cols:

        ## Obtengo el conjunto de valores asociados a la feature correspondiente
        y = df[feat].values

        ## Instancio un objeto de la clase LinearRegression() para poder hacer la regresión lineal
        lin = LinearRegression()

        ## Hago el ajuste del modelo de regresión lineal al conjunto de datos empíricos
        lin.fit(X, y)

        ## Obtengo los valores predichos de la variable dependiente a partir del modelo de regresión
        y_pred_lin = lin.predict(X)

        ## Obtengo el coeficiente de ajuste R^2 correspondiente al modelo de regresión lineal
        linear_r2 = r2_score(y, y_pred_lin)

        ## Obtengo el valor del slope (b_1) asociado al modelo de regresión
        slope = lin.coef_[0]

        ## Obtengo el valor del intercept (b_0) asociado al modelo de regresión
        intercept = lin.intercept_

        ## Instancio un objeto de la clase PolynomialFeatures() que me permita generar el polinomio
        ## de grado igual al parámetro de entrada
        poly = PolynomialFeatures(degree = poly_degree, include_bias = False)

        ## Hago el ajuste del polinomio a los datos y luego lo transformo
        X_poly = poly.fit_transform(X)

        ## Instancio un objeto de la clase LinearRegression() para poder hacer la regresión polinomial
        poly_model = LinearRegression()

        ## Hago el ajuste del modelo polnomial al dataset empírico
        poly_model.fit(X_poly, y)

        ## Obtengo los valores predichos de la variable dependiente a partir del modelo de regresión
        y_pred_poly = poly_model.predict(X_poly)

        ## Obtengo el coeficiente de ajuste R^2 correspondiente al modelo de regresión polinomial
        poly_r2 = r2_score(y, y_pred_poly)

        ## Ajusto un modelo de regresión lineal mediante Mínimos Cuadrados Ordinarios (OLS),
        ## a partir del cual luego se derivan los tests de significación de los coeficientes
        model = sm.OLS(y, X_ols).fit()

        ## Obtengo el p-valor asociado al test de significación del coeficiente de la variable independiente (slope)
        p_value = model.pvalues[x_col]

        ## Si el p-valor es menor que el nivel de significación, se rechaza la hipótesis nula.
        ## Esto indica evidencia estadística de una asociación lineal entre la feature y la edad.
        significant = p_value < alpha

        ## Construyo un diccionario con el resultado del análisis de regresión para los datos edad-feature
        results.append({"feature": feat, "slope": slope, "intercept": intercept, "linear_r2": linear_r2,
            "poly_r2": poly_r2, "p_value": p_value, "significant": significant})

    ## Retorno los resultados en forma de un dataframe ordenados según la significación
    return (pd.DataFrame(results).sort_values(by = ["significant", "p_value"],
            ascending = [False, True]).reset_index(drop = True))