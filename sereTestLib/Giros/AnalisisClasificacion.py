import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def compute_information_gain_features(df, feature_cols, target_col = "age_group",
    discrete_target = True, random_state = 42):
    """
    Calcula Information Gain (Mutual Information) por feature respecto a la clase.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset con features y variable objetivo.
    feature_cols : list of str
        Lista de features numéricas.
    target_col : str
        Columna objetivo (clases: age groups).
    discrete_target : bool
        Si True, asegura encoding discreto del target.
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    pandas.DataFrame
        Ranking de features por Information Gain.
    """

    ## Obtengo la matriz de datos de dimensiones (M,N) donde tengo:
    ## M: Cantidad de observaciones (en este caso giros) que tengo
    ## N: Cantidad de features por cada giro
    X = df[feature_cols].values

    ## Defino el vector con las asignaciones de cada giro al grupo etario al que corresponde
    y = df[target_col].values

    ## En caso de que el objetivo sea discreto, me aseguro de que la codificación sea consistente
    if discrete_target:

        ## Construyo un objeto LabelEncoder() que me permita asegurar la codificación discreta
        le = LabelEncoder()

        ## Hago la discretización del vector de asignaciones a un conjunto discreto obedeciendo
        ## el indexado de i = 0, 1, ..., n - 1 siendo n la cantidad de clases/grupos
        y = le.fit_transform(y)

    ## Hago la estimación de la información mutua en base a la matriz de datos de giros y features,
    ## con el vector de asignaciones de cada giro a los grupos etarios
    ig_scores = mutual_info_classif(X, y, random_state = random_state)

    ## Hago la construcción de la tabla de resultados de cada feature con su Information Gain
    results = pd.DataFrame({"feature": feature_cols, "information_gain": ig_scores})

    ## Ordeno los resultados de manera descendente según la Information Gain asociada a la feature
    results = results.sort_values("information_gain", ascending = False).reset_index(drop = True)

    ## Retorno la tabla conteniendo los resultados de las features con sus Information Gains
    return results