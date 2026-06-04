import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (GridSearchCV, StratifiedKFold)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score)
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import os
from datetime import datetime

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

def rbf_svm_univariate_feature_error(df, feature_cols, target_col = "age_group", 
    C = 1, gamma = "scale", n_splits = 5):
    """
    Calcula el rendimiento de clasificación univariada por feature utilizando un SVM con kernel RBF
    y validación cruzada estratificada.

    Para cada feature se entrena un modelo independiente y se evalúa su capacidad predictiva
    sobre el target especificado. Se reporta la precisión balanceada media, desviación estándar
    y el error asociado.

    Opcionalmente, genera y guarda matrices de confusión agregadas por feature en formato imagen.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset que contiene las features y la variable objetivo.

    feature_cols : list of str
        Lista de columnas numéricas a evaluar individualmente.

    target_col : str, default="age_group"
        Nombre de la variable objetivo (clases).

    C : float, default=1
        Parámetro de regularización del SVM.

    gamma : str or float, default="scale"
        Parámetro del kernel RBF.

    n_splits : int, default=5
        Número de folds para validación cruzada estratificada.

    save_confusion_dir : str or None, default=None
        Directorio donde se guardan las matrices de confusión por feature.
        Si es None, no se generan archivos.

    Returns
    -------
    pandas.DataFrame
        Tabla con resultados por feature:
        - feature
        - mean_balanced_accuracy
        - std_accuracy
        - error
    """

    ## Obtengo la matriz de datos, cuyas filas son los giros mientras que las columnas son las features
    X = df[feature_cols].values

    ## Obtengo el vector con los grupos etarios asociados a cada uno de los giros
    y = df[target_col].values

    ## Construyo un objeto especificando la cantidad de folds que yo voy a tomar  
    cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)

    ## Inicializo una lista vacía en la cual voy a almacenar los resultados de los análisis
    results = []

    ## Inicializo un diccionario en el cual voy a almacenar las predicciones del SVM
    predictions = {}

    ## Itero para cada una de las features que tengo
    for i, feat in enumerate(feature_cols):

        ## Obtengo el conjunto de valores de esa feature asociados a todos los giros
        X_feat = X[:, [i]]

        ## Inicializo una lista en donde voy a guardar los resultados (precisión y error) de cada fold
        fold_scores = []

        ## Inicializo una lista en la cual voy a guardar los grupos etarios reales de los giros
        y_true_all = []

        ## Inicializo una lista en la cual voy a guardar los grupos etarios predichos de los giros
        y_pred_all = []

        ## Itero para cada uno de los k folds que definí
        for train_idx, test_idx in cv.split(X_feat, y):

            ## Obtengo los valores de los estadísticos asociados a los conjuntos de entrenamiento y validación
            X_train, X_test = X_feat[train_idx], X_feat[test_idx]

            ## Obtengo los valores de las etiquetas asociadas a los conjuntos de entrenamiento y validación
            y_train, y_test = y[train_idx], y[test_idx]

            ## Construyo un objeto Pipeline especificando el modelo, los hiperparámetros y el escalamiento
            model = Pipeline([("scaler", StandardScaler()),
                    ("svm", SVC(kernel = "rbf", C = C, gamma = gamma))])

            ## Hago el entrenamiento del modelo SVM con el conjunto de datos de entrenamiento del fold k
            model.fit(X_train, y_train)

            ## Hago la validación del modelo entrenado para el fold k correspondiente
            y_pred = model.predict(X_test)

            ## Agrego la medida de precisión asociada al fold a la lista correspondiente
            fold_scores.append(balanced_accuracy_score(y_test, y_pred))

            ## Almaceno el grupo etario real del giro en la lista correspondiente
            y_true_all.extend(y_test)

            ## Almaceno el grupo etario predicho en la lista correspondiente
            y_pred_all.extend(y_pred)

        ## Retorno el resumen de la precisión media asociada al feature para el SVM en todos los folds
        results.append({"feature": feat, "mean_balanced_accuracy": np.mean(fold_scores),
                        "std_accuracy": np.std(fold_scores), "error": 1 - np.mean(fold_scores)})

        ## Agrego los resultados de las predicciones y las etiquetas reales para la feature correspondiente
        predictions[feat] = {"y_true": np.array(y_true_all), "y_pred": np.array(y_pred_all)}

    ## Retorno los resultados de todas las features ordenados descendentemente por la precisión
    ## junto con el diccionario de las precisiones correspondientes para cada una de las features
    return (pd.DataFrame(results).sort_values("mean_balanced_accuracy", ascending = False)
    .reset_index(drop = True), predictions)

def evaluar_features_svm_rbf(df, feature_cols, target_col, cv = 5, random_state = 42):
    """
    Evalúa el poder discriminativo univariado de cada feature usando un SVM con kernel RBF.

    Se utiliza validación cruzada estratificada por sujeto (StratifiedGroupKFold)
    para evitar leakage entre giros del mismo individuo.

    El modelo se optimiza internamente mediante GridSearchCV en cada fold externo.
    """

    ## Obtengo la matriz de datos, cuyas filas son los giros mientras que las columnas son las features
    X = df[feature_cols].values

    ## Obtengo el vector con los grupos etarios asociados a cada uno de los giros
    y = df[target_col].values

    ## Construyo un objeto especificando la cantidad de folds que yo voy a tomar  
    cv_outer = StratifiedKFold(n_splits = cv, shuffle = True, random_state = random_state)

    ## Construyo la grilla de valores posibles de parámetros
    param_grid = {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.1]}

    ## Inicializo una lista vacía en la cual voy a almacenar los resultados de los análisis
    results = []

    ## Itero para cada una de las features que tengo
    for i, feat in enumerate(feature_cols):

        ## Obtengo el conjunto de valores de esa feature asociados a todos los giros
        X_feat = X[:, [i]]

        ## Inicializo una lista en donde voy a guardar los resultados (precisión y error) de cada fold
        fold_scores = []

        ## Itero para cada uno de los k folds que definí
        for train_idx, test_idx in cv_outer.split(X_feat, y):

            ## Obtengo los valores de los estadísticos asociados a los conjuntos de entrenamiento y validación
            X_train, X_test = X_feat[train_idx], X_feat[test_idx]

            ## Obtengo los valores de las etiquetas asociadas a los conjuntos de entrenamiento y validación
            y_train, y_test = y[train_idx], y[test_idx]

            ## Construyo el pipeline especificando el escalamiento y el modelo a optimizar
            pipeline = Pipeline([("scaler", StandardScaler()),
                ("model", SVC(kernel = "rbf", class_weight = "balanced"))])

            ## Dada mi grilla de parámetros y el pipeline con el modelo, construyo el objeto de Grid Search
            grid = GridSearchCV(estimator = pipeline, param_grid = param_grid,
                scoring = "balanced_accuracy", cv = 3, n_jobs = -1)

            ## Hago el Grid Search sobre la grilla de parámetros
            grid.fit(X_train, y_train)

            ## Obtengo las predicciones del modelo con el conjunto 'optimo' de parámetros
            y_pred = grid.best_estimator_.predict(X_test)

            ## Agrego los resultados de los fold scores a la lista correspondiente
            fold_scores.append(balanced_accuracy_score(y_test, y_pred))

        ## Construyo un diccionario con las mejores estimaciones de la grilla para la feature dada
        results.append({"feature": feat, "mean_balanced_accuracy": np.mean(fold_scores),
            "std_balanced_accuracy": np.std(fold_scores), "error": 1 - np.mean(fold_scores)})

    ## Retorno el resumen de todos los resultados en forma de dataframe pandas
    return pd.DataFrame(results).sort_values("mean_balanced_accuracy", ascending = False
            ).reset_index(drop = True)

def _cv_score_svm(X, y, model, cv):
    """
    Calcula la media del balanced accuracy mediante validación cruzada para un modelo SVM.

    Esta función evalúa un clasificador ya definido utilizando validación cruzada estándar.
    No realiza ajuste de hiperparámetros; el modelo debe estar completamente configurado
    antes de llamar a esta función.

    Parameters
    ----------
    X : array-like de forma (n_samples, n_features)
        Matriz de características.

    y : array-like de forma (n_samples,)
        Vector de etiquetas objetivo.

    model : estimador de scikit-learn
        Clasificador ya configurado que implementa los métodos fit y predict
        (por ejemplo, un Pipeline con StandardScaler + SVC).

    cv : objeto de validación cruzada de scikit-learn
        Estrategia de particionado de los datos (por ejemplo, StratifiedKFold).

    Returns
    -------
    float
        Valor medio del balanced accuracy obtenido en los folds de validación cruzada.
    """

    ## Inicializo una lista vacía en la cual voy a almacenar las precisiones en todos los folds
    scores = []

    ## Itero para cada uno de los folds que definí
    for train_idx, test_idx in cv.split(X, y):

        ## Obtengo los valores de los estadísticos asociados a los conjuntos de entrenamiento y validación
        X_train, X_test = X[train_idx], X[test_idx]

        ## Obtengo los valores de las etiquetas asociadas a los conjuntos de entrenamiento y validación
        y_train, y_test = y[train_idx], y[test_idx]

        ## Hago el entrenamiento del modelo SVM con el conjunto de datos de entrenamiento del fold k
        model.fit(X_train, y_train)
        
        ## Hago la validación del modelo entrenado para el fold k correspondiente
        y_pred = model.predict(X_test)

        ## Agrego los resultados de los fold scores a la lista correspondiente
        scores.append(balanced_accuracy_score(y_test, y_pred))

    ## Retorno el valor de la precisión media de todas las folds para la feature en cuestión
    return np.mean(scores)

def sfs_svm_fixed(df, feature_cols, target_col, k = 5, C = 1.0, gamma = "scale", cv = 5,
                random_state = 42):
    """
    Selecciona un subconjunto óptimo de k características utilizando
    Sequential Forward Selection (SFS) con un SVM de kernel RBF.

    El algoritmo construye iterativamente el conjunto de features,
    añadiendo en cada paso aquella variable que maximiza el rendimiento
    promedio en validación cruzada (balanced accuracy).

    El modelo utilizado es un SVM con hiperparámetros fijos, sin ajuste
    de grilla, y con estandarización previa de los datos.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las variables predictoras y la variable objetivo.

    feature_cols : list of str
        Lista de nombres de las columnas candidatas a ser seleccionadas.

    target_col : str
        Nombre de la variable objetivo.

    k : int, default=5
        Número máximo de características a seleccionar.

    C : float, default=1.0
        Parámetro de regularización del SVM.

    gamma : str or float, default="scale"
        Parámetro del kernel RBF del SVM.

    cv : int, default=5
        Número de folds para validación cruzada estratificada.

    random_state : int, default=42
        Semilla para reproducibilidad del particionado.

    Returns
    -------
    pandas.DataFrame
        Historial del proceso de selección, incluyendo en cada iteración:
        - step: número de características seleccionadas
        - added_feature: feature incorporada en ese paso
        - score: balanced accuracy promedio en CV
        - features: lista acumulada de features seleccionadas
    """

    ## Obtengo la matriz de datos, cuyas filas son los giros mientras que las columnas son las features
    X = df[feature_cols].values.astype(float)

    ## Obtengo el vector con los grupos etarios asociados a cada uno de los giros
    y = df[target_col].values

    ## Construyo un objeto especificando la cantidad de folds que yo voy a tomar  
    skf = StratifiedKFold(n_splits = cv, shuffle = True, random_state = random_state)

    ## Inicializo la lista en la cual voy a ir guardando el historial de cada una 
    ## de las features que selecciono en cada una de las iteraciones
    history = []

    ## Inicializo una lista donde voy a almacenar aquellas features que selecciono
    selected = []

    ## Construyo la lista en la que voy a guardar aquellas features
    remaining = list(range(len(feature_cols)))

    ## Construyo el pipeline especificando el escalamiento y el modelo a optimizar
    base_model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel = "rbf", 
                C = C, gamma = gamma))])

    ## Mientras me queden features por seleccionar
    while len(selected) < k and len(remaining) > 0:

        ## Inicializo la variable que me almacena la 'mejor' feature en la iteración
        best_feat = None

        ## Inicializo la varaible que me almacena el 'mejor' valor del indicador
        best_score = - np.inf

        ## Itero para cada una de las features que aún me quedan sin seleccionar
        for j in remaining:

            ## Hago la concatenación de los índices de las features seleccionadas y la feature candidata
            candidate_idx = selected + [j]

            ## Obtengo el conjunto de valores correspondientes al conjnuto formado por la unión de las
            ## features ya seleccionadas y la nueva feature candidata
            X_subset = X[:, candidate_idx]

            ## Obtengo la precisión media correspondiente para el modelo dado para las features candidatas
            score = _cv_score_svm(X_subset, y, base_model, skf)

            ## En caso de que la precisión media con el nuevo conjunto de features mejore el valor previo
            if score > best_score:

                ## Asigno como nuevo valor de precisión el actual (al finalizar queda el valor óptimo)
                best_score = score

                ## Asigno como mejor feature del resto la feature actual
                best_feat = j

        ## Agrego la 'mejor feature' de las que quedan, a la lista de seleccionadas
        selected.append(best_feat)

        ## Elimino la 'mejor feature' de las que quedan, a la lista de las que quedan por seleccionar
        remaining.remove(best_feat)

        ## Actualizo el historial de features con la nueva feature agregada y el nuevo score
        history.append({"step": len(selected), "added_feature": feature_cols[best_feat],
            "score": best_score, "features": [feature_cols[i] for i in selected]})

    ## Retorno el historial en forma de un dataframe pandas
    return pd.DataFrame(history)