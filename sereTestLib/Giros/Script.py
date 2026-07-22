## Configuro nombre de la carpeta raíz
## (modificar si se utiliza en otro equipo)
root = "C:/Yo/Tesis"

## Importación de librerías
import sys
sys.path.append("{}/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica".format(root))
from LecturaDatosPacientes import *
from LecturaDatos import *
import numpy as np
import pickle
from scipy.constants import g
from itertools import combinations
from Visualizacion import *
from Preprocesamiento import *
from ExtraccionFeatures import *
from AnalisisEstadistico import *
from MetadatosFeatures import *
from AnalisisClasificacion import *

## Programa principal
if __name__== '__main__':

    ## Opcion 1: Graficar histograma caracterizando la distribución de edades de la población
    ## Opcion 2: Detectar y extraer features de los giros
    ## Opcion 3: Procesar features de giros previamente extraídas (análisis por persona)
    ## Opcion 4: Procesar features de giros previamente extraídas (análisis por giro)
    ## Opcion 5: Extender features ya existentes en el parquet (feature store update)
    opcion = 4

    ## Obtengo la información correspondiente a todos los pacientes en la base de datos
    pacientes, ids_existentes = LecturaDatosPacientes()

    ## En caso de que quiera caracterizar la población según la edad
    if opcion == 1:

        ## Obtengo el histograma por edad
        graficar_histograma_edades(pacientes)

        ## Obtengo el histograma de la cantidad de caídas para pacientes mayores a 75 años
        graficar_caidas_por_rango_etario(pacientes, "Edad", "Caida", [75, float("inf")])

        ## Obtengo el histograma de la cantidad de caídas para pacientes entre 60 y 75 años
        graficar_caidas_por_rango_etario(pacientes, "Edad", "Caida", [60, 75])

        ## Obtengo el histograma de la cantidad de caídas para pacientes menores a 60 años
        graficar_caidas_por_rango_etario(pacientes, "Edad", "Caida", [0, 60])

    ## En caso que quiera hacer la detección y extracción de features de giros
    elif opcion == 2:

        ## Configuro una variable que me de a elegir si quiero graficar datos o no
        graficar = False

        ## Seteo el sistema inercial que voy a usar de referencia para el cálculo de orientación
        sist_inercial = 'ENU'

        ## Construyo el vector de aceleración gravitatoria asociado al sistema de referencia inercial
        gravity = np.array([0.0, 0.0, -g if sist_inercial == "ENU" else g])

        ## Inicializo una lista en la cual voy a almacenar todas las features de todos los giros de todos los pacientes
        features_giros_total = []

        ## Itero para cada uno de los pacientes presentes en la base de datos
        for id_paciente in ids_existentes:

            ## Coloco un bloque try-except en caso de que ocurra algún error
            try:

                ## Despliego un mensaje indicando el paciente que estoy procesando
                print("Procesando giros del paciente de ID: {}".format(id_paciente))

                ## Hago la lectura de las mediciones de la IMU del individuo
                ## Las medidas del Shimmer3 vienen en m/s2 para el acelerómetro y grados/s para el giroscopio
                data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = id_paciente, 
                lectura_datos_propios = False, ruta = '{}/sereData/sereData/Registros/MarchaLibre_Sabrina.txt'.format(root))

                ## Hago la conversión de los valores de velocidad angular de grados/s a rad/s
                gyro = np.deg2rad(gyro)

                ## Defino la frecuencia de muestreo del sistema
                frec_muestreo = 1 / periodoMuestreo

                ## Hago la estimación de la orientación del sistema de la IMU con respecto al sistema de referencia inercial
                imu_quat = estimar_orientacion_ekf(acel, gyro, frec_muestreo, sist_inercial)

                ## Hago la rotación de la señal del giroscopio del sistema de la IMU al sistema inercial
                ang_vel_inercial = rotate_body_to_world(gyro, imu_quat)

                ## Hago la rotación de la señal del acelerómetro del sistema de la IMU al sistema inercial
                acc_inercial = rotate_body_to_world(acel, imu_quat)

                ## Hago el suavizado de las señales del giroscopio expresadas en el sistema inercial
                ## con el fin de mitigar las excursiones significativas causadas por el ruido
                gyro_suav = np.column_stack([moving_average(ang_vel_inercial[:, 0], frec_muestreo),
                                        moving_average(ang_vel_inercial[:, 1], frec_muestreo),
                                        moving_average(ang_vel_inercial[:, 2], frec_muestreo)])

                ## Hago el suavizado de las señales del acelerómetro expresadas en el sistema inercial
                ## con el fin de mitigar las excursiones significativas causadas por el ruido
                acc_suav = np.column_stack([moving_average(acc_inercial[:, 0], frec_muestreo),
                                        moving_average(acc_inercial[:, 1], frec_muestreo),
                                        moving_average(acc_inercial[:, 2], frec_muestreo)])

                ## Hago la detección de los giros en base a la velocidad angular en el eje vertical
                giros = detect_turns_windowed(gyro_suav[:, 2], frec_muestreo)

                ## En caso de que quiera graficar velocidades angulares con los tramos de giro
                if graficar:

                    ## Hago la graficación de los tramos en los que se detectan giros de los que no
                    plot_signal_with_events(acc_suav[:,0], giros, fs = frec_muestreo,
                            save_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/GirosTemporales/Suavizadas/{}/ax.png".format(root, id_paciente))
                    plot_signal_with_events(acc_suav[:,1], giros, fs = frec_muestreo,
                            save_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/GirosTemporales/Suavizadas/{}/ay.png".format(root, id_paciente))
                    plot_signal_with_events(acc_suav[:,2], giros, fs = frec_muestreo,
                            save_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/GirosTemporales/Suavizadas/{}/az.png".format(root, id_paciente))
                    plot_signal_with_events(gyro_suav[:,0], giros, fs = frec_muestreo,
                            save_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/GirosTemporales/Suavizadas/{}/wx.png".format(root, id_paciente))
                    plot_signal_with_events(gyro_suav[:,1], giros, fs = frec_muestreo,
                            save_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/GirosTemporales/Suavizadas/{}/wy.png".format(root, id_paciente))
                    plot_signal_with_events(gyro_suav[:,2], giros, fs = frec_muestreo,
                            save_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/GirosTemporales/Suavizadas/{}/wz.png".format(root, id_paciente))

                ## Extraigo los segmentos de acelerómetros y giroscopios separados por giros
                segmentos = extraer_segmentos_giros(acel, gyro, giros)

                ## Hago la extracción de features correspondientes a los giros, pasando como argumento
                ## señales tanto de acelerómetro como giroscopio preprocesadas
                features_giros = extraer_features_basicas(imu_quat, segmentos, frec_muestreo, 
                                                        id_paciente, gyro_suav, acc_suav, gravity)

                ## Almaceno las features de los giros de dicho paciente a la lista general de features de giros de pacientes
                features_giros_total.extend(features_giros)

            ## En caso de que ocurra algún error en el procesamiento de los giros de los pacientes
            except:

                ## Notifico el ID de la persona en la cual el algoritmo dio error y continúo con la ejecución
                print("Error en el procesamiento para la persona de ID: {}".format(id_paciente))

                ## Conntinúo con el procesamiento de los datos correspondientes a la siguiente persona
                continue

        ## Hago la conversión de lista de diccionarios a DataFrame Pandas
        df_features = pd.DataFrame(features_giros_total)

        ## Hago el guardado del dataframe pandas bajo la extensión de .parquet
        output_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/features_giros_acc+gyro+ind_sgrav.parquet".format(root)
        df_features.to_parquet(output_path, index = False)

    ## En caso de que yo quiera procesar las features de los giros que fueron previamente extraídas
    ## pero hago el procesamiento por cada giro separadamente sin hacer sumarización por persona
    elif opcion == 4:

        ## ======================================================
        ## CONFIGURACIÓN DE PARÁMETROS Y CARGADO DE ARCHIVOS DEL PIPELINE
        ## ======================================================

        ## Configuro una variable que me de a elegir si quiero graficar boxplots de features
        graficar_boxplots = False

        ## Configuro una variable que me de a elegir si quiero graficar scatterplots de features
        graficar_scatter = False

        ## Configuro una variable que me de a elegir si quiero graficar matrices de significación
        ## del test de hipótesis de Wilcoxon
        graficar_wilcoxon = False

        ## Configuro una variable que me de a elegir si quiero graficar la evolución de cada una de
        ## las features en función de la edad a la que corresponde el giro
        graficar_featvsedad = False

        ## Configuro una variable que me de a elegir si quiero graficar las matrices de confusión
        ## para cada una de las features usando SVM con K-Fold Cross Validation
        graficar_matconf = False

        ## Hago la lectura del archivo .parquet donde guardo el dataframe Pandas que contiene
        ## la lista con los diccionarios con todos los parámetros de los giros detectados
        features_giros_total = pd.read_parquet(
            "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/features_giros_acc+gyro+ind_sgrav.parquet".format(root))

        ## Hago la lectura del archivo .pickle el cual contiene los segmentos de señales de acelerómetro
        ## y giroscopio correspondientes a los giros, rotados a una base inerc
        with open("{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/turns_raw_segments.pickle".format(root), "rb") as f:
            turns_raw = pickle.load(f)

        ## Hago la lectura del archivo .pickle el cual contiene los segmentos de señales de acelerómetro
        ## y giroscopio correspondientes a los giros y suavizados (a partir de las cuales se calcualn 
        ## las features), rotados a una base inercial
        with open("{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/turns_smoothed_segments.pickle".format(root), "rb") as f:
            turns_smooth = pickle.load(f)
        
        ## Hago la lectura de archivo .parquet donde tengo el historial óptimo de features con sfs
        sfs_features_results = pd.read_parquet(
            "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/sfs_features.parquet".format(root))

        ## ======================================================
        ## PREPROCESAMIENTO DE LOS DATOS (FEATURES EXTRAÍDAS)
        ## ======================================================

        ## Obtengo el diccionario conteniendo el conjunto de IDs, indices de comienzo y indices de
        ## terminación de cada uno de los giros detectados (uso los giros suavizados pero es arbitrario)
        turns_map = {(t["id"], t["start_idx"], t["end_idx"]): t for t in turns_smooth}

        ## Me aseguro que el campo "id" del dataframe de features esté en formato string
        features_giros_total["id"] = features_giros_total["id"].astype(str)

        ## Agrego como nueva feature la norma L2 de la energía de jerk de las componentes de la
        ## aceleración en el plano horizontal de la marcha
        features_giros_total["acc_horz_jerk_energy_L2"] = np.sqrt(
            features_giros_total["ax_jerk_energy"] ** 2 + features_giros_total["ay_jerk_energy"] ** 2)

        ## Agrego como nueva feature la norma L2 de la energía de jerk de las componentes de la
        ## velocidad angular en el plano horizontal de la marcha
        features_giros_total["gyro_horz_jerk_energy_L2"] = np.sqrt(
            features_giros_total["wx_jerk_energy"] ** 2 + features_giros_total["wy_jerk_energy"] ** 2)

        ## Selecciono y ordeno las columnas de features por giro para asegurar consistencia
        array_features = features_giros_total[["id", "start_idx", "end_idx"] + FEATURE_COLUMNS].copy()

        ## Obtengo únicamente información de la edad y la ID asociada a cada paciente
        df_patients = pacientes[["sampleid", "Edad", "Caida"]].copy()

        ## Me aseguro que los IDs se encuentran todos expresados en formato string
        df_patients["sampleid"] = df_patients["sampleid"].astype(str)

        ## Me aseguro que el número de caídas por año está expresado como número entero
        df_patients["Caida"] = df_patients["Caida"].astype(int)

        ## Asocio cada giro con la edad del sujeto correspondiente mediante un merge por ID,
        ## eliminando la columna redundante que contiene la ID de la persona a la que corresponde el giro
        df_dataset = array_features.merge(df_patients, left_on = "id", right_on = "sampleid", 
                                        how = "inner").drop(columns = ["sampleid"])

        ## Asigno cada persona al grupo etario correspondiente según su edad (0: edad < 60, 1: 
        ## 60 < edad < 75, 2: edad > 75) generando una nueva columna denominada "age_group"
        df_dataset["age_group"] = asignar_grupo_edad(df_dataset["Edad"])

        ## Asigno una variable que me diga si el giro corresponde a una persona que tiene al menos
        ## una caída por año (discriminación binaria por caídas) de modo que asigno 1 a los giros
        ## de los pacientes con al menos una caída, y 0 a los giros de pacientes sin caídas
        df_dataset["caida_bin"] = (df_dataset["Caida"] >= 1).astype(int)

        ## Normalizo el ángulo de giro para eliminar dependencia de la dirección (CW vs CCW)
        df_dataset["angle_deg"] = np.abs(df_dataset["angle_deg"])

        ## Elimino las columnas del dataframe que no corresponden a features numéricas de los giros
        feature_cols = df_dataset.columns.drop(["id", "Edad", "Caida", "age_group", "caida_bin"])

        ## Creo una copia de los datos para visualización (de manera de no tocar los datos originales)
        df_plot = df_dataset.copy()

        ## Tomo la magnitud del ángulo de giro para normalizar la información
        df_plot["angle_deg"] = np.abs(df_plot["angle_deg"])

        ## Construyo una lista conteniendo los nombres de las tres señales de giroscopio
        gyro_axes = ["wx", "wy", "wz"]

        ## Itero para cada una de las señales del giroscopio en los tres ejes
        for axis in gyro_axes:

            ## En caso de que yo tenga como feature el valor medio de la señal
            if f"{axis}_mean" in df_plot.columns:

                ## Me quedo con la magnitud del valor medio que voy a usar para la visualización
                df_plot[f"{axis}_mean"] = np.abs(df_plot[f"{axis}_mean"])

        ## Obtengo una lista con todos los posibles pares de features de los giros
        pairs = list(combinations(feature_cols, 2))

        ## Hago una selección de aquellos features estadísticos que son significativos para los tests
        ## Elimino aquellas features para las cuales todos los valores sean constantes
        valid_features = [c for c in feature_cols if df_dataset[c].nunique() > 1]

        ## ======================================================
        ## TEST DE HIPÓTESIS DE KRUSKAL-WALLIS EN EL DATASET
        ## ======================================================

        ## IMPORTANTE EN LOS TESTS DE HIPÓTESIS:
        ## Si p_valor <= alpha --> Rechazo H0 al nivel de significación alpha
        ## Si p_valor > alpha --> No rechazo H0 al nivel de significación alpha
        ## alpha = Probabilidad de cometer error tipo I (error de significación de la prueba)
        ## Cuanto menor es el p-valor, implica una mayor contradicción a H0 por parte de los datos muestrales

        ## Aplico el test de Kruskal Wallis a todas las features que tengo para comprobar si para alguna de
        ## las features de los giros existe al menos una distribución diferente a las demás
        ## Si <<significant_fdr>> = True entonces estoy rechazando la hipótesis nula lo cual implica que no
        ## todas las distribuciones son iguales, o lo que es lo mismo, al menos una distribución es distinta
        ## Formalmente, <<epsilon_sq>> me da la proporción de la variabilidad de la feature que se explica por
        ## el grupo etario (pensar similitudes con R^2 de la regresión lineal) --> En mi caso con mirar H alcanza
        results_kruskal = kruskal_wallis_features(df_dataset, valid_features)

        ## Hago la graficación de aquellas features que más rechazan la idea de que todas las distribuciones
        ## son iguales (que es H0 de Kruskal Wallis)
        plot_kruskal_results(results_kruskal)

        ## ======================================================
        ## TEST DE SUMA DE RANGOS DE WILCOXON EN EL DATASET
        ## ======================================================

        ## El test de Wilcoxon de suma de rangos intenta rechazar la hipótesis nula de que dos muestras
        ## independientes provienen de la misma distribución. Entonces ejecuto el test de hipótesis de
        ## Wilcoxon de suma de rangos para comprobar diferencias entre distribuciones uno a uno
        results_wilcoxon = pairwise_wilcoxon_rank_sum(df_dataset, valid_features, group_col = "age_group")

        ## Selecciono únicamente las columnas más relevantes del dataframe de resultados de aplicar el
        ## test de hipótesis de Wilcoxon al dataset dado con los grupos definidos
        res_min = results_wilcoxon[["feature", "group_1", "group_2", "p_value", "p_fdr", "significant_fdr"]]

        ## Represento los resultados del test de hipótesis de Wilcoxon uno a uno en una matriz
        ## La idea es que para cada una de las features tenga un diccionario que resuma para cada feature,
        ## si para cada par de grupos etarios el test de Wilcoxon de suma de rangos se rechaza (True
        ## en caso de que rechazo H0) al nivel de significación pasado como entrada (por defecto alpha = 0.05)
        matrices_wilcoxon = wilcoxon_pairwise_matrices(results_wilcoxon)

        ## ======================================================
        ## CONSTRUCCIÓN DE MODELOS DE REGRESIÓN LINEAL (FEATURE vs EDAD)
        ## ======================================================

        ## --> TEST SIGNIFICACIÓN: Suponiendo que tengo un modelo de regresión lineal dado y = b0 + b1 * x + eps
        ## donde tengo b0 como el intercept y b1 como el slope, el test de significación tiene como hipótesis
        ## nula H0: b1 = 0 es decir que no hay relación lineal entre la edad y la feature.
        ## Cuanto menor sea el p-valor, los datos muestrales sugieren con más fuerza la existencia de
        ## una relación lineal DE PENDIENTE NO NULA entre la edad y la feature

        ## --> VALOR DE R^2: R^2 me indica el porcentaje de la varianza de la variable dependiente (feature)
        ## que es explicada en base al modelo de regresión lineal (coeficiente de ajuste). Intuitivamente
        ## un mayor valor de R^2 implica que el modelo de regresión lineal captura más efectivaemente la
        ## dependencia que existe entre la edad y la feature

        ## Hago análisis de regresión (en principio lineal y polinomial) comparando directamente las
        ## features (como variable dependiente) contra la edad (variable indpendiente).
        results_df = regression_analysis(df = df_dataset, feature_cols = feature_cols, x_col = "Edad",
                                        poly_degree = 2, alpha = 0.05)

        ## ======================================================
        ## CLUSTERING K-MEANS DEL DATASET CON TODAS LAS FEATURES
        ## ======================================================

        ## Selecciono features numéricas válidas para clustering (sin NaNs ni constantes)
        valid_cluster_features = [c for c in feature_cols if df_dataset[c].nunique() > 1]

        ## Hago clustering para todos mis giros con mi conjunto de features correspondiente
        cluster_results = aplicar_clustering_giros(df = df_dataset.copy(),
                    feature_cols = valid_cluster_features, k_range = range(2, 7), random_state = 42)

        ## Evalúo qué tan bien el clustering separa los grupos etarios
        ## Como regla de decisión, tengo que una precisión > 0.85 representa una buena separación
        cluster_eval = evaluar_clustering_por_edad(cluster_results, group_col = "age_group")

        ## ======================================================
        ## RANKING UNIVARIADO DE FEATURES USANDO INFORMACIÓN MUTUA
        ## ======================================================

        ## Obtengo los resultados de calcular el Information Gain de las features respecto al grupo etario
        ig_results = compute_information_gain_features(df_dataset, feature_cols = feature_cols,
                                target_col = "age_group")

        ## Hago el ordenamiento de todas las features con su respectiva Information Gain de mayor a menor
        candidate_features = (ig_results.sort_values("information_gain", ascending = False)
                                ["feature"].tolist())

        ## ======================================================
        ## RANKING UNIVARIADO DE FEATURES USANDO SVM (SUPPORT VECTOR MACHINE)
        ## ======================================================

        ## Configuro un único valor del hiperparámetro C de SVM para usar en todo el pipeline
        C = 10

        ## Configuro un único valor del hiperparámetro C de SVM para usar en todo el pipeline
        gamma = "scale"

        ## Configuro la cantidad de splits que voy a hacer con K-Fold Cross Validation
        n_splits = 5

        ## SVM Univariado: Resultados al hacer los tests de clasificación univariada usando una SVM
        results_svm, svm_predictions = rbf_svm_univariate_feature_error(df = df_dataset, 
                feature_cols = feature_cols, target_col = "age_group", C = C, gamma = gamma, 
                n_splits = n_splits)

        ## Hago la graficación de las features según el error de predicción medio en las K-Folds
        plot_svm_feature_error_ranking(results_svm, top_k = 10, annotate = True,
            title = "Tres Clases - Ranking univariado de features con SVM – clasificación por grupo etario",
            C = C, gamma = gamma)

        ## Obtengo un conjunto de las mejores features luego de hacer el SVM univariado
        top_features_univ = results_svm.nsmallest(5, "error")

        ## ======================================================
        ## RANKING MULTIVARIADO DE FEATURES USANDO SVM (SUPPORT VECTOR MACHINE)
        ## ======================================================

        ## Inicializo una variable que especifique la cantidad máxima de features con las que me quiero quedar
        k_top_features = 2

        ## Sequential Forward Feature Selection (SFFS): Obtengo aquellos conjuntos de features que me dan 
        ## la mejor performance. En otras palabras, elijo el conjunto de las k features que me da mayor 
        ## discriminación de los feature vectors de los giros en los rangos etarios correspondientes
        sfs_results = sfs_svm_fixed(df = df_dataset, feature_cols = candidate_features,
                                target_col = "age_group", k = k_top_features, C = C, gamma = gamma, 
                                cv = n_splits)

        ## Obtengo un conjunto de las mejores features luego de hacer el SVM SFFS multivariado
        top_features_multiv = sfs_results['features'][k_top_features - 1]

        ## ======================================================
        ## RANKING UNIVARIADO DE FEATURES USANDO SVM (SUPPORT VECTOR MACHINE) -- DOS GRUPOS ETARIOS
        ## ======================================================

        ## Hago una copia del dataframe con los valores originales
        df_dataset_binary = df_dataset.copy()

        ## Hago un reagrupamiento de modo que construyo dos conjuntos que están determinados
        ## Asigno como grupo etario 0 a las personas de edad menor o igual a 75, y 1 a las personas
        ## con una edad mayor a 75 años (se extiende la asignación a los giros)
        df_dataset_binary["age_group_binary"] = np.where(df_dataset_binary["Edad"] <= 75, 0, 1)

        ## Obtengo el conjunto de giros y sus features para personas que no tienen ninguna caída por año
        df_no_fall = df_dataset_binary[df_dataset_binary["caida_bin"] == 0].copy()

        ## Obtengo el conjunto de giros y sus features para personas que tienen al menos una caida al año
        df_fall = df_dataset_binary[df_dataset_binary["caida_bin"] == 1].copy()

        ## Hago el test de Wilcoxon con agrupación etaria binaria para todos los giros,
        ## comparando las distribuciones de cada feature entre: grupo 0 -> Edad <= 75, grupo 1 -> Edad > 75
        results_wilcoxon_bin = pairwise_wilcoxon_rank_sum(df_dataset_binary, valid_features,
            group_col = "age_group_binary")

        ## SVM Univariado: Resultados al hacer los tests de clasificación univariada usando una SVM
        ## pero en este caso aplicado al problema de clasificación binaria con las clases como antes      
        results_svm_bin, svm_predictions_bin = rbf_svm_univariate_feature_error(df = df_dataset_binary, 
            feature_cols = feature_cols, target_col = "age_group_binary", C = C, gamma = gamma, 
            n_splits = n_splits)

        ## Hago la graficación de las features según el error de predicción medio en las K-Folds
        plot_svm_feature_error_ranking(results_svm_bin, top_k = 10,
            annotate = True, title = "Ranking univariado de features con SVM optimizado (división etaria binaria) – dataset completo",
            C = C, gamma = gamma, wilcoxon_results = results_wilcoxon_bin, group_pair = (0, 1))

        ## ======================================================
        ## RANKING MULTIVARIADO DE FEATURES USANDO SVM (SUPPORT VECTOR MACHINE) -- DOS GRUPOS ETARIOS
        ## ======================================================

        ## Sequential Forward Feature Selection (SFFS): Obtengo aquellos conjuntos de features que me dan 
        ## la mejor performance. En otras palabras, elijo el conjunto de las k features que me da mayor 
        ## discriminación de los feature vectors de los giros en los rangos etarios correspondientes
        sfs_results_bin = sfs_svm_fixed(df = df_dataset_binary, feature_cols = candidate_features,
                                target_col = "age_group_binary", k = k_top_features, C = C, gamma = gamma,
                                cv = n_splits)

        ## Obtengo un conjunto de las mejores features luego de hacer el SVM SFFS multivariado
        top_features_multiv = sfs_results_bin['features'][k_top_features - 1]

        ## ======================================================
        ## TESTS DE HIPÓTESIS DE SUMA DE RANGOS DE WILCOXON - AGRUPACIÓN ETARIA BINARIA
        ## Y SEPARADOS SEGÚN CAIDAS/NO CAIDAS
        ## ======================================================

        ## Hago el test de Wilcoxon con agrupación etaria binaria para giros correspondientes 
        ## a personas que no tienen ninguna caída por año
        results_wilcoxon_no_fall = pairwise_wilcoxon_rank_sum(df_no_fall, valid_features, 
                                group_col = "age_group_binary")

        ## Construyo las matrices con los resultados de tests de Wilcoxon para giros correspondientes
        ## a personas que no tienen ninguna caída por año (a partir de los resultados de los tests)
        matrices_wilcoxon_no_fall = wilcoxon_pairwise_matrices(results_wilcoxon_no_fall)

        ## Hago el test de Wilcoxon con agrupación etaria binaria para giros correspondientes
        ## a personas que tienen al menos una caída por año
        results_wilcoxon_fall = pairwise_wilcoxon_rank_sum(df_fall, valid_features, 
                                group_col = "age_group_binary")

        ## Construyo las matrices con los resultados de tests de Wilcoxon para giros correspondientes
        ## a personas que no tienen al menos una caída al año (a partir de los resultados de los tests)
        matrices_wilcoxon_fall = wilcoxon_pairwise_matrices(results_wilcoxon_fall)

        ## ======================================================
        ## RANKING FEATURES SVM UNIVARIADO - DIVISION ETARIA BINARIA 
        ## SEPARADOS SEGÚN CAIDAS/NO CAIDAS
        ## ======================================================

        ## SVM Univariado con división etaria binaria (>75, <75) para aquellos giros asociados
        ## a las personas para las que no se registra ninguna caída por año
        results_svm_no_fall, svm_predictions_no_fall = rbf_svm_univariate_feature_error(
            df = df_no_fall, feature_cols = feature_cols, target_col = "age_group_binary",
            C = C, gamma = gamma, n_splits = n_splits)

        ## Hago la graficación de las features según el error de predicción medio en las K-Folds
        ## para el ranking univariado con división etaria binaria para personas sin caídas
        plot_svm_feature_error_ranking(results_svm_no_fall, top_k = 10,
            annotate = True, title = "Ranking univariado de features con SVM (división etaria binaria) – personas sin caídas",
            C = C, gamma = gamma, wilcoxon_results = results_wilcoxon_no_fall, group_pair = (0, 1))

        ## Hago el SFFS Multivariado para los resultados de los giros asociados a personas sin caídas
        sfs_results_no_fall = sfs_svm_fixed(df = df_no_fall, feature_cols = feature_cols,
            target_col = "age_group_binary", k = k_top_features, C = C, gamma = gamma,
            cv = n_splits)

        ## Obtengo un conjunto de las mejores features luego de hacer el SVM SFFS multivariado
        top_features_multiv_no_fall = sfs_results_no_fall['features'][k_top_features - 1]

        ## SVM Univariado con división etaria binaria (>75, <75) para aquellos giros asociados
        ## a las personas para las que se registra al menos una caída por año
        results_svm_fall, svm_predictions_fall = rbf_svm_univariate_feature_error(
            df = df_fall, feature_cols = feature_cols, target_col = "age_group_binary",
            C = C, gamma = gamma, n_splits = n_splits)

        ## Hago la graficación de las features según el error de predicción medio en las K-Folds
        ## para el ranking univariado con división etaria binaria para personas con al menos una caida
        plot_svm_feature_error_ranking(results_svm_fall, top_k = 10, annotate = True,
            title = "Ranking univariado de features con SVM (división etaria binaria) – personas con al menos una caída",
            C = C, gamma = gamma, wilcoxon_results = results_wilcoxon_fall, group_pair = (0, 1))

        ## Hago el SFFS Multivariado para los resultados de los giros asociados a personas con caidas
        sfs_results_fall = sfs_svm_fixed(df = df_fall, feature_cols = feature_cols,
            target_col = "age_group_binary", k = k_top_features, C = C, gamma = gamma, cv = n_splits)

        ## Obtengo un conjunto de las mejores features luego de hacer el SVM SFFS multivariado
        top_features_multiv_fall = sfs_results_fall['features'][k_top_features - 1]

        ## ======================================================
        ## VISUALIZACIÓN DEL ESPACIO DE FEATURES EN 2D
        ## SEPARADOS SEGÚN CAIDAS/NO CAIDAS Y ERRORES DE CLASIFICACIÓN SVM
        ## ======================================================

        ## Selecciono las 2 features óptimas según el proceso SFS con SVM 
        ## correspondientes al dataset de giros asociados a personas que no tienen niguna caída
        features_no_fall = sfs_results_no_fall.iloc[-1]["features"][:2]

        ## Obtengo las predicciones mediante un algoritmo de validación LOO con SVM para los giros
        ## asociados a las personas que no tienen ninguna caída
        y_pred_no_fall = svm_loo_predict(df_no_fall, features_no_fall, target_col = "age_group_binary",
            C = C, gamma = gamma)

        ## Hago una copia del dataframe que contiene los giros de los no caedores
        df_no_fall = df_no_fall.copy()

        ## Creo una nueva columna en el dataframe de los giros de personas sin caídas la cual
        ## contiene las predicciones asociadas a cada uno de los giros mediante el algoritmo SVM LOO
        df_no_fall["y_pred"] = y_pred_no_fall

        ## Hago el cálculo del tipo de error de clasificación para todos los giros con el 
        ## algoritmo SVM LOO
        df_no_fall["error_type"] = compute_error_type(df_no_fall["age_group_binary"].values, 
            y_pred_no_fall)

        ## Selecciono las 2 features óptimas según el proceso SFS con SVM 
        ## correspondientes al dataset de giros asociados a personas que tienen al menos una caida
        features_fall = sfs_results_fall.iloc[-1]["features"][:2]

        ## Obtengo las predicciones mediante un algoritmo de validación LOO con SVM para los giros
        ## asociados a las personas que tienen al menos una caída
        y_pred_fall = svm_loo_predict(df_fall, features_fall, target_col = "age_group_binary",
            C = C, gamma = gamma)

        ## Creo una nueva columna en el dataframe de los giros de personas con al menos una caida que
        ## contiene las predicciones asociadas a cada uno de los giros mediante el algoritmo SVM LOO
        df_fall = df_fall.copy()

        ## Creo una nueva columna en el dataframe de los giros de personas con al menos una caída que
        ## contiene las predicciones asociadas a cada uno de los giros mediante el algoritmo SVM LOO
        df_fall["y_pred"] = y_pred_fall

        ## Hago el cálculo del tipo de error de clasificación para todos los giros 
        ## con el algoritmo SVM LOO (Leave One Out)
        df_fall["error_type"] = compute_error_type(df_fall["age_group_binary"].values, y_pred_fall)

        ## Hago la graficación del diagrama de dispersión para los no caedores
        plot_error_space(df_no_fall, features_no_fall[0], features_no_fall[1],
                        title = "Giros de personas sin ninguna caída (SVM LOO)")

        ## Hago la graficación del diagrama de dispersión para los caedores
        plot_error_space(df_fall, features_fall[0], features_fall[1],
                        title = "Giros de personas con al menos una caída (SVM LOO)")

        ## Hago el SFFS Multivariado para los resultados de los giros asociados a personas con caidas
        sfs_results_fall = sfs_svm_fixed(df = df_fall, feature_cols = feature_cols,
            target_col = "age_group_binary", k = k_top_features, C = C, gamma = gamma, cv = n_splits)

        ## ======================================================
        ## ANÁLISIS JOVENES NO CAEDORES (<60 AÑOS) VS MAYORES CAEDORES (> 75 AÑOS)
        ## ======================================================

        ## Hago una copia del dataset original
        df_turns = df_dataset_binary.copy()

        ## Clase 0: personas mayores a 75 años con al menos una caída
        condition_fallers = (df_turns["Edad"] > 75) & (df_turns["Caida"] > 0)

        ## Clase 1: personas menores o iguales a 60 años sin caídas
        condition_controls = (df_turns["Edad"] <= 60) & (df_turns["Caida"] == 0)

        ## Mantengo únicamente los dos grupos extremos
        df_turns = df_turns[condition_fallers | condition_controls].copy()

        ## Asigno las dos clases extremas como 0 y 1 con el siguiente criterio
        ## 0 -> giros de personas de edad mayor a 75 con al menos una caída
        ## 1 -> giros de personas de edad menor o igual a 60 sin ninguna caída
        df_turns["group_extreme"] = np.where(condition_fallers.loc[df_turns.index], 0, 1)

        ## Hago un ranking de features univariado usando un algoritmo SVM
        results_svm_extreme, svm_predictions_extreme = rbf_svm_univariate_feature_error(df = df_turns,
            feature_cols = feature_cols, target_col = "group_extreme", C = C, gamma = gamma, 
            n_splits = n_splits)

        ## Hago el diagrama de dispersión en el plano de las 2 features más discriminadoras
        plot_feature_space_2d(df = df_turns, feature_x = "acc_horz_jerk_energy_L2", 
            feature_y = "wz_jerk_energy", target_col = "group_extreme", 
            class_labels = {0: "Mayor a 75 - Al menos una caída", 1: "Menor a 60 - Sin caídas"}, 
            title = "Scatter Plot - Clases Extremas", alpha = 0.5)

        ## Hago el SFFS Multivariado para los resultados de los giros asociados a personas con caidas
        sfs_results_fall = sfs_svm_fixed(df = df_turns, feature_cols = feature_cols, 
            target_col = "group_extreme", k = k_top_features, C = C, gamma = gamma, cv = n_splits)

        ## Hago el test de hipótesis de Wilcoxon uno a uno para ver si las features individuales
        ## pueden rechazarme el hecho de que las distribuciones de las clases sean iguales
        wilcoxon_extreme_results = pairwise_wilcoxon_rank_sum(df = df_turns, feature_cols = feature_cols,
            group_col = "group_extreme")

        ## Grafico el ranking univariado de features para dicho conjunto de giros con las agrupaciones
        ## correspondientes escribiendo para cada feature el error estimado por SVM y el p-valor de Wilcoxon
        plot_svm_feature_error_ranking(results_svm_extreme, top_k = 10, annotate = True,
            title = "Ranking univariado de features con SVM - mayores >75 con caídas vs menores <=60 sin caídas",
            C = C, gamma = gamma, wilcoxon_results = wilcoxon_extreme_results, group_pair = (0, 1))

        ## Hago el diagrama de dispersión en el plano de las features "acc_horz_jerk_energy_L2" y
        ## "gyro_horz_jerk_energy_L2" que representan las energías promedio de las derivadas discretas
        ## de los segmentos de acelerómetro y giroscopio en el giro (clases extremas)
        plot_feature_space_2d(df = df_turns, feature_x = "acc_horz_jerk_energy_L2", 
            feature_y = "gyro_horz_jerk_energy_L2", target_col = "group_extreme", 
            class_labels = {0: "Mayor a 75 - Al menos una caída", 1: "Menor a 60 - Sin caídas"}, 
            title = "Scatter Plot - Clases Extremas", alpha = 0.5)

        ## ======================================================
        ## ANÁLISIS DE REGRESIÓN LINEAL ENTRE PARES DE FEATURES DE GIROS
        ## ======================================================

        ## Hago el análisis de regresión lineal tomando como referencia las features de mejor capacidad
        ## discriminadora según el análisis de error de SVM para giros de personas sin caidas mayores a 75
        no_fall_results = analyze_age_feature_dependencies(df_no_fall,
            svm_best_features = [results_svm_no_fall['feature'][0], results_svm_no_fall['feature'][1]],
            feature_cols = feature_cols, group_name = "no_fallers")

        ## Obtengo el ranking de mejores pares de features con mayores relaciones de dependencia lineal
        ## para giros correspondientes a personas mayores a 75 años
        raking_regr_no_fall = no_fall_results["no_fallers_older_75"].head(10)

        ## Hago la graficación de las mejores relaciones de regresión lineal para los no caedores
        ## para giros correspondientes a personas mayores a 75 años
        plot_regression_dependencies(raking_regr_no_fall, top_n = 10, 
            title = "Ranking de dependencias lineales - Personas mayores a 75 años sin caídas")

        ## Obtengo el ranking de mejores pares de features con mayores relaciones de dependencia lineal
        ## para giros correspondientes a personas menores a 75 años
        raking_regr_no_fall = no_fall_results["no_fallers_younger_75"].head(10)

        ## Hago la graficación de las mejores relaciones de regresión lineal para los no caedores
        ## para giros correspondientes a personas menores a 75 años
        plot_regression_dependencies(raking_regr_no_fall, top_n = 10, 
            title = "Ranking de dependencias lineales - Personas menores a 75 años sín caídas")

        ## Hago el análisis de regresión lineal tomando como referencia las features de mejor capacidad
        ## discriminadora según el análisis de error de SVM para giros de personas fallers mayores a 75
        fall_results = analyze_age_feature_dependencies(df_fall,
            svm_best_features = [results_svm_fall['feature'][0], results_svm_fall['feature'][1]],
            feature_cols = feature_cols, group_name = "fallers")

        ## Obtengo el ranking de mejores pares de features con mayores relaciones de dependencia lineal
        ## para giros correspondientes a personas mayores a 75 años
        raking_regr_fall = fall_results["fallers_older_75"].head(10)

        ## Hago la graficación de las mejores relaciones de regresión lineal para los caedores
        ## para giros correspondientes a personas mayores a 75 años
        plot_regression_dependencies(raking_regr_fall, top_n = 10, 
            title = "Ranking de dependencias lineales - Personas mayores a 75 años con al menos una caída")

        ## Obtengo el ranking de mejores pares de features con mayores relaciones de dependencia lineal
        ## para giros correspondientes a personas menores a 75 años
        raking_regr_fall = fall_results["fallers_younger_75"].head(10)

        ## Hago la graficación de las mejores relaciones de regresión lineal para los caedores
        ## para giros correspondientes a personas menores a 75 años
        plot_regression_dependencies(raking_regr_fall, top_n = 10, 
            title = "Ranking de dependencias lineales - Personas menores a 75 años con al menos una caída")

        ## ======================================================
        ## ANÁLISIS DE CLASIFICACIONES ERRÓNEAS EN BASE A LA ID DE LA PERSONA
        ## ======================================================

        ## Selecciono únicamente los giros clasificados erróneamente de 0 → 1 para los no caedores
        df_no_fall_fp = df_no_fall[df_no_fall["error_type"] == "young_to_old"].copy()

        ## Hago la distribución de IDs para los giros clasificados erróneamente de 0 → 1
        plot_id_distribution(df_no_fall_fp, group_col = "id",
            title = "No caedores - Distribución de IDs (young_to_old)")

        ## Selecciono únicamente los giros clasificados erróneamente de 1 → 0 para los no caedores
        ## (giros de personas mayores de edad mayor a 75 mal clasificados)
        df_no_fall_fn = df_no_fall[df_no_fall["error_type"] == "old_to_young"].copy()

        ## Hago la distribución de IDs para los giros clasificados erróneamente de 1 → 0
        plot_id_distribution(df_no_fall_fn, group_col = "id", 
            title = "No caedores - Distribución de IDs (old_to_young)")

        ## Selecciono únicamente los giros clasificados erróneamente de 0 → 1 para los caedores
        ## (giros de personas jovenes de edad menor a 75 mal clasificados)
        df_fall_fp = df_fall[df_fall["error_type"] == "young_to_old"].copy()

        ## Hago la distribución de IDs para los giros clasificados erróneamente de 0 → 1
        plot_id_distribution(df_fall_fp, group_col = "id",
            title = "Caedores - Distribución de IDs (young_to_old)")

        ## Selecciono únicamente los giros clasificados erróneamente de 1 → 0 para los caedores
        df_fall_fn = df_fall[df_fall["error_type"] == "old_to_young"].copy()

        ## Hago la distribución de IDs para los giros clasificados erróneamente de 1 → 0
        plot_id_distribution(df_fall_fn, group_col = "id",
            title = "Caedores - Distribución de IDs (old_to_young)")

        ## ======================================================
        ## ANÁLISIS DE OUTLIERS DISTINGUIENDO CAEDORES/NO CAEDORES
        ## ======================================================

        ## Hago la detección de outliers para el conjunto de los caedores
        outliers_fall, outliers_fall_by_feat = univariate_outliers(
            df_fall, [features_fall[0], features_fall[1]])

        ## Construyo el etiquetado de outliers para los caedores
        outliers_fall["outlier_label"] = outliers_fall["outlier_flag"].map(
            {True: "outlier", False: "normal"})

        ## Construyo un diccionario donde separo los outliers por feature
        outlier_dfs_fall = {feat: df_fall.loc[idx].copy()
            for feat, idx in outliers_fall_by_feat.items()}

        ## Hago el gráfico de distribución de IDs para la primera feature (fallers)
        plot_id_distribution(outlier_dfs_fall[features_fall[0]], group_col = "id",
            title = "Fallers - Distribución de IDs para {}".format(features_fall[0]))

        ## Hago el gráfico de distribución de IDs para la segunda feature (fallers)
        plot_id_distribution(outlier_dfs_fall[features_fall[1]], group_col = "id",
            title = "Fallers - Distribución de IDs para {}".format(features_fall[1]))

        ## Hago el gráfico de dispersión en el espacio de features diferenciando outliers/no outliers (fallers)
        plot_feature_space_2d(df = outliers_fall, feature_x = features_fall[0],
            feature_y = features_fall[1], target_col = "outlier_label", title = "Outliers Fallers")

        ## Hago la detección de outliers para el conjunto de los no caedores
        outliers_no_fall, outliers_no_fall_by_feat = univariate_outliers(df_no_fall, 
            [features_no_fall[0], features_no_fall[1]])

        ## Construyo el etiquetado de outliers para los no caedores
        outliers_no_fall["outlier_label"] = outliers_no_fall["outlier_flag"].map(
            {True: "outlier", False: "normal"})

        ## Construyo un diccionario donde separo los outliers por feature
        outlier_dfs_no_fall = {feat: df_no_fall.loc[idx].copy()
            for feat, idx in outliers_no_fall_by_feat.items()}

        ## Hago el gráfico de dispersión en el espacio de features diferenciando outliers/no outliers (no fallers)
        plot_feature_space_2d(df = outliers_no_fall, feature_x = features_no_fall[0],
            feature_y = features_no_fall[1], target_col = "outlier_label", title = "Outliers No Fallers")

        ## ======================================================
        ## ANÁLISIS DE FALLERS/NO FALLERS LUEGO DE REMOVER OUTLIERS
        ## ======================================================

        ## Elimino los outliers del conjunto de giros correspondientes a personas que tienen al menos una caída
        df_fall_soutliers = df_fall[df_fall["id"] != "255"].copy()

        ## SVM Univariado con división etaria binaria (>75, <75) para aquellos giros asociados
        ## a las personas para las que se registra al menos una caída por año
        results_svm_fall, svm_predictions_fall = rbf_svm_univariate_feature_error(
            df = df_fall_soutliers, feature_cols = feature_cols, target_col = "age_group_binary",
            C = C, gamma = gamma, n_splits = n_splits)

        ## Hago el SFFS Multivariado para los resultados de los giros asociados a personas con caidas
        sfs_results_fall = sfs_svm_fixed(df = df_fall_soutliers, feature_cols = feature_cols,
            target_col = "age_group_binary", k = k_top_features, C = C, gamma = gamma,
            cv = n_splits)

        ## Selecciono las 2 features óptimas según el proceso SFS con SVM 
        ## correspondientes al dataset de giros asociados a personas que tienen al menos una caida
        features_fall = sfs_results_fall.iloc[-1]["features"][:2]

        ## Hago el grafico de dispersión del feature space proyectado en el plano de las 2 mejores features
        plot_feature_space_2d(df = df_fall_soutliers, feature_x = "gyro_horz_jerk_energy_L2", 
            feature_y = "acc_horz_jerk_energy", target_col = "age_group_binary", 
            class_labels = {0: "≤75", 1: ">75"}, title = "Espacio de features de giros de caedores " \
            "luego de remover outliers", alpha = 0.5)

        ## ======================================================
        ## ANÁLISIS DE DIRECCIONES DOMINANTES USANDO PCA PARA FALLERS - NO FALLERS
        ## ======================================================

        ## Hago el análisis de PCA para fallers
        expl_fall, cum_fall, pca_fall = run_pca(df_fall, feature_cols)

        ## Hago el análsis de PCA para no fallers
        expl_nofall, cum_nofall, pca_nofall = run_pca(df_no_fall, feature_cols)

        ## Obtengo la distribución de las direcciones dominantes de PCA con respecto al feature set original
        ## para aquellos giros asociados a personas que tienen al menos una caída
        ## Porcentaje de contribuciones relativas de cada feature a las componentes principales
        load_fall = pc_loading_distributions(W = pca_fall.components_, feature_cols = feature_cols, K = 10)

        ## Obtengo la distribución de las direcciones dominantes de PCA con respecto al feature set original
        ## para aquellos giros asociados a personas que no tienen ninguna caída
        ## Porcentaje de contribuciones relativas de cada feature a las componentes principales
        load_nofall = pc_loading_distributions(W = pca_nofall.components_, feature_cols = feature_cols, K = 10)

        ## Hago la gráfica de las distribuciones de las componentes de PCA en el feature set original 
        ## para aquellos giros asociados a personas que tienen al menos una caída
        plot_pca_feature_contributions(load_fall, K = 10, N = 10, title = "Fallers - PCA loadings")

        ## Hago la gráfica de las distribuciones de las componentes de PCA en el feature set original 
        ## para aquellos giros asociados a personas que no tienen ninguna caída
        plot_pca_feature_contributions(load_nofall, K = 10, N = 10, title = "Non-fallers - PCA loadings")

        ## ======================================================
        ## ESTIMACIÓN DEL ERROR DE PREDICCIÓN EN 3 CLASES: JOVENES NO CAEDORES,
        ## MAYORES CAEDORES Y MAYORES NO CAEDORES -- AGRUPACIÓN ETARIA BINARIA + CAÍDAS
        ## ======================================================

        ## Hago una copia del dataset de giros con agrupación etaria y de riesgo de caídas
        df_turns = df_dataset_binary.copy()

        ## Construyo una nueva columna en la cual voy a agregar la clase del SVM a la que va a
        ## pertenecer cada giro (análisis de error en 3 clases)
        df_turns["svm_class"] = np.nan

        ## Asigno como clase 0 a los giros correspondientes a personas menores a 75 años
        df_turns.loc[df_turns["age_group"] == 0, "svm_class"] = 0

        ## Asigno como clase 1 a los giros correspondientes a aquellas personas mayores a 75 años
        ## las cuales no tienen registrada ninguna caída por año
        df_turns.loc[(df_turns["age_group"] == 1) & (df_turns["caida_bin"] == 0), "svm_class"] = 1

        ## Asigno como clase 1 a los giros correspondientes a aquellas personas mayores a 75 años
        ## las cuales tienen registrada al menos una caída por año
        df_turns.loc[(df_turns["age_group"] == 1) & (df_turns["caida_bin"] == 1), "svm_class"] = 2

        ## Elimino todos aquellos giros del dataset los cuales no estén asignados a ninguna clase
        df_turns = df_turns.dropna(subset = ["svm_class"])

        ## Me aseguro que el campo de la clase SVM a la que corresponde cada giro es de tipo entero
        df_turns["svm_class"] = df_turns["svm_class"].astype(int)

        ## Hago el ranking univariado de features correspondientes a esta segmentación de clases
        results_svm, svm_predictions = rbf_svm_univariate_feature_error(df = df_turns, 
            feature_cols = feature_cols, target_col = "svm_class", C = C, gamma = gamma, n_splits = n_splits)

        ## ======================================================
        ## CONSTRUCCIÓN Y GUARDADO DE GRÁFICOS
        ## ======================================================

        ## En caso de que quiera graficar y guardar los scatter plots combinando features dos a dos
        if graficar_scatter:

            ## Imprimo mensaje de aviso
            print("Generando gráficos de dispersión [...]")

            ## Construyo los diagramas de scatter con algunas de las features 1vs1 donde cada punto
            ## representa un giro y está pintado según el color del grupo etario al que pertenezca
            plot_turn_feature_pairs_by_age_group(df_plot, pairs, 
                "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/Scatter/".format(root), 
                FEATURE_NAMES, FEATURE_FILE_NAMES)

        ## En caso de que quiera graficar y guardar los boxplots de las features por rango etario
        if graficar_boxplots:

            ## Imprimo mensaje de aviso
            print("Generando gráficos de boxplot [...]")

            ## Hago la graficación de los boxplots con las features individualmente por el grupo etario
            ## y los guardo como imágenes en la carpeta de graficos correspondiente
            plot_turn_features_by_age_group(df_plot, feature_cols,
                "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/Boxplots/".format(root),
                FEATURE_NAMES, FEATURE_FILE_NAMES)

        ## En caso de que quiera graficar y guardar las matrices de significancia de Wilcoxon
        if graficar_wilcoxon:

            ## Imprimo mensaje de aviso
            print("Generando matrices de significación de Wilcoxon [...]")

            ## Grafico y guardo los resultados de aplicar el test de Wilcoxon dos a dos para 3 grupos etarios
            plot_wilcoxon_significance_matrices(results_wilcoxon, 
                        "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/Wilcoxon/TresGrupos"
                        .format(root), False, FEATURE_NAMES)

            ## Grafico y guardo los resultados de aplicar el test de Wilcoxon dos a dos para 2 grupos etarios
            ## y únicamente para personas que no hayan tenido ninguna caída por año
            plot_wilcoxon_significance_matrices(results_wilcoxon_no_fall, 
                        "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/Wilcoxon/DosGrupos/NoFallers"
                        .format(root), False, FEATURE_NAMES)

            ## Grafico y guardo los resultados de aplicar el test de Wilcoxon dos a dos para 2 grupos etarios
            ## y únicamente para personas que hayan tenido al menos una caída por año
            plot_wilcoxon_significance_matrices(results_wilcoxon_fall, 
                        "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/Wilcoxon/DosGrupos/Fallers"
                        .format(root), False, FEATURE_NAMES)

        ## En caso de que quiera graficar la variación de la feature en función de la edad junto
        ## con los correspondientes resultados de las regresiones lineal y polinomial
        if graficar_featvsedad:

            ## Imprimo mensaje de aviso
            print("Generando gráficos de features vs edad y regresión [...]")

            ## Hago los gráficos correspondientes y los guardo en la ruta de salida
            save_regression_plots(df = df_dataset, results_df = results_df, feature_cols = feature_cols,
                        output_dir = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/Regresion/"
                        .format(root), x_col = "Edad", only_significant = False)

        ## En caso de que quiera graficar las matrices de confusión usando un modelo de clasificación
        ## SVM con hiperparámetros dados y con un algoritmo K-Fold Cross Validation para validación
        if graficar_matconf:

            ## Imprimo mensaje de aviso
            print("Generando matrices de confusión [...]")

            ## Construyo matrices de confusión estilo Wilcoxon -- Tres Grupos Etarios
            plot_svm_univariate_confusion_matrices(predictions = svm_predictions, feature_cols = feature_cols,
            save_dir = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/Confusion/TresGrupos".format(root),
            C = C, gamma = gamma)

            ## Construyo matrices de confusión estilo Wilcoxon -- Dos Grupos Etarios
            plot_svm_univariate_confusion_matrices(predictions = svm_predictions_bin, 
            feature_cols = feature_cols, save_dir = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos"
            "/Confusion/DosGrupos".format(root), C = C, gamma = gamma)

    ## En caso de que quiera extender features ya existentes en el parquet (feature store update)
    elif opcion == 5:

        ## ======================================================
        ## CARGADO DEL DATASET DE GIROS EXISTENTE
        ## ======================================================

        ## Especifico la ruta del archivo .parquet donde tengo las features existentes para los giros
        input_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/features_giros_acc+gyro_v4." \
        "parquet".format(root)

        ## Hago el cargado de dicho dataframe con los datos previamente computados de los giros
        df_features = pd.read_parquet(input_path)

        ## Seteo el sistema inercial que voy a usar de referencia para el cálculo de orientación
        sist_inercial = 'ENU'

        ## Construyo una lista conteniendo los nombres de todas las features que voy a agregar
        axes = ["ax", "ay", "az", "wx", "wy", "wz"]

        ## Construyo la lista con las abreviaciones de las features que voy a agregar al dataset
        rwe_feats = ["jerk_burstiness","jerk_concentration","jerk_spectral_centroid"]

        ## Hago la concetenación de las abreviaciones con los nombres de los ejes
        feature_names = [f"{axis}_{feat}" for axis in axes for feat in rwe_feats]

        ## Hago una copia del dataframe de las features para no modificarlo directamente
        df_features = df_features.copy()

        ## Itero para cada uno de los nombres de las nuevas features que voy a agregar
        for name in feature_names:

            ## Genero un nuevo campo en el dataframe correspondiente a la nueva feature
            df_features[name] = np.nan

        ## Hago el agrupamiento del dataframe de features de giros según la ID de la persona
        grouped = df_features.groupby("id")

        ## ======================================================
        ## ITERACIÓN POR PACIENTE Y AGREGADO DE NUEVAS FEATURES
        ## ======================================================

        ## Itero para cada uno de los pacientes presentes en la base de datos
        for id_paciente, df_sub in grouped:

            ## Coloco un bloque try-except en caso de que ocurra algún error
            try:

                ## Despliego un mensaje indicando el paciente que estoy procesando
                print("Procesando giros del paciente de ID: {}".format(id_paciente))

                ## Hago la lectura de las mediciones de la IMU del individuo
                ## Las medidas del Shimmer3 vienen en m/s2 para el acelerómetro y grados/s para el giroscopio
                data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(
                id_persona = id_paciente, lectura_datos_propios = False, 
                ruta = '{}/sereData/sereData/Registros/MarchaLibre_Sabrina.txt'.format(root))

                ## Hago la conversión de los valores de velocidad angular de grados/s a rad/s
                gyro = np.deg2rad(gyro)

                ## Defino la frecuencia de muestreo del sistema
                frec_muestreo = 1 / periodoMuestreo

                ## Hago la estimación de la orientación del sistema de la IMU con respecto al sistema de referencia inercial
                imu_quat = estimar_orientacion_ekf(acel, gyro, frec_muestreo, sist_inercial)

                ## Hago la rotación de la señal del giroscopio del sistema de la IMU al sistema inercial
                ang_vel_inercial = rotate_body_to_world(gyro, imu_quat)

                ## Hago la rotación de la señal del acelerómetro del sistema de la IMU al sistema inercial
                acc_inercial = rotate_body_to_world(acel, imu_quat)

                ## Hago el suavizado de las señales del giroscopio expresadas en el sistema inercial
                ## con el fin de mitigar las excursiones significativas causadas por el ruido
                gyro_suav = np.column_stack([moving_average(ang_vel_inercial[:, 0], frec_muestreo),
                                        moving_average(ang_vel_inercial[:, 1], frec_muestreo),
                                        moving_average(ang_vel_inercial[:, 2], frec_muestreo)])
                
                ## Hago el suavizado de las señales del acelerómetro expresadas en el sistema inercial
                ## con el fin de mitigar las excursiones significativas causadas por el ruido
                acc_suav = np.column_stack([moving_average(acc_inercial[:, 0], frec_muestreo),
                                        moving_average(acc_inercial[:, 1], frec_muestreo),
                                        moving_average(acc_inercial[:, 2], frec_muestreo)])

                ## Obtengo un diccionario conteniendo las señales de acelerómetro, giroscopio
                ## y secuencia de cuaterniones correspondinetes al registro completo
                signals = {"gyro": gyro_suav, "acc": acc_suav, "quat": imu_quat}

                ## Construyo una estructura donde voy a almacenar información de las nuevas features
                new_values = {name: np.empty(len(df_sub)) for name in feature_names}

                ## Itero para cada uno de los giros detectados en el registro de la persona
                for i, (_, row) in enumerate(df_sub.iterrows()):

                    ## Obtengo los índices de comienzo y terminación del giro correspondiente
                    seg = {"start_idx": int(row["start_idx"]), "end_idx": int(row["end_idx"])}

                    ## Hago la segmentación de las señales en el intervalo en el que se produce el giro
                    data_seg = get_segment(signals, seg)

                    ## Obtengo el diccionario correspondiente con todas las features nuevas
                    feat_dict = feature_fn_nueva(data_seg, frec_muestreo)

                    ## Itero para cada uno de los nombres de las nuevas features agregadas
                    for name in feature_names:
                    
                        ## Agrego los nombres de las features nuevas al dataframe correspondiente
                        new_values[name][i] = feat_dict.get(name, np.nan)

                ## Itero para cada uno de los nombres de las nuevas features agregadas
                for name in feature_names:

                    ## Hago la concatenación del dataframe con las features nuevas y el original
                    df_features.loc[df_sub.index, name] = new_values[name]

            ## En caso de que ocurra algún error en el procesamiento
            except Exception as e:

                ## Imprimo el pantalla el número del paciente en el que se produjo el error
                print(f"Error en paciente {id_paciente}: {e}")

                ## Continúo con el procesamiento en el siguiente paciente
                continue

        ## ======================================================
        ## GUARDADO DEL DATASET MODIFICADO
        ## ======================================================

        ## Construyo la ruta de salida en la que voy a guardar el nuevo dataframe
        output_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/features_giros_acc+gyro_v5" \
                    ".parquet".format(root)

        ## Hago el guardado del nuevo dataframe en el archivo .parquet
        df_features.to_parquet(output_path, index = False)

        ## Imprimo en pantalla un mensaje avisando que la actualización con la(s) nuevas features fue exitoso
        print("Feature agregada y dataset guardado en:", output_path)