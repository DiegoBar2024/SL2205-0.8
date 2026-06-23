## Configuro nombre de la carpeta raíz
## (modificar si se utiliza en otro equipo)
root = "C:/Yo/Tesis"

## Importación de librerías
import sys
sys.path.append("{}/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica".format(root))
from LecturaDatosPacientes import *
from LecturaDatos import *
import numpy as np
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
                    plot_signal_with_events(ang_vel_inercial[:,2], giros)

                ## Extraigo los segmentos de acelerómetros y giroscopios separados por giros
                segmentos = extraer_segmentos_giros(acel, gyro, giros)

                ## Hago la extracción de features correspondientes a los giros, pasando como argumento
                ## señales tanto de acelerómetro como giroscopio preprocesadas
                features_giros = extraer_features_basicas(imu_quat, segmentos, frec_muestreo, 
                                                        id_paciente, gyro_suav, acc_suav)

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
        output_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/features_giros_acc+gyro+ind.parquet".format(root)
        df_features.to_parquet(output_path, index = False)

    ## En caso de que yo quiera procesar las features de los giros que fueron previamente extraídas
    ## pero haciendo una sumarización de los valores estadísticos de las features extraídas por persona
    elif opcion == 3:

        ## Hago la lectura del archivo .parquet donde guardo el dataframe Pandas que contiene
        ## la lista con los diccionarios con todos los parámetros de los giros detectados
        features_giros_total = pd.read_parquet(
            "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/features_giros_total.parquet".format(root))

        ## Especifico el orden de las columnas del dataframe según las features para asegurar consistencia
        ## y reorganizo todo el dataframe pandas para que sea consistente con el orden que quiero
        array_features = features_giros_total[["id"] + FEATURE_COLUMNS].copy()
        
        ## Hago la sumarización de las features estadísticas de los giros por cada paciente        
        features_por_paciente = agrupar_por_paciente(array_features)

        ## Hago la conversión de numpy a dataframe para hacer el procesamiento
        df_features_por_paciente = pd.DataFrame(features_por_paciente)

        ## Me aseguro que los IDs se encuentran todos expresados en formato string
        df_features_por_paciente["id"] = df_features_por_paciente["id"].astype(str)
        pacientes["sampleid"] = pacientes["sampleid"].astype(str)

        ## Obtengo únicamente información de la edad y la ID asociada a cada paciente
        df_patients = pacientes[["sampleid", "Edad"]].copy()

        ## Hago un inner join entre el dataframe con la edad y la ID de cada persona y el dataframe
        ## con la sumarización estadística de las features de todos los giros detectados para el paciente
        df_dataset = df_features_por_paciente.merge(df_patients, left_on = "id", right_on = "sampleid", 
                                                    how = "inner")

        ## Elimino la columna redundante con la ID del paciente para no tener datos duplicados
        df_dataset = df_dataset.drop(columns = ["id"])

        ## Asigno cada persona al grupo etario correspondiente según su edad (0: edad < 60, 1: 
        ## 60 < edad < 75, 2: edad > 75) generando una nueva columna denominada "age_group"
        df_dataset["age_group"] = asignar_grupo_edad(df_dataset["Edad"])

        ## Selecciono únicamente aquellas columnas que correspondan a los features
        feature_cols = df_dataset.columns.drop(["sampleid", "Edad", "age_group"])

        ## Genero los boxplots correspondientes a la distribución de la pobilación por edad según feature
        plot_feature_distributions_by_age_group(df_dataset, feature_cols, 'age_group', 
        "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/".format(root), FEATURE_NAMES)

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
            "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/features_giros_acc+gyro_v4.parquet".format(root))

        ## Hago la lectura de archivo .parquet donde tengo el historial óptimo de features con sfs
        sfs_features_results = pd.read_parquet(
            "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/sfs_features.parquet".format(root))

        ## ======================================================
        ## PREPROCESAMIENTO DE LOS DATOS (FEATURES EXTRAÍDAS)
        ## ======================================================

        ## Elimino aquellas columnas correspondientes a los índices que denotan el inicio y la
        ## terminación de cada uno de los giros (no voy a utilizarlas en esta parte del procesamiento)
        df_features = features_giros_total.drop(columns = ["start_idx", "end_idx"], errors = "ignore")

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
        array_features = features_giros_total[["id"] + FEATURE_COLUMNS].copy()

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

        # ## Hago la graficación de la distribución de rangos etarios por clúster y de clústers por rango etario
        # cluster_age, age_cluster = plot_cluster_age_distributions(cluster_results["df"])

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
        title = "Ranking univariado de features con SVM – clasificación por grupo etario",
        C = C, gamma = gamma)

        ## Obtengo un conjunto de las mejores features luego de hacer el SVM univariado
        top_features_univ = results_svm.nsmallest(5, "error")

        ## ======================================================
        ## RANKING MULTIVARIADO DE FEATURES USANDO SVM (SUPPORT VECTOR MACHINE)
        ## ======================================================

        ## Inicializo una variable que especifique la cantidad máxima de features con las que me quiero quedar
        k_top_features = 5

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

        ## SVM Univariado: Resultados al hacer los tests de clasificación univariada usando una SVM
        ## pero en este caso aplicado al problema de clasificación binaria con las clases como antes      
        results_svm_bin, svm_predictions_bin = rbf_svm_univariate_feature_error(df = df_dataset_binary, 
            feature_cols = feature_cols, target_col = "age_group_binary", C = C, gamma = gamma, 
            n_splits = n_splits)

        ## Hago la graficación de las features según el error de predicción medio en las K-Folds
        plot_svm_feature_error_ranking(results_svm_bin, top_k = 10, annotate = True,
        title = "Ranking univariado de features con SVM optimizado (división etaria binaria) " \
        "– dataset completo", C = C, gamma = gamma)

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
        ## VISUALIZACIÓN DEL ESPACIO DE FEATURES EN 2D (BINARIO)
        ## ======================================================

        ## Selecciono las dos features más discriminativas según SFFS o análisis previo
        feature_x = "wx_jerk_energy"
        feature_y = "wy_jerk_energy"

        ## Hago la visualización del espacio de features en 2D
        plot_feature_space_2d(df = df_dataset_binary, feature_x = feature_x,
            feature_y = feature_y, target_col = "age_group_binary", class_labels = {0: "≤75", 1: ">75"},
            title = "Espacio de features: energía de jerk en plano horizontal", alpha = 0.5)

        ## Selecciono las energias de las diferenciaciones de acelerómetro y giroscopio cuyos
        ## ejes se encuentran en el plano horizontal
        feature_x = "gyro_horz_jerk_energy"
        feature_y = "acc_horz_jerk_energy"

        ## Hago la visualización del espacio de features en 2D
        plot_feature_space_2d(df = df_dataset_binary, feature_x = feature_x,
            feature_y = feature_y, target_col = "age_group_binary", class_labels = {0: "≤75", 1: ">75"},
            title = "Espacio de features: energía de jerk en plano horizontal", alpha = 0.5)

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
        plot_svm_feature_error_ranking(results_svm_no_fall, top_k = 10, annotate = True,
            title = "Ranking univariado de features con SVM (división etaria binaria) – personas sin caídas",
            C = C, gamma = gamma)

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
            title = "Ranking univariado de features con SVM (división etaria binaria) – personas con " \
            "al menos una caída", C = C, gamma = gamma)

        ## Hago el SFFS Multivariado para los resultados de los giros asociados a personas con caidas
        sfs_results_fall = sfs_svm_fixed(df = df_fall, feature_cols = feature_cols,
            target_col = "age_group_binary", k = k_top_features, C = C, gamma = gamma,
            cv = n_splits)

        ## Obtengo un conjunto de las mejores features luego de hacer el SVM SFFS multivariado
        top_features_multiv_fall = sfs_results_fall['features'][k_top_features - 1]

        ## ======================================================
        ## VISUALIZACIÓN DEL ESPACIO DE FEATURES EN 2D
        ## SEPARADOS SEGÚN CAIDAS/NO CAIDAS
        ## ======================================================

        ## Selecciono las energías horizontales de jerk de acelerómetro y giroscopio
        feature_x = "gyro_horz_jerk_energy"
        feature_y = "acc_horz_jerk_energy"

        ## Hago la visualización en el espacio 2D de las features para sujetos sin caídas
        plot_feature_space_2d(df = df_no_fall, feature_x = feature_x, feature_y = feature_y, 
            target_col = "age_group_binary", class_labels = {0: "≤75", 1: ">75"},
            title = ("Espacio de features (sin caídas): energía de jerk horizontal"), alpha = 0.5)

        ## Hago la visualización en el espacio 2D de las features para sujetos con caídas
        plot_feature_space_2d(df = df_fall, feature_x = feature_x, feature_y = feature_y, 
            target_col = "age_group_binary", class_labels = {0: "≤75", 1: ">75"}, 
            title = ("Espacio de features (con caídas): energía de jerk horizontal"), alpha = 0.5)

        ## Hago la visualización en el espacio 2D de las features separando por color caidas/no caidas
        plot_feature_space_2d(df = df_dataset_binary, feature_x = feature_x, feature_y = feature_y,
            target_col = "caida_bin", class_labels = {0: "No caídas", 1: "Caídas"},
            title = "Espacio de features: energía de jerk en plano horizontal", alpha = 0.5)

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
        feature_cols = feature_cols, target_col = "svm_class", C = C, gamma = gamma, 
        n_splits = n_splits)

        ## Hago la graficación de las features según el error de predicción medio en las K-Folds
        plot_svm_feature_error_ranking(results_svm, top_k = 10, annotate = True,
        title = "Ranking univariado de características mediante SVM con kernel RBF para discriminación " \
        "de tres clases (edad y riesgo de caídas)", C = C, gamma = gamma)

        ## ======================================================
        ## RANKING SVM UNIVARIADO CON TUNEADO DE HIPERPARÁMETROS Y DIVISIÓN
        ## ETARIA BINARIA (DATASET COMPLETO Y LUEGO SEGMENTANDO CAIDAS/NO CAIDAS)
        ## ======================================================

        ## Hago el ranking de features de giros univariado para división etaria binaria con
        ## tuneado de hiperparámetros de SVM para todo el dataset completo (sin dividir caídas)
        results_svm_bin_tun = evaluar_features_svm_rbf(df = df_dataset_binary, feature_cols = feature_cols,
            target_col = "age_group_binary", cv = n_splits)

        ## Despliego el ranking de features correspondiente a la evaluación anterior de SVM + tuning
        plot_svm_feature_error_ranking(results_svm_bin_tun, top_k = 10, annotate = True,
        title = "Ranking univariado de features con SVM y ajuste de hiperparámetros (grupos etarios binarios)")

        ## Hago el ranking de features de giros univariado para división etaria binaria con tuneado de 
        ## hiperparámetros de SVM para los giros correspondientes a personas sin ninguna caída por año
        results_svm_bin_no_fall_tun = evaluar_features_svm_rbf(df = df_no_fall, feature_cols = feature_cols, 
            target_col = "age_group_binary", cv = n_splits)

        ## Despliego el ranking de features correspondiente a la evaluación anterior de SVM + tuning
        plot_svm_feature_error_ranking(results_svm_bin_no_fall_tun, top_k = 10, annotate = True,
        title = "Ranking univariado de features con SVM optimizado (C, γ) – grupo sin caídas")

        ## Hago el ranking de features de giros univariado para división etaria binaria con tuneado de 
        ## hiperparámetros de SVM para los giros correspondientes a personas con al menos una caída por año
        results_svm_bin_fall_tun = evaluar_features_svm_rbf(df = df_fall, feature_cols = feature_cols,
            target_col = "age_group_binary", cv = n_splits)

        ## Despliego el ranking de features correspondiente a la evaluación anterior de SVM + tuning
        plot_svm_feature_error_ranking(results_svm_bin_fall_tun, top_k = 10, annotate = True,
        title = "Ranking de features con SVM optimizado (división etaria binaria) – personas con al" \
        " menos una caída")

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
                        .format(root), True, FEATURE_NAMES)

            ## Grafico y guardo los resultados de aplicar el test de Wilcoxon dos a dos para 2 grupos etarios
            ## y únicamente para personas que no hayan tenido ninguna caída por año
            plot_wilcoxon_significance_matrices(results_wilcoxon_no_fall, 
                        "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/Wilcoxon/DosGrupos/NoFallers"
                        .format(root), True, FEATURE_NAMES)

            ## Grafico y guardo los resultados de aplicar el test de Wilcoxon dos a dos para 2 grupos etarios
            ## y únicamente para personas que hayan tenido al menos una caída por año
            plot_wilcoxon_significance_matrices(results_wilcoxon_fall, 
                        "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/Wilcoxon/DosGrupos/Fallers"
                        .format(root), True, FEATURE_NAMES)

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
        input_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/features_giros_acc+gyro_v3." \
        "parquet".format(root)

        ## Hago el cargado de dicho dataframe con los datos previamente computados de los giros
        df_features = pd.read_parquet(input_path)

        ## Seteo el sistema inercial que voy a usar de referencia para el cálculo de orientación
        sist_inercial = 'ENU'

        ## Construyo una lista conteniendo los nombres de todas las features que voy a agregar
        axes = ["ax", "ay", "az", "wx", "wy", "wz"]

        ## Construyo la lista con las abreviaciones de las features que voy a agregar al dataset
        rwe_feats = ["std","iqr","cv"]

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
        output_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/features_giros_acc+gyro_v4" \
                    ".parquet".format(root)

        ## Hago el guardado del nuevo dataframe en el archivo .parquet
        df_features.to_parquet(output_path, index = False)

        ## Imprimo en pantalla un mensaje avisando que la actualización con la(s) nuevas features fue exitoso
        print("Feature agregada y dataset guardado en:", output_path)