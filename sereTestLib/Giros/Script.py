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

## Programa principal
if __name__== '__main__':

    ## Opcion 1: Graficar histograma caracterizando la distribución de edades de la población
    ## Opcion 2: Detectar y extraer features de los giros
    ## Opcion 3: Procesar features de giros previamente extraídas (análisis por persona)
    ## Opcion 4: Procesar features de giros previamente extraídas (análisis por giro)
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
        output_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/features_giros_acc+gyro.parquet".format(root)
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

        ## Configuro una variable que me de a elegir si quiero graficar boxplots de features
        graficar_boxplots = False

        ## Configuro una variable que me de a elegir si quiero graficar scatterplots de features
        graficar_scatter = False

        ## Configuro una variable que me de a elegir si quiero graficar matrices de significación
        ## del test de hipótesis de Wilcoxon
        graficar_wilcoxon = True

        ## Hago la lectura del archivo .parquet donde guardo el dataframe Pandas que contiene
        ## la lista con los diccionarios con todos los parámetros de los giros detectados
        features_giros_total = pd.read_parquet(
            "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/features_giros_acc+gyro.parquet".format(root))
        
        ## Me aseguro que el campo "id" del dataframe de features esté en formato string
        features_giros_total["id"] = features_giros_total["id"].astype(str)

        ## Selecciono y ordeno las columnas de features por giro para asegurar consistencia
        array_features = features_giros_total[["id"] + FEATURE_COLUMNS].copy()

        ## Obtengo únicamente información de la edad y la ID asociada a cada paciente
        df_patients = pacientes[["sampleid", "Edad"]].copy()

        ## Me aseguro que los IDs se encuentran todos expresados en formato string
        df_patients["sampleid"] = df_patients["sampleid"].astype(str)

        ## Asocio cada giro con la edad del sujeto correspondiente mediante un merge por ID,
        ## eliminando la columna redundante que contiene la ID de la persona a la que corresponde el giro
        df_dataset = array_features.merge(df_patients, left_on = "id", right_on = "sampleid", 
                                        how = "inner").drop(columns = ["sampleid"])

        ## Asigno cada persona al grupo etario correspondiente según su edad (0: edad < 60, 1: 
        ## 60 < edad < 75, 2: edad > 75) generando una nueva columna denominada "age_group"
        df_dataset["age_group"] = asignar_grupo_edad(df_dataset["Edad"])

        ## Elimino las columnas del dataframe que no corresponden a features numéricas de los giros
        feature_cols = df_dataset.columns.drop(["id", "Edad", "age_group"])

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

        ## En caso de que quiera graficar y guardar los scatter plots combinando features dos a dos
        if graficar_boxplots:

            ## Construyo los diagramas de scatter con algunas de las features 1vs1 donde cada punto
            ## representa un giro y está pintado según el color del grupo etario al que pertenezca
            plot_turn_feature_pairs_by_age_group(df_plot, pairs, 
                "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/Scatter/".format(root), 
                FEATURE_NAMES, FEATURE_FILE_NAMES)

        ## En caso de que quiera graficar y guardar los boxplots de las features por rango etario
        if graficar_scatter:

            ## Hago la graficación de los boxplots con las features individualmente por el grupo etario
            ## y los guardo como imágenes en la carpeta de graficos correspondiente
            plot_turn_features_by_age_group(df_plot, feature_cols,
                "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/Boxplots/".format(root),
                FEATURE_NAMES, FEATURE_FILE_NAMES)

        ## En caso de que quiera graficar y guardar las matrices de significancia de Wilcoxon
        if graficar_wilcoxon:

            ## Grafico y guardo los resultados de aplicar el test de Wilcoxon dos a dos
            plot_wilcoxon_significance_matrices(results_wilcoxon, 
                        "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/Wilcoxon/".format(root),
                        True, FEATURE_NAMES)