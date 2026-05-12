## Configuro nombre de la carpeta raíz
## (modificar si se utiliza en otro equipo)
root = "C:/Yo/Tesis"

## Importación de librerías
import sys
sys.path.append("{}/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica".format(root))
from LecturaDatosPacientes import *
from LecturaDatos import *
from Utils import *
import numpy as np

## Programa principal
if __name__== '__main__':

    ## Opcion 1: Graficar histograma caracterizando la distribución de edades de la población
    ## Opcion 2: Detectar y extraer features de los giros
    ## Opcion 3: Procesar features de giros previamente extraídas (análisis por persona)
    ## Opcion 4: Procesar features de giros previamente extraídas (análisis por giro)
    opcion = 3

    ## Obtengo la información correspondiente a todos los pacientes en la base de datos
    pacientes, ids_existentes = LecturaDatosPacientes()

    ## En caso de que quiera caracterizar la población según la edad
    if opcion == 1:

        ## Obtengo el histograma por edad
        graficar_histograma_edades(pacientes)
    
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

                ## Hago la rotación de la velocidad angular del sistema de la IMU al sistema inercial
                ang_vel_inercial = rotate_body_to_world(gyro, imu_quat)

                ## Hago el suavizado de la señal de velocidad angular en el eje vertical usando filtro de promedios
                ## móviles con el fin de remover picos no deseados de ruido en la señal
                wz_suav = moving_average(ang_vel_inercial[:,2], frec_muestreo)

                ## Hago la detección de los giros en base a la velocidad angular en el eje vertical
                giros = detect_turns_windowed(wz_suav, frec_muestreo)

                ## En caso de que quiera graficar velocidades angulares con los tramos de giro
                if graficar:

                    ## Hago la graficación de los tramos en los que se detectan giros de los que no
                    plot_signal_with_events(ang_vel_inercial[:,2], giros)

                ## Extraigo los segmentos de acelerómetros y giroscopios separados por giros
                segmentos = extraer_segmentos_giros(acel, gyro, giros)

                ## Hago la extracción de features correspondientes a los giros
                features_giros = extraer_features_basicas(imu_quat, segmentos, frec_muestreo, id_paciente, wz_suav)

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
        output_path = "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/features_giros_total.parquet".format(root)
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
        array_features = features_giros_total[["id", "duration_s", "angle_deg","mean_w_deg_s", "peak_w_deg_s", 
                                "rms_w_deg_s", "time_to_peak", "peak_mean_ratio"]]

        ## Genero un diccionario con los nombres de todos los features presentes (para graficación)
        feature_names = {"f0": "Duration (s)", "f1": "Angle (deg)", "f2": "Mean angular velocity (°/s)",
            "f3": "Peak angular velocity (°/s)", "f4": "RMS angular velocity (°/s)", "f5": "Time to peak (s)",
            "f6": "Peak/mean ratio"}
        
        ## Hago la sumarización de las features estadísticas de los giros por cada paciente        
        features_por_paciente = agrupar_por_paciente(array_features)

        ## Hago la conversión de numpy a dataframe para hacer el procesamiento
        df_features_por_paciente = pd.DataFrame(features_por_paciente)

        ## Me aseguro que los IDs se encuentran todos expresados en formato string
        df_features_por_paciente["id"] = df_features_por_paciente["id"].astype(str)
        pacientes["sampleid"] = pacientes["sampleid"].astype(str)

        ## Obtengo únicamente información de la edad y la ID asociada a cada paciente
        df_patients = pacientes[["sampleid", "Edad"]]

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
        "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Graficos/".format(root), feature_names)

    ## En caso de que yo quiera procesar las features de los giros que fueron previamente extraídas
    ## pero hago el procesamiento por cada giro separadamente sin hacer sumarización por persona
    elif opcion == 4:

        ## Hago la lectura del archivo .parquet donde guardo el dataframe Pandas que contiene
        ## la lista con los diccionarios con todos los parámetros de los giros detectados
        features_giros_total = pd.read_parquet(
            "{}/SL2205-0.8/SL2205-0.8/sereTestLib/Giros/Datos/features_giros_total.parquet".format(root))

        ## Especifico el orden de las columnas del dataframe según las features para asegurar consistencia
        ## y reorganizo todo el dataframe pandas para que sea consistente con el orden que quiero
        array_features = features_giros_total[["id", "duration_s", "angle_deg","mean_w_deg_s", "peak_w_deg_s", 
                                "rms_w_deg_s", "time_to_peak", "peak_mean_ratio"]]
        
        ## Hago la conversión del Dataframe de features por paciente a array numpy
        array_features = array_features.to_numpy()

        ## Extraigo el conjunto de las IDs de los pacientes correspondientes a todos los giros detectados
        ids = np.array([f["id"] for f in features_giros_total])

        ## Extraigo el conjunto de las velocidades angulares correspondientes a los giros detectados
        mean_w = np.array([f["mean_w_deg_s"] for f in features_giros_total])

        ## Obtengo las IDs y las edades de todos los pacientes para los cuales tengo registros
        patient_ids = np.array(pacientes["sampleid"])
        ages = np.array(pacientes["Edad"])

        ## Hago la asignación de cada uno de los pacientes al grupo etario correspondiente
        groups_patient = asignar_grupo_edad(ages)

        ## Hago la asignación de la ID de cada paciente con su correspondiente rango etario
        id_to_group = dict(zip(patient_ids, groups_patient))

        ## Por intermedio de la ID del paciente, hago la asignación de cada uno de los conjuntos de features
        ## de giros con el rango etario correspondiente
        groups = np.array([id_to_group[i] for i in ids])

        ## Hago la graficación del boxplot de las velocidades angulares medias de giro segmentadas por
        ## el rango etario definido
        data = [np.abs(mean_w[groups == g]) for g in [0, 1, 2]]
        plt.boxplot(data, tick_labels=["<60", "60-75", ">75"])
        plt.ylabel("Velocidad angular promedio de giro (°/s)")
        plt.title("Velocidad angular promedio de giro según rango etario")
        plt.grid(True, axis = "y")
        plt.show()