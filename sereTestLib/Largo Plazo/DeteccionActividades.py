## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from LecturaDatos import *
from Muestreo import *
from LecturaDatosPacientes import *
import numpy as np
import pandas as pd
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from scipy import *
import json
import seaborn as sns
from scipy.stats import *
from sklearn.svm import SVC
from joblib import dump, load

## -------------------------------------- DETECCIÓN DE ACTIVIDADES ------------------------------------------

def DeteccionActividades(acel, tiempo, muestras_ventana, muestras_solapamiento, periodoMuestreo, cant_muestras, plot = False):

    ## ---------------------------------------- FILTRADO DE MEDIANA ----------------------------------------

    ## Aplico filtrado de mediana con una ventana de tamaño 3 a la aceleración mediolateral
    acc_ML = signal.medfilt(volume = acel[:,0], kernel_size = 3)

    ## Aplico filtrado de mediana con una ventana de tamaño 3 a la aceleración vertical
    acc_VT = signal.medfilt(volume = acel[:,1], kernel_size = 3)

    ## Aplico filtrado de mediana con una ventana de tamaño 3 a la aceleración anteroposterior
    acc_AP = signal.medfilt(volume = acel[:,2], kernel_size = 3)

    ## -------------------------------------- GRAFICACIÓN DE SEÑALES ---------------------------------------

    ## En caso de que quiera graficar
    if plot:

        ## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
        plt.plot(tiempo, acc_ML, color = 'r', label = 'Aceleración ML')
        plt.plot(tiempo, acc_VT, color = 'b', label = 'Aceleración VT')
        plt.plot(tiempo, acc_AP, color = 'g', label = 'Aceleración AP')

        ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Aceleracion $(m/s^2)$")

        ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
        plt.legend()

        ## Despliego la gráfica
        plt.show()

    ## ---------------------------------------- FILTRADO PASABAJOS -----------------------------------------

    ## Basado en la publicación "Implementation of a Real-Time Human Movement Classifier Using a Triaxial Accelerometer for Ambulatory Monitoring"
    ## Defino un filtro pasabajos IIR elíptico de tercer orden con frecuencia de corte 0.25Hz
    ## El filtro tiene 0.01dB ripple máximo en banda pasante y atenuación mínima de -100dB en banda supresora
    sos_iir = signal.iirfilter(3, 0.25, btype = 'lowpass', analog = False, rp = 0.01, rs = 100, ftype = 'ellip', output = 'sos', fs = 1 / periodoMuestreo)

    ## Se seleccionan las componentes de GA (Gravity Acceleration) haciendo un filtrado pasabajos a las señales
    ## Filtrado de la señal de aceleración Mediolateral
    acc_ML_GA = signal.sosfiltfilt(sos_iir, acc_ML)

    ## Filtrado de la señal de aceleración Vertical
    acc_VT_GA = signal.sosfiltfilt(sos_iir, acc_VT)

    ## Filtrado de la señal de aceleración Anteroposterior
    acc_AP_GA = signal.sosfiltfilt(sos_iir, acc_AP)

    ## -------------------------------------- GRAFICACIÓN DE SEÑALES ---------------------------------------

    ## En caso de que quiera graficar
    if plot:

        ## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
        plt.plot(tiempo, acc_ML_GA, color = 'r', label = 'Aceleración ML')
        plt.plot(tiempo, acc_VT_GA, color = 'b', label = 'Aceleración VT')
        plt.plot(tiempo, acc_AP_GA, color = 'g', label = 'Aceleración AP')

        ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Aceleracion $(m/s^2)$")

        ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
        plt.legend()

        ## Despliego la gráfica
        plt.show()

    ## --------------------------------- OBTENCIÓN DE COMPONENTES BA ---------------------------------------

    ## En base a las componentes de GA (Gravity Acceleration) en los tres ejes, se obtienen los componentes de BA (Body Acceleration) en los tres ejes
    ## Obtengo la componente de BA en dirección mediolateral
    acc_ML_BA = acc_ML - acc_ML_GA

    ## Obtengo la componente de BA en dirección vertical
    acc_VT_BA = acc_VT - acc_VT_GA 

    ## Obtengo la componente de BA en dirección anteroposterior
    acc_AP_BA = acc_AP - acc_AP_GA

    ## -------------------------------------- GRAFICACIÓN DE SEÑALES ---------------------------------------

    ## En caso de que quiera graficar
    if plot:

        ## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
        plt.plot(tiempo, acc_ML_BA, color = 'r', label = 'Aceleración ML')
        plt.plot(tiempo, acc_VT_BA, color = 'b', label = 'Aceleración VT')
        plt.plot(tiempo, acc_AP_BA, color = 'g', label = 'Aceleración AP')

        ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Aceleracion $(m/s^2)$")

        ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
        plt.legend()

        ## Despliego la gráfica
        plt.show()

    ## ------------------------------------------- SEGMENTACIÓN --------------------------------------------

    ## Defino la variable que contiene la posición de la muestra en la que estoy parado, la cual se inicializa en 0
    ubicacion_muestra = 0

    ## Genero un vector para guardar los valores de SMA computados
    vector_SMA = []

    ## Pregunto si ya terminó de recorrer todo el vector de muestras
    while ubicacion_muestra < cant_muestras:

        ## En caso que la última ventana no pueda tener el tamaño predefinido, la seteo manualmente
        if (cant_muestras - ubicacion_muestra < muestras_ventana):
        
            ## Me quedo con el segmento de la aceleración mediolateral (componente BA)
            segmento_ML = acc_ML_BA[ubicacion_muestra :]

            ## Me quedo con el segmento de la aceleración vertical (componente BA)
            segmento_VT = acc_VT_BA[ubicacion_muestra :]

            ## Me quedo con el segmento de la aceleración anteroposterior (componente BA)
            segmento_AP = acc_AP_BA[ubicacion_muestra :]
        
        ## En otro caso, digo que la ventana tenga el tamaño predefinido
        else:

            ## Me quedo con el segmento de la aceleración mediolateral (componente BA)
            segmento_ML = acc_ML_BA[ubicacion_muestra : ubicacion_muestra + muestras_ventana]

            ## Me quedo con el segmento de la aceleración vertical (componente BA)
            segmento_VT = acc_VT_BA[ubicacion_muestra : ubicacion_muestra + muestras_ventana]

            ## Me quedo con el segmento de la aceleración anteroposterior (componente BA)
            segmento_AP = acc_AP_BA[ubicacion_muestra : ubicacion_muestra + muestras_ventana]

        ## Hago el cálculo del SMA (Signal Magnitude Area) para dicha ventana siguiendo la definición
        SMA = (np.sum(np.abs(segmento_ML)) + np.sum(np.abs(segmento_VT)) + np.sum(np.abs(segmento_AP))) / (periodoMuestreo * muestras_ventana)

        ## Agrego el valor computado de SMA al vector correspondiente
        vector_SMA.append(SMA)

        ## Actualizo el valor de la ubicación de la muestra para que me guarde la posición en la que debe comenzar la siguiente ventana
        ubicacion_muestra += muestras_ventana - muestras_solapamiento

    ## --------------------------------------- GRAFICACIÓN VALORES SMA -------------------------------------

    ## En caso de que quiera graficar
    if plot:

        ## Gráfica de Scatter
        plt.scatter(x = np.arange(start = 0, stop = len(vector_SMA)), y = vector_SMA)
        plt.show()

        ## Gráfica de Boxplot
        plt.boxplot(vector_SMA)
        plt.show()

    ## Retorno el vector con los valores de SMA
    return vector_SMA

## Ejecución principal del programa
if __name__== '__main__':

    ## Opción 1: Generar los ficheros JSON con los SMA para cada actividad
    ## Opción 2: Calcular el valor del umbral óptimo según los valores de SMA en los archivos JSON
    ## Opción 3: Llevar a cabo la clasificación de actividades
    opcion = 3

    ## Hago la lectura de los datos generales de los pacientes
    pacientes, ids_existentes = LecturaDatosPacientes()

    ## Especifico el diccionario de actividades con sus identificadores
    actividades = {'Sentado':'1','Parado':'2','Caminando':'3','Escalera':'4'}

    ## Defino la cantidad de muestras de la ventana que voy a tomar
    muestras_ventana = 200

    ## Defino la cantidad de muestras de solapamiento entre ventanas
    muestras_solapamiento = 100

    ## En caso de que quiera generar y guardar la base de datos (es decir si tengo la opción 1)
    if opcion == 1:

        ## Itero para cada uno de los identificadores de los pacientes
        for id_persona in ids_existentes:
        
            ## Itero para cada una de las cuatro actividades que tengo detectada
            for actividad in list(actividades.keys()):

                ## Coloco un bloque try en caso de que ocurra algún error de procesamiento
                try:

                        ## Ruta del archivo
                        ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S{}/{}S{}.csv".format(id_persona, actividades[actividad], id_persona)

                        ## Selecciono las columnas deseadas
                        data = pd.read_csv(ruta)

                        ## Armamos una matriz donde las columnas sean las aceleraciones
                        acel = np.array([np.array(data['AC_x']), np.array(data['AC_y']), np.array(data['AC_z'])]).transpose()

                        ## Armamos una matriz donde las columnas sean los valores de los giros
                        gyro = np.array([np.array(data['GY_x']), np.array(data['GY_y']), np.array(data['GY_z'])]).transpose()

                        ## Separo el vector de tiempos del dataframe
                        tiempo = np.array(data['Time'])

                        ## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
                        tiempo = (tiempo - tiempo[0]) / 1000

                        ## Cantidad de muestras de la señal
                        cant_muestras = len(tiempo)

                        ## En caso de que no haya un período de muestreo bien definido debido al vector de tiempos de la entrada
                        if all([x == True for x in np.isnan(tiempo)]):

                            ## Asigno arbitrariamente una frecuencia de muestreo de 200Hz es decir período de muestreo de 0.005s
                            periodoMuestreo = 0.005

                        ## En caso de que el vector de tiempos contenga elementos numéricos
                        else:

                            ## Calculo el período de muestreo en base al vector correspondiente
                            periodoMuestreo = PeriodoMuestreo(data)

                        ## Hago el cálculo del vector de SMA para dicha persona
                        vector_SMA = DeteccionActividades(acel, tiempo, muestras_ventana, muestras_solapamiento, periodoMuestreo, cant_muestras)

                        ## Hago la lectura del archivo JSON previamente existente
                        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/SMA_{}.json".format(actividad), 'r') as openfile:

                            # Cargo el diccionario el cual va a ser un objeto JSON
                            vectores_SMAs = json.load(openfile)

                        ## Agrego en el diccionario los datos de SMAs el vector correspondiente al paciente actual
                        vectores_SMAs[str(id_persona)] = vector_SMA

                        ## Especifico la ruta del archivo JSON sobre la cual voy a reescribir
                        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/SMA_{}.json".format(actividad), "w") as outfile:

                            ## Escribo el diccionario actualizado
                            json.dump(vectores_SMAs, outfile)

                ## Si hay un error de procesamiento
                except:

                    ## Que siga a la siguiente muestra
                    continue

    ## En caso de que quiera procesar los JSON de SMA para poder calcular el umbral de clasificación (opción 2)
    elif opcion == 2:

        ## Hago la lectura del archivo JSON previamente existente
        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/SMA_Parado.json", 'r') as openfile:

            # Cargo el diccionario el cual va a ser un objeto JSON
            SMA_parado = json.load(openfile)
        
        ## Hago la lectura del archivo JSON previamente existente
        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/SMA_Sentado.json", 'r') as openfile:

            # Cargo el diccionario el cual va a ser un objeto JSON
            SMA_sentado = json.load(openfile)
        
        ## Hago la lectura del archivo JSON previamente existente
        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/SMA_Caminando.json", 'r') as openfile:

            # Cargo el diccionario el cual va a ser un objeto JSON
            SMA_caminando = json.load(openfile)
        
        ## Hago la lectura del archivo JSON previamente existente
        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/SMA_Escalera.json", 'r') as openfile:

            # Cargo el diccionario el cual va a ser un objeto JSON
            SMA_escaleras = json.load(openfile)
        
        ## Creo un vector donde voy a guardar todos los valores de SMA de registros parado
        valores_SMA_parado = []

        ## Creo un vector donde voy a guardar todos los valores de SMA de registros sentado
        valores_SMA_sentado = []
        
        ## Creo un vector donde voy a guardar todos los valores de SMA de registros caminando
        valores_SMA_caminando = []
        
        ## Creo un vector donde voy a guardar todos los valores de SMA de registros de escaleras
        valores_SMA_escaleras = []

        ## Itero para cada una de las claves en el diccionario de SMA parado
        for id in list(SMA_parado.keys()):

            ## Concateno los SMAs del registro parado a la lista actual
            valores_SMA_parado += SMA_parado[id]
        
        ## Itero para cada una de las claves en el diccionario de SMA sentado
        for id in list(SMA_sentado.keys()):

            ## Concateno los SMAs del registro sentado a la lista actual
            valores_SMA_sentado += SMA_sentado[id]

        ## Itero para cada una de las claves en el diccionario de SMA caminando
        for id in list(SMA_caminando.keys()):

            ## Concateno los SMAs del registro caminando a la lista actual
            valores_SMA_caminando += SMA_caminando[id]

        ## Itero para cada una de las claves en el diccionario de SMA parado
        for id in list(SMA_escaleras.keys()):

            ## Concateno los SMAs del registro parado a la lista actual
            valores_SMA_escaleras += SMA_escaleras[id]
        
        ## Defino el diccionario con los correspondientes valores de SMA para cada actividad para luego hacer el boxplot
        valores_SMA = {'Parado': valores_SMA_parado, 'Sentado': valores_SMA_sentado, 'Caminando': valores_SMA_caminando, 'Escaleras': valores_SMA_escaleras}

        ## Hago el boxplot comparando los valores de SMA para las actividades consideradas
        plt.boxplot(valores_SMA.values(), tick_labels = valores_SMA.keys())
        plt.show()

        ## Defino un vector con todos los valores de SMA correspondientes a movimiento (caminando y escaleras)
        SMA_movimiento = valores_SMA_caminando + valores_SMA_escaleras

        ## Defino un vector con todos los valores de SMA correspondientes a reposo (sentado y parado)
        SMA_reposo = valores_SMA_sentado + valores_SMA_parado

        ## Construyo un diccionario diferenciando los valores de SMA para movimiento y reposo
        valores_SMA_actividad = {'Movimiento' : SMA_movimiento, 'Reposo' : SMA_reposo}

        ## Hago el boxplot comparando los valores de SMA para las actividades de movimiento y reposo
        plt.boxplot(valores_SMA_actividad.values(), tick_labels = valores_SMA_actividad.keys())
        plt.show()

        ## Test de Hipótesis de Anderson-Darling para la comprobación de normalidad de los valores de SMA en reposo
        anderson_reposo = anderson(SMA_reposo, dist='norm')

        ## Test de Hipótesis de Anderson-Darling para la comprobación de normalidad de los valores de SMA en movimiento
        anderson_mov = anderson(SMA_movimiento, dist='norm')

        ## Test de Hipótesis para la comprobación de la igualdad de las medianas entre ambas poblaciones
        test_medianas = kruskal(SMA_movimiento, SMA_reposo)

        ## Hago el histograma comparando los valores de SMA para las actividades de movimiento (caminando y escaleras)
        plt.hist(SMA_movimiento)
        plt.xlabel('Valor SMA')
        plt.ylabel('Frecuencia absoluta')
        plt.show()

        ## Hago el histograma comparando los valores de SMA para las actividades de movimiento (sentado y parado)
        plt.hist(SMA_reposo, bins = 10)
        plt.xlabel('Valor SMA')
        plt.ylabel('Frecuencia absoluta')
        plt.gca().set_xlim((0, 100))
        plt.show()

        ## ---------------------------------- MÉTODO I -----------------------------------------
        ## El primer método de cálculo de umbral óptimo involucra el uso de estadísticas provenientes de la muestra
        ## Construyo un vector de indicadores estadísticos para los valores de SMA en movimiento
        stats_SMA_mov = [np.mean(SMA_movimiento), np.median(SMA_movimiento), np.std(SMA_movimiento)]

        ## Construyo un vector de indicadores estadísticos para los valores de SMA en reposo
        stats_SMA_rep = [np.mean(SMA_reposo), np.median(SMA_reposo), np.std(SMA_reposo)]

        ## El criterio que tomo para definir el umbral de SMA es el siguiente:
        ## Sea a = Media(SMA_movimiento) - Desv(SMA_movimiento)
        ## Sea b = Media(SMA_reposo) + Desv(SMA_reposo)
        ## Entonces el valor del umbral lo ajusto al punto medio de a y b para que quede margen: umbral = (a + b) / 2
        umbral = (stats_SMA_mov[0] - stats_SMA_mov[2] + stats_SMA_rep[0] + stats_SMA_rep[2]) / 2

        ## ---------------------------------- MÉTODO II -----------------------------------------
        ## El segundo método de cálculo de umbral óptimo involucra el entrenamiento de un SVM con los datos etiquetados por actividad
        ## Construyo la Support Vector Machine
        clf = SVC(C = 1, gamma = 1, kernel = 'rbf')

        ## Llevo a cabo el entrenamiento del clasificador
        ## <<values>> es la secuencia de valores de entrada
        ## <<ground_truth>> es la secuencia de valores de salida
        ## Se etiquetan como 0 las actividades de reposo mientras que se etiquetan como 1 las actividades de movimiento
        clf.fit(np.array((SMA_movimiento + SMA_reposo)).reshape(-1, 1), np.concatenate((np.ones(len(SMA_movimiento)), np.zeros(len(SMA_reposo)))))

        ## Guardo el modelo entrenado en la ruta de salida
        dump(clf, "C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/SVM.joblib")

        ## Hago la lectura del archivo JSON previamente existente
        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/Umbrales.json", 'r') as openfile:

            # Cargo el valor existente del JSON
            dicc_umbral = json.load(openfile)
        
        ## Genero el diccionario conteniendo el valor del umbral calculado
        ## La clave estará generada por la cantidad de muestras por ventana y la cantidad de muestras de solapamiento
        dicc_umbral = {'U_{}_{}'.format(muestras_ventana, muestras_solapamiento): umbral}
        
        ## Especifico la ruta del archivo JSON sobre la cual voy a reescribir
        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/Umbrales.json", "w") as outfile:

            ## Escribo el diccionario actualizado
            json.dump(dicc_umbral, outfile)
        
    ## En caso de que quiera realizar la clasificación
    else:

        ## Hago la lectura del archivo JSON previamente existente
        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/Umbrales.json", 'r') as openfile:

            # Cargo el valor existente del JSON
            dicc_umbral = json.load(openfile)

        ## Especifico actividad de analisis
        actividad = 'Caminando'

        ## Especifico el ID de la persona
        id_persona = 37

        ## Ruta del archivo
        ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S{}/{}S{}.csv".format(id_persona, actividades[actividad], id_persona)

        ## Selecciono las columnas deseadas
        data = pd.read_csv(ruta)

        ## Armamos una matriz donde las columnas sean las aceleraciones
        acel = np.array([np.array(data['AC_x']), np.array(data['AC_y']), np.array(data['AC_z'])]).transpose()

        ## Armamos una matriz donde las columnas sean los valores de los giros
        gyro = np.array([np.array(data['GY_x']), np.array(data['GY_y']), np.array(data['GY_z'])]).transpose()

        ## Separo el vector de tiempos del dataframe
        tiempo = np.array(data['Time'])

        ## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
        tiempo = (tiempo - tiempo[0]) / 1000

        ## Cantidad de muestras de la señal
        cant_muestras = len(tiempo)

        ## En caso de que no haya un período de muestreo bien definido debido al vector de tiempos de la entrada
        if all([x == True for x in np.isnan(tiempo)]):

            ## Asigno arbitrariamente una frecuencia de muestreo de 200Hz es decir período de muestreo de 0.005s
            periodoMuestreo = 0.005

        ## En caso de que el vector de tiempos contenga elementos numéricos
        else:

            ## Calculo el período de muestreo en base al vector correspondiente
            periodoMuestreo = PeriodoMuestreo(data)
        
        ## Hago el cálculo del vector de SMA para dicha persona
        vector_SMA = DeteccionActividades(acel, tiempo, muestras_ventana, muestras_solapamiento, periodoMuestreo, cant_muestras)

        ## ---------------------------------- MÉTODO I -----------------------------------------
        ## Obtengo el valor del umbral para comparar
        umbral = dicc_umbral['U_{}_{}'.format(muestras_ventana, muestras_solapamiento)]

        ## Obtengo un vector con el resultado de la clasificación para el registro disponible de la actividad correspondiente
        ## El i-ésimo valor de dicho vector va a estar dado por:
        ## <<True>> si el i-ésimo segmento del registro se asocia con movimiento (caminar, subir/bajar escaleras)
        ## <<False>> si el i-ésimo segmento del registro se asocia con reposo (parado, sentado)
        clas_registro = np.array([vector_SMA]) > umbral

        ## ---------------------------------- MÉTODO II -----------------------------------------
        ## Cargo el modelo del clasificador ya entrenado según la ruta del clasificador
        clf_trained = load("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/SVM.joblib")

        ## Determino la predicción del clasificador ante mi muestra de entrada
        ## Etiqueta 0: Reposo
        ## Etiqueta 1: Movimiento
        pat_predictions = clf_trained.predict(np.array((vector_SMA)).reshape(-1, 1))