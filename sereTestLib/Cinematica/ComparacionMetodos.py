## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from ContactosIniciales import *
from ContactosTerminales import *
from ParametrosGaitPy import *
from LongitudPasoM1 import LongitudPasoM1
from LongitudPasoM2 import LongitudPasoM2
from SegmentacionGaitPy import Segmentacion as SegmentacionGaitPy
from Segmentacion import Segmentacion
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/gaitpy/gaitpy')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo')
from parameters import *
import pandas as pd
from LecturaDatosPacientes import *
from ParametrosGaitPy import *
from DeteccionActividades import DeteccionActividades
from EliminacionGirosTransitorios import *
from joblib import load

## Función que me calcula los parámetros de marcha usando el algoritmo propio
def AlgoritmoPropio(acel, gyro, cant_muestras, periodoMuestreo, tiempo,  graficarGiros = True):

    ## Especifico la cantidad de muestras que va a tener la ventana de analisis
    muestras_ventana = 100

    ## Especifico la cantidad de muestras de solapamiento entre ventanas
    muestras_solapamiento = 50

    ## Hago el cálculo del vector de SMA para dicha persona
    vector_SMA, features, ventanas, nombres_features = DeteccionActividades(acel, tiempo, muestras_ventana, muestras_solapamiento, periodoMuestreo, cant_muestras, actividad = None, CalcFeatures = False)

    ## Cargo el modelo del clasificador ya entrenado según la ruta del clasificador
    clf_entrenado = load(ruta_SVM)

    ## Determino la predicción del clasificador ante mi muestra de entrada
    ## Etiqueta 0: Reposo
    ## Etiqueta 1: Movimiento
    pat_predictions = clf_entrenado.predict(np.array((vector_SMA)).reshape(-1, 1))

    ## Me quedo sólo con las ventanas en las cuales se ha detectado marcha
    ventanas_marcha = ventanas[np.where(pat_predictions == 1)]

    ## Tomo la hipótesis de que el reposo puede estar al inicio y al final únicamente
    ## Selecciono entonces el tramo de aceleración en la cual se ha detectado marcha
    acel_marcha = acel[ventanas_marcha[0][0] : ventanas_marcha[-1][1], :]

    ## Selecciono entonces el tramo de giroscopios en la cual se ha detectado marcha
    gyro_marcha = gyro[ventanas_marcha[0][0] : ventanas_marcha[-1][1], :]

    ## Obtengo la cantidad total de muestras únicamente del tramo de marcha
    cant_muestras_marcha = acel_marcha.shape[0]

    ## Obtengo el vector de tiempos correspondiente durante el tramo de marcha
    tiempo_marcha = np.arange(start = 0, stop = cant_muestras_marcha * periodoMuestreo, step = periodoMuestreo)

    ## Cálculo de contactos iniciales
    contactos_iniciales, muestras_paso, acc_AP_norm, frec_fund = ContactosIniciales(acel_marcha, cant_muestras_marcha, periodoMuestreo, graficar = False)

    ## Cálculo de contactos terminales
    contactos_terminales = ContactosTerminales(acel_marcha, cant_muestras_marcha, periodoMuestreo, graficar = False)

    ## Hago la segmentación de la marcha
    pasos, duraciones_pasos, giros = Segmentacion(contactos_iniciales, contactos_terminales, muestras_paso, periodoMuestreo, acc_AP_norm, gyro_marcha)

    ## En caso de que quera graficar giros
    if graficarGiros:

        ## Hago la conversión de los giros a vector numpy para hacer la graficación
        giros = np.array(giros)

        ## Itero para cada uno de los pasos detectados
        for i in range (1, len(pasos)):

            ## En caso de que en dicho paso se haya detectado un giro
            if (pasos[i]['IC'][0] in giros[:, 0]) or (pasos[i]['IC'][1] in giros[:, 1]):

                plt.plot(tiempo[pasos[i]['IC'][0] - 1 : pasos[i]['IC'][1]],
                        gyro[:, 1][pasos[i]['IC'][0] - 1 : pasos[i]['IC'][1]], color = 'r', label = 'Giros')

            ## En caso de que en dicho tramo no se haya detectado un giro
            else:

                plt.plot(tiempo[pasos[i]['IC'][0] - 1 : pasos[i]['IC'][1]],
                        gyro[:, 1][pasos[i]['IC'][0] - 1: pasos[i]['IC'][1]], color = 'b', label = 'No giros')

        ## Despliego la gráfica y configuro los parámetros de Leyendas
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.xlabel('Tiempo(s)')
        plt.ylabel("Velocidad Angular Eje Vertical $(rad/s)$")
        plt.legend(by_label.values(), by_label.keys())
        plt.title('Deteccion Giros')
        plt.show()

    ## Elimino giros y transitorios donde paso como parámetro la cantidad de pasos que tiene mi transitorio
    pasos, duraciones_pasos = EliminarGirosTransitorios(pasos, duraciones_pasos, giros, cant_pasos_transitorio = 0)

    ## Cálculo de parámetros de marcha usando el método I
    pasos_numerados, frecuencias, velocidades, long_pasos_m1, coeficientes_m1 = LongitudPasoM1(pasos, acel_marcha, tiempo_marcha, periodoMuestreo, frec_fund, duraciones_pasos, 'R_SinGiros', giros, long_pierna = 1.035)

    ## Retorno los parámetros de marcha calculados
    return pasos_numerados, frecuencias, velocidades, long_pasos_m1, duraciones_pasos, giros

## Función que haga el cálculo de los parámetros de marcha usando el algoritmo de GaitPy
def AlgoritmoGaitPy(data):

    ## Obtengo el conjunto de todos los pacientes ingresados en la base de datos
    pacientes, ids_existentes = LecturaDatosPacientes()

    ## Hago el cálculo de los parámetros de marcha usando GaitPy
    features_gp, acc_VT = ParametrosMarcha(10, data, periodoMuestreo, pacientes)

    ## Hago la segmentación usando GaitPy
    pasos_gp = SegmentacionGaitPy(features_gp, periodoMuestreo, acc_VT, plot = False)

    ## Retorno las features de GaitPy y los pasos segmentados
    return pasos_gp, features_gp

## Ejecución principal del programa
if __name__== '__main__':

    ## Especifico la ruta en la cual se encuentra el registro a leer
    ruta_lectura = ruta_registro + 'MarchaLibre_Rodrigo.txt'

    ## Hago la lectura del registro de datos asociados a la persona
    data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = None, lectura_datos_propios = True, ruta = ruta_lectura)

    ## Ejecución del algoritmo propio
    pasos_numerados, frecuencias, velocidades, long_pasos_m1, duraciones_pasos, giros = AlgoritmoPropio(acel, gyro, cant_muestras, periodoMuestreo, tiempo)

    ## Ejecución del algoritmo de GaitPy
    pasos_gp, features_gp = AlgoritmoGaitPy(data)

    ## Cantidad de pasos
    print("Pasos GP: {}".format(len(pasos_gp)))
    print("Pasos metodo: {}".format(len(pasos_numerados)))

    ## Duración de pasos
    print("\nDuracion GP media: {}".format(np.mean(np.array(features_gp['step_duration']))))
    print("Duración metodo media: {}".format(np.mean(duraciones_pasos)))
    print("Duración GP desv: {}".format(np.std(np.array(features_gp['step_duration']))))
    print("Duración metodo desv: {}".format(np.std(duraciones_pasos)))

    ## Cadencia de pasos
    print("\nCadencia GP media: {}".format(np.mean(np.array(features_gp['cadence'])) / 60))
    print("Cadencia metodo media: {}".format(np.mean(frecuencias)))
    print("Cadencia GP desv: {}".format(np.std(np.array(features_gp['cadence'])) / 60))
    print("Cadencia metodo desv: {}".format(np.std(frecuencias)))

    ## Longitud de pasos
    print("\nLongitud GP media: {}".format(np.mean(np.array(features_gp['step_length']))))
    print("Longitud metodo media: {}".format(np.mean(long_pasos_m1)))
    print("Longitud GP desv: {}".format(np.std(np.array(features_gp['step_length']))))
    print("Longitud metodo desv: {}".format(np.std(long_pasos_m1)))