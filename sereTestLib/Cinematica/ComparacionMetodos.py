## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from ContactosIniciales import *
from ContactosTerminales import *
from ParametrosGaitPy import *
from LongitudPasoM1 import LongitudPasoM1
from LongitudPasoM2 import LongitudPasoM2
from SegmentacionGaitPy import Segmentacion as SegmentacionGaitPy
from Segmentacion import Segmentacion
import sys
import pandas as pd
from LecturaDatosPacientes import *
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/gaitpy/gaitpy')
from ParametrosGaitPy import *

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## La idea de ésta parte consiste en poder hacer una discriminación entre reposo y actividad
## Especifico la ruta en la cual se encuentra el registro a leer
ruta_registro = 'C:/Yo/Tesis/sereData/sereData/Registros/MarchaEstandar_Rodrigo.txt'

##  Hago la lectura de los datos
data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = 299, lectura_datos_propios = True, ruta = ruta_registro)

## ------------------------------------- CÁLCULO DE ESTADÍSTICAS ---------------------------------------

## MÉTODO I: ALGORITMO PROPIO
## Cálculo de contactos iniciales
contactos_iniciales, muestras_paso, acc_AP_norm, frec_fund = ContactosIniciales(acel, cant_muestras, periodoMuestreo, graficar = False)

## Cálculo de contactos terminales
contactos_terminales = ContactosTerminales(acel, cant_muestras, periodoMuestreo, graficar = False)

## Hago la segmentación de la marcha
pasos, duraciones_pasos, giros = Segmentacion(contactos_iniciales, contactos_terminales, muestras_paso, periodoMuestreo, acc_AP_norm, gyro)

## Cálculo de parámetros de marcha usando método de long paso I
pasos_numerados, frecuencias_m1, velocidades_m1, long_pasos_m1, coeficientes_m1 = LongitudPasoM1(pasos, acel, tiempo, periodoMuestreo, frec_fund, duraciones_pasos, 10, giros, long_pierna = 1.035)

## Cálculo de parámetros de marcha usando método de long paso II
pasos_numerados, frecuencias_m2, velocidades_m2, long_pasos_m2, coeficientes_m2 = LongitudPasoM2(pasos, acel, tiempo, periodoMuestreo, frec_fund, duraciones_pasos, 10, giros, long_pierna = 1.035, long_pie = 0.28)

## MÉTODO II: USANDO GAITPY
## Hago la lectura de los datos generales de los pacientes
pacientes, ids_existentes = LecturaDatosPacientes()

## Hago el cálculo de los parámetros de marcha usando GaitPy
features_gp, acc_VT = ParametrosMarcha(10, data, periodoMuestreo, pacientes)

## Hago la segmentación usando GaitPy
pasos_gp = SegmentacionGaitPy(features_gp, periodoMuestreo, acc_VT, plot = True)

## Cantidad de pasos
print("Pasos GP: {}".format(len(pasos_gp)))
print("Pasos metodo: {}".format(len(pasos)))

## Duración de pasos
print("\nDuracion GP media: {}".format(np.mean(np.array(features_gp['step_duration']))))
print("Duración metodo media: {}".format(np.mean(duraciones_pasos)))
print("Duración GP desv: {}".format(np.std(np.array(features_gp['step_duration']))))
print("Duración metodo desv: {}".format(np.std(duraciones_pasos)))

## Cadencia de pasos
print("\nCadencia GP media: {}".format(np.mean(np.array(features_gp['cadence'])) / 60))
print("Cadencia metodo media: {}".format(np.mean(frecuencias_m1)))
print("Cadencia GP desv: {}".format(np.std(np.array(features_gp['cadence'])) / 60))
print("Cadencia metodo desv: {}".format(np.std(frecuencias_m1)))

## Longitud de pasos
print("\nLongitud GP media: {}".format(np.mean(np.array(features_gp['step_length']))))
print("Longitud metodo media: {}".format(np.mean(long_pasos_m1)))
print("Longitud GP desv: {}".format(np.std(np.array(features_gp['step_length']))))
print("Longitud metodo desv: {}".format(np.std(long_pasos_m1)))

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