## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from ContactosIniciales import *
from ContactosTerminales import *
from ParametrosGaitPy import *
from LongitudPasoM1 import LongitudPasoM1
from SegmentacionGaitPy import Segmentacion as SegmentacionGaitPy
from Segmentacion import Segmentacion
import sys
import pandas as pd
from LecturaDatosPacientes import *
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/gaitpy/gaitpy')
from ParametrosGaitPy import *

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Ruta donde voy a abrir el archivo
ruta_registro = "C:/Yo/Tesis/sereData/sereData/Registros/MarchaEstandar_Rodrigo.txt"

## Abro el fichero correspondiente
fichero = open(ruta_registro, "r")

## Hago la lectura de todas las lineas correspondientes al fichero
lineas = fichero.readlines()

## Creo un array vacío en donde voy a guardar los datos
data = []

## Itero para todas aquellas lineas que tengan información útil
for linea in lineas[3:]:

    ## Hago la traducción de la línea de datos a una lista de numeros flotantes, segmentando la línea por tabulación
    lista_datos = list(map(float,linea.split("\t")[:-1]))

    ## Agrego la lista de datos como renglón de la matriz de datos
    data.append(lista_datos)

## Hago una lista con todos los headers de los datos tomados
headers = lineas[1].split("\t")[:-1]

## Hago el pasaje de los datos en forma de matriz a forma de dataframe
data = pd.DataFrame(data, columns = headers)

## Creo una lista con las columnas deseadas
columnas_deseadas = ['Time', 'AC_x', 'AC_y', 'AC_z', 'GY_x', 'GY_y', 'GY_z']

## Creo un diccionario con los nombres originales de las columnas y sus nombres nuevos
nombres_columnas = {'Timestamp': 'Time', 'Accel_LN_X_CAL' : 'AC_x', 'Accel_LN_Y_CAL' : 'AC_y', 'Accel_LN_Z_CAL' : 'AC_z'
                    ,'Gyro_X_CAL' : 'GY_x', 'Gyro_Y_CAL' : 'GY_y', 'Gyro_Z_CAL' : 'GY_z'}

## Itero para cada una de las columnas del dataframe
for columna in data.columns:

    ## Itero para cada uno de los nombres posibles
    for nombre in nombres_columnas.keys():

        ## En caso de que un nombre esté en la columna
        if nombre in columna:

            ## Renombro la columna
            data = data.rename(columns = {columna : nombres_columnas[nombre]})

## Selecciono las columnas deseadas
data = data[columnas_deseadas]

## Armamos una matriz donde las columnas sean las aceleraciones
acel = np.array([np.array(data['AC_x']), np.array(data['AC_y']), np.array(data['AC_z'])]).transpose()

## Armamos una matriz donde las columnas sean los valores de los giros
gyro = np.array([np.array(data['GY_x']), np.array(data['GY_y']), np.array(data['GY_z'])]).transpose()

## Separo el vector de tiempos del dataframe
tiempo = np.array(data['Time'])

## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
tiempo = (tiempo - tiempo[0]) / 1000

## Obtengo el período de muestreo
periodoMuestreo = PeriodoMuestreo(data)

## Obtengo la cantidad total de muestras
cant_muestras = len(tiempo)

## ------------------------------------- CÁLCULO DE ESTADÍSTICAS ---------------------------------------

## MÉTODO I: ALGORITMO PROPIO
## Cálculo de contactos iniciales
contactos_iniciales, muestras_paso, acc_AP_norm, frec_fund = ContactosIniciales(acel, cant_muestras, periodoMuestreo, graficar = True)

## Cálculo de contactos terminales
contactos_terminales = ContactosTerminales(acel, cant_muestras, periodoMuestreo, graficar = True)

## Hago la segmentación de la marcha
pasos, duraciones_pasos,giros = Segmentacion(contactos_iniciales, contactos_terminales, muestras_paso, periodoMuestreo, acc_AP_norm, gyro)

## Cálculo de parámetros de marcha
pasos_numerados, frecuencias, velocidades, long_pasos_m1, coeficientes_m1 = LongitudPasoM1(pasos, acel, tiempo, periodoMuestreo, frec_fund, duraciones_pasos, id_persona = 1)

# ## MÉTODO II: USANDO GAITPY
# ## Hago la lectura de los datos generales de los pacientes
# pacientes, ids_existentes = LecturaDatosPacientes()

# ## Hago el cálculo de los parámetros de marcha usando GaitPy
# features_gp, acc_VT = ParametrosMarcha(id_persona, data, periodoMuestreo, pacientes)

# ## Hago la segmentación usando GaitPy
# pasos_gp = SegmentacionGaitPy(features_gp, periodoMuestreo, acc_VT, plot = True)

# ## Cantidad de pasos
# print("Pasos GP: {}".format(len(pasos_gp)))
# print("Pasos metodo: {}".format(len(pasos)))

# ## Duración de pasos
# print("Duracion GP: {}".format(np.mean(np.array(features_gp['step_duration']))))
# print("Duración metodo: {}".format(np.mean(duraciones_pasos)))

# ## Cadencia de pasos
# print("Cadencia GP: {}".format(np.mean(np.array(features_gp['cadence'])) / 60))
# print("Cadencia metodo: {}".format(np.mean(frecuencias)))

# ## Longitud de pasos
# print("Longitud GP: {}".format(np.mean(np.array(features_gp['step_length']))))
# print("Longitud metodo: {}".format(np.mean(long_pasos_m1)))

print(giros)

plt.plot(gyro[:, 1])
plt.show()