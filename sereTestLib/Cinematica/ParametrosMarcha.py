from Magnitud import Magnitud
from Normalizacion import Normalizacion
from DeteccionPicos import *
from Filtros import FiltroMediana
from Fourier import TransformadaFourier
import numpy as np
from Muestreo import PeriodoMuestreo
import pandas as pd
from ValoresMagnetometro import ValoresMagnetometro
from matplotlib import pyplot as plt
from scipy import signal
from harm_analysis import *
from control import *
from skinematics.imus import analytical, IMU_Base

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S278/3S278.csv"

## Lectura de datos
data = pd.read_csv(ruta)

## Hallo el período de muestreo de las señales
periodoMuestreo = PeriodoMuestreo(data)

## Armamos una matriz donde las columnas sean las aceleraciones
acel = np.array([np.array(data['AC_x']), np.array(data['AC_y']), np.array(data['AC_z'])]).transpose()

## Armamos una matriz donde las columnas sean los valores de los giros
gyro = np.array([np.array(data['GY_x']), np.array(data['GY_y']), np.array(data['GY_z'])]).transpose()

## Separo el vector de tiempos del dataframe
tiempo = np.array(data['Time'])

## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
tiempo = (tiempo - tiempo[0]) / 1000

## -------------------------------------- CORRECCIÓN DE EJES ----------------------------------

## Armamos el diccionario con los datos a ingresar
dict_datos = {'acc': acel, 'omega' : gyro, 'rate' : 1 / periodoMuestreo}

## Matriz de rotación inicial
## Es importante tener en cuenta la orientación inicial de la persona
orient_inicial = np.array([np.array([1,0,0]), np.array([0,0,1]), np.array([0,1,0])])

## Creacion de instancia IMU_Base
imu_analytic = IMU_Base(in_data = dict_datos, q_type = 'analytical', R_init = orient_inicial, calculate_position = False)

## Accedo a los cuaterniones resultantes
q_analytic = imu_analytic.quat

## Accedo a la velocidad resultante
vel_analytic = imu_analytic.vel

## Accedo a la posición resultante
pos_analytic = imu_analytic.pos

## Accedo a la aceleración corregida
acc_analytic = imu_analytic.accCorr

## ----------------------------------------- PREPROCESADO ------------------------------------------

## Se calcula la magnitud de la señal de aceleración
magnitud = Magnitud(acc_analytic)

## Se hace la normalización en amplitud y offset de la señal de magnitud
mag_normalizada = Normalizacion(magnitud)

## Se realiza un filtrado de medianas para eliminar algunos picos no relevantes de la señal
normal_filtrada = signal.medfilt(mag_normalizada, kernel_size = 11)

## ----------------------------------------- ANÁLISIS EN FRECUENCIA ---------------------------------------------------

## Defino la señal a procesar
señal = magnitud

## Se obtiene el espectro de toda la señal completa aplicando la transformada de Fourier
## Ya que es una señal real se cumple la simetría conjugada
(frecuencias, transformada) = TransformadaFourier(señal, periodoMuestreo)

## Obtengo información de los armónicos de la señal mediante el uso de <<harm_analysis>>
armonicos = harm_analysis(señal, FS = 1 / periodoMuestreo)

## Obtengo frecuencia fundamental de la señal
frec_fund = armonicos['fund_freq']

## Potencia de la componente fundamental de la señal
fund_db = armonicos['fund_db']

## Hago el pasaje del valor obtenido en decibeles a magnitud
fund_mag = db2mag(fund_db)

## Potencias de los armónicos no fundamentales
no_fund_db = np.array(armonicos['pot_armonicos'])

## Hago el pasaje de los valores obtenidos en decibeles a magnitud
no_fund_mag = db2mag(no_fund_db)

## ----------------------------------------- DETECCIÓN DE PICOS ------------------------------------------

## Cálculo de umbral óptimo
(T, stdT) = CalculoUmbral(señal = normal_filtrada)

## Se hace el llamado a la función de detección de picos configurando un umbral de entrada T
picos = DeteccionPicos(normal_filtrada, umbral = T)

## Cálculo del índice donde se encuentra ciclo potencial inicial
IPC = CicloPotencialInicial(picos)

## Se hace la graficación de la señal marcando los picos
GraficacionPicos(normal_filtrada, picos)

## Obtengo el vector con las separaciones de los picos
separaciones = SeparacionesPicos(picos)

## Hago la traduccion de muestras a valores de tiempo usando la frecuencia de muestreo dada
sep_tiempos = separaciones * periodoMuestreo

## Valor medio de la separación de tiempos
tiempo_medio = np.mean(sep_tiempos)

## Desviación estándar de la separación de tiempos
desv_tiempos = np.std(sep_tiempos)

## ----------------------------------------- GRÁFICAS ------------------------------------------

# plt.plot(acc_analytic[:,0], color = 'r', label = '$a_x$')
# plt.plot(acc_analytic[:,1], color = 'b', label = '$a_y$')
# plt.plot(acc_analytic[:,2], color = 'g', label = '$a_z$')

# ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
# plt.xlabel("Tiempo (s)")
# plt.ylabel("Aceleracion $(m/s^2)$")

# ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
# plt.legend()

# ## Despliego la gráfica
# plt.show()