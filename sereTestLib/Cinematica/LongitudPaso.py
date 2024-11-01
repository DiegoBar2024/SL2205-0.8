import skinematics.quat
from skinematics.imus import analytical, IMU_Base
from skinematics.sensors import xsens
from skinematics.quat import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Integracion import CalcularVelocidades
from Fourier import TransformadaFourier
from ValoresMagnetometro import ValoresMagnetometro
from Muestreo import PeriodoMuestreo
from scipy import constants
from skinematics.sensors.xsens import XSens
import numpy as np
from scipy import integrate
from Filtros import FiltroMediana
from scipy.integrate import cumulative_trapezoid, simpson
from Magnitud import *
from Normalizacion import *
from DeteccionPicos import * 

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S299/3S299.csv"

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

## ------------------------------------------ CORRECCIÓN DE EJES ------------------------------------

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

## Accedo a la acleración del sensor luego de restar la gravedad
accSens_analytic = imu_analytic.accSens

## ----------------------------------------- PREPROCESADO ------------------------------------------

## Se calcula la magnitud de la señal de aceleración
magnitud = Magnitud(acc_analytic)

## Se hace la normalización en amplitud y offset de la señal de magnitud
mag_normalizada = Normalizacion(magnitud)

## Se realiza un filtrado de medianas para eliminar algunos picos no relevantes de la señal
normal_filtrada = signal.medfilt(mag_normalizada, kernel_size = 11)

## ------------------------------------------------ INTEGRACIÓN --------------------------------------------------------

## Integro aceleración para obtener velocidad
vel_z = cumulative_trapezoid(acc_analytic[:,2], dx = periodoMuestreo, initial = 0)
#vel_z = cumulative_trapezoid(acel[:,1] - constants.g, dx = periodoMuestreo, initial = 0)

## Integro velocidad para obtener posición
pos_z = cumulative_trapezoid(vel_z, dx = periodoMuestreo, initial = 0)

## ----------------------------------------- DETECCIÓN DE PICOS ------------------------------------------

## Cálculo de umbral óptimo
(T, stdT) = CalculoUmbral(señal = mag_normalizada)

## Se hace el llamado a la función de detección de picos configurando un umbral de entrada T
picos = DeteccionPicos(mag_normalizada, umbral = T)

## Se hace la graficación de la señal marcando los picos
GraficacionPicos(mag_normalizada, picos)

## Obtengo el vector con las separaciones de los picos
separaciones = SeparacionesPicos(picos)

## Hago la traduccion de muestras a valores de tiempo usando la frecuencia de muestreo dada
sep_tiempos = separaciones * periodoMuestreo

## Valor medio de la separación de tiempos
tiempo_medio = np.mean(sep_tiempos)

## Desviación estándar de la separación de tiempos
desv_tiempos = np.std(sep_tiempos)

## ------------------------------------------------ FILTRADO --------------------------------------------------------

## Con el fin de eliminar la deriva hago una etapa de filtrado pasaaltos
## Etapa de filtrado pasaaltos de Butterworth con frecuencia de corte 0.1Hz de orden 4
sos = signal.butter(N = 4, Wn = 0.1, btype = 'highpass', fs = 1 / periodoMuestreo, output = 'sos')

## Calculo la posición en el eje vertical luego de hacer el filtrado
## La cantidad de tiempo que transcurre entre dos valles debe ser igual al tiempo de paso
pos_z_filtrada = signal.sosfilt(sos, pos_z)

## ----------------------------------------- SEGMENTACIÓN ------------------------------------------

## Creo una lista vacía en donde voy a guardar los pasos segmentados
segmentada = []

## Itero para cada uno de los pasos que tengo detectados
for i in range (len(picos) - 1):

    ## Hago la segmentación de la señal
    segmento = pos_z_filtrada[picos[i] : picos[i + 1]]

    ## Luego lo agrego a la señal segmentada
    segmentada.append(segmento)

## ------------------------------------------------ GRAFICACIÓN --------------------------------------------------------

# ## Valores a graficar
# plt.plot(tiempo, acc_analytic[:,0], color = 'r', label = '$a_x$')
# plt.plot(tiempo, acc_analytic[:,1], color = 'b', label = '$a_y$')
# plt.plot(tiempo, acc_analytic[:,2], color = 'g', label = '$a_z$')

# ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
# plt.xlabel("Tiempo (s)")
# plt.ylabel("Aceleracion $(m/s^2)$")

# ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
# plt.legend()

# ## Despliego la gráfica
# plt.show()

# GraficacionPicos(pos_z_filtrada, picos)

# plt.plot(tiempo, pos_z_filtrada)
# plt.plot(tiempo, acc_analytic[:,2])
# plt.show()

plt.plot(segmentada[4])
plt.show()