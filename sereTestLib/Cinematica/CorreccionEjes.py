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
import math
import pykalman

integrate.cumulative_trapezoid

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S269/3S269.csv"

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

## Calculamos los valores del magnetometro
data_mag = ValoresMagnetometro("C:/Yo/Tesis/sereData/sereData/Dataset/raw_2_process", 151, ['Caminando'])

## Armamos una matriz donde las columnas sean los valores del magnetometro
mag = np.array([np.array(data_mag['Mag_x']), np.array(data_mag['Mag_y']), np.array(data_mag['Mag_z'])], dtype = float).transpose()

## ------------------------------------ MÉTODO CON FILTRO DE KALMAN ------------------------------------

# ## Armamos el diccionario con los datos a ingresar
# dict_datos = {'acc': acel, 'omega' : gyro, 'rate' : 200, 'mag' : mag}

# ## Se crea una instancia IMU_Base con filtro de Kalman pasando como argumento el diccionario de datos
# imu_kalman = IMU_Base(in_data = dict_datos, q_type = 'kalman')

# ## Accedo al atributo <<quat>> para ver las orientaciones de cuaterniones
# q_kalman = imu_kalman.quat

# ## Accedo a la velocidad resultante
# vel_kalman = imu_kalman.vel

# ## Accedo a la posición resultante
# pos_kalman = imu_kalman.pos

# ## Accedo a la aceleración corregida
# acc_kalman = imu_kalman.accCorr

# ## Accedo a la acleración del sensor luego de restar la gravedad
# accSens_kalman = imu_kalman.accSens

# ## Accedo a la aceleración gravitatoria en el sensor
# gSens_kalman = imu_kalman.gCorr

## ------------------------------------------ MÉTODO ANALÍTICO ------------------------------------

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

## Accedo al vector de la gravedad en el sistema del sensor
## Obs. La norma del vector de aceleración gravitatoria se conserva, implicando rotacion con cuaterniones unitarios
gSens_analytic = imu_analytic.gCorr

## Accedo a la aceleración con compensación de deriva
accComp_analytic = imu_analytic.accReSpaceComp

## ------------------------------------------------GRAFICACIÓN --------------------------------------------------------

# ## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
# #plt.plot(tiempo, acel[:,2], color = 'r', label = '$a_x$')
# plt.plot(gSens_kalman[:,0], color = 'r', label = '$a_x$')
# plt.plot(gSens_kalman[:,1], color = 'b', label = '$a_y$')
# plt.plot(gSens_kalman[:,2], color = 'g', label = '$a_z$')

# plt.plot(vel_analytic[:,0], color = 'r', label = '$a_x$')
# plt.plot(vel_analytic[:,1], color = 'b', label = '$a_y$')
# plt.plot(vel_analytic[:,2], color = 'g', label = '$a_z$')

plt.plot(acc_analytic[:,0], color = 'r', label = '$a_x$')
plt.plot(acc_analytic[:,1], color = 'b', label = '$a_y$')
plt.plot(acc_analytic[:,2], color = 'g', label = '$a_z$')

# plt.plot(acc_analytic[:,2], color = 'b', label = '$a_y$')
# plt.plot(acel[:,2], color = 'g', label = '$a_z$')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleracion $(m/s^2)$")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()

# plt.plot(gyro[:,0], color = 'r', label = '$a_x$')
# plt.plot(gyro[:,1], color = 'b', label = '$a_y$')
# plt.plot(gyro[:,2], color = 'g', label = '$a_z$')
# #plt.plot(tiempo, math.sqrt(acc_analytic[:,0]**2+acc_analytic[:,1]**2), color = 'g', label = '$a_z$')

# ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
# plt.xlabel("Tiempo (s)")
# plt.ylabel("Aceleracion $(m/s^2)$")

# ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
# plt.legend()

# ## Despliego la gráfica
# plt.show()

# plt.plot(tiempo, vel_analyitic[:,0], color = 'r', label = '$v_x$')
# plt.plot(tiempo, vel_analyitic[:,1], color = 'b', label = '$v_y$')
# plt.plot(tiempo, vel_analyitic[:,2], color = 'g', label = '$v_z$')

# ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
# plt.xlabel("Tiempo (s)")
# plt.ylabel("Aceleracion $(m/s^2)$")

# ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
# plt.legend()

# ## Despliego la gráfica
# plt.show()

# plt.plot(vel_analyitic[:,0], color = 'r', label = '$a_x$')

# ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
# plt.xlabel("Tiempo (s)")
# plt.ylabel("Aceleracion $(m/s^2)$")

# ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
# plt.legend()

# ## Despliego la gráfica
# plt.show()

## ------------------------------------------------PRUEBAS CON FILTROS --------------------------------------------------------
# ## Separo el vector de tiempos del dataframe
# tiempo = np.array(data['Time'])

# ## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
# tiempo = (tiempo - tiempo[0]) / 1000

# ## ------------- BUTTERWORTH --------------
# ## Al usar Butterworth hay que tener en cuenta que se introduce un RETARDO en la señal filtrada
# ## Éste filtro me elimina muy bien los picos pero las señales van perdiendo su forma
# ## El compromiso es muy grande y no se anulan los saltos a la salida
# sos = signal.butter(N = 5, Wn = 1, btype = 'lowpass', fs = 1 / periodoMuestreo, output = 'sos')

# GY_x_F = signal.sosfilt(sos, np.array(data['GY_x']))

# GY_y_F = signal.sosfilt(sos, np.array(data['GY_y']))

# GY_z_F = signal.sosfilt(sos, np.array(data['GY_z']))

# ## Armamos una matriz donde las columnas sean los valores de los giros
# gyro = np.array([GY_x_F, GY_y_F, GY_z_F]).transpose()
# ## ----------------------------------------

# ## Armamos el diccionario con los datos a ingresar
# dict_datos = {'acc': acel, 'omega' : gyro, 'rate' : 1 / periodoMuestreo, 'mag' : mag}

# ## Creacion de instancia IMU_Base
# imu_kalman_FILTRADO = IMU_Base(in_data = dict_datos, q_type = 'kalman')

# ## Accedo a los cuaterniones resultantes
# q_kalman_FILTRADO = imu_kalman_FILTRADO.quat

# ## Accedo a la velocidad resultante
# vel_kalman_FILTRADO = imu_kalman_FILTRADO.vel

# ## Accedo a la posición resultante
# pos_kalman_FILTRADO = imu_kalman_FILTRADO.pos

# ## Accedo a la aceleración corregida
# acc_kalman_FILTRADO = imu_kalman_FILTRADO.accCorr

# ## Accedo a la acleración del sensor luego de restar la gravedad
# accSens_kalman_FILTRADO = imu_kalman_FILTRADO.accSens

# ## Accedo al vector de la gravedad en el sistema del sensor
# ## Obs. La norma del vector de aceleración gravitatoria se conserva, implicando rotacion con cuaterniones unitarios
# gSens_kalman_FILTRADO = imu_kalman_FILTRADO.gCorr

# # ## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
# plt.plot(tiempo, acc_kalman[:,0], color = 'r', label = '$Original$')
# plt.plot(tiempo, acc_kalman_FILTRADO[:,0], color = 'b', label = '$Filtrado$')
# # #plt.plot(acc_analytic[:,1], color = 'b', label = '$a_y$')
# # #plt.plot(acc_analytic[:,2], color = 'g', label = '$a_z$')

# ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
# plt.xlabel("Tiempo (s)")
# plt.ylabel("Aceleracion $(m/s^2)$")

# ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
# plt.legend()

# ## Despliego la gráfica
# plt.show()

# # # Grafico los datos. En mi caso las tres velocidades angulares
# plt.plot(tiempo, GY_x_F, color = 'r', label = '$w_x$')
# plt.plot(tiempo, np.array(data['GY_x']), color = 'b', label = '$w_y$')
# # plt.plot(tiempo, GY_z, color = 'g', label = '$w_z$')

# # Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la velocidad angular (rad/s)
# plt.xlabel("Tiempo (s)")
# plt.ylabel("Velocidad angular (rad/s)")

# # Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
# plt.legend()

# plt.show()