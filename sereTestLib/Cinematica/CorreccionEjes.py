import skinematics.quat 
from skinematics.imus import analytical, IMU_Base
from skinematics.sensors import xsens
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Integracion import CalcularVelocidades
from Fourier import TransformadaFourier
from ValoresMagnetometro import ValoresMagnetometro
from Muestreo import PeriodoMuestreo

from skinematics.sensors.xsens import XSens
import numpy as np
from scipy import integrate

integrate.cumulative_trapezoid
ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S299/3S299.csv"

## Lectura de datos
data = pd.read_csv(ruta)

## Hallo el período de muestreo de las señales
periodoMuestreo = PeriodoMuestreo(data)

## Armamos una matriz donde las columnas sean las aceleraciones
acel = np.array([np.array(data['AC_x']), np.array(data['AC_y']), np.array(data['AC_z'])]).transpose()

## Armamos una matriz donde las columnas sean los valores de los giros
gyro = np.array([np.array(data['GY_x']), np.array(data['GY_y']), np.array(data['GY_z'])]).transpose()

## ------------------------------------ MÉTODO CON FILTRO DE KALMAN ------------------------------------

# ## Calculamos los valores del magnetometro
# data_mag = ValoresMagnetometro("C:/Yo/Tesis/sereData/sereData/Dataset/raw_2_process", 169, ['Caminando'])

# ## Armamos una matriz donde las columnas sean los valores del magnetometro
# mag = np.array([np.array(data_mag['Mag_x']), np.array(data_mag['Mag_y']), np.array(data_mag['Mag_z'])], dtype = float).transpose()

# ## Armamos el diccionario con los datos a ingresar
# dict_datos = {'acc': acel, 'omega' : gyro, 'rate' : 1 / periodoMuestreo, 'mag' : mag}

# ## Se crea una instancia IMU_Base con filtro de Kalman pasando como argumento el diccionario de datos
# imu_kalman = IMU_Base(in_data = dict_datos, q_type = 'kalman')

# ## Accedo al atributo <<quat>> para ver las orientaciones de cuaterniones
# orient_kalman = imu_kalman.quat

# ## Accedo a la velocidad resultante
# vel_kalman = imu_kalman.vel

# ## Accedo a la posición resultante
# pos_kalman = imu_kalman.pos

# ## Accedo a la aceleración corregida
# acc_kalman = imu_kalman.accCorr

## ------------------------------------ MÉTODO ANALÍTICO ------------------------------------

## Armamos el diccionario con los datos a ingresar
dict_datos = {'acc': acel, 'omega' : gyro, 'rate' : 1 / periodoMuestreo}

## Creacion de instancia IMU_Base
imu_analytic = IMU_Base(in_data = dict_datos, q_type = 'analytical')

## Accedo a la velocidad resultante
vel_analyitic = imu_analytic.vel

## Accedo a la posición resultante
pos_analyitic = imu_analytic.pos

## Accedo a la aceleración corregida
acc_analytic = imu_analytic.accCorr

## --------------------------- -----TESTING --------------------------------------------------------
## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
##plt.plot(acc_analytic[:,0], color = 'r', label = '$a_x$')
##plt.plot(acc_analytic[:,1], color = 'b', label = '$a_y$')
plt.plot(acc_analytic[:,2], color = 'g', label = '$a_z$')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleracion $(m/s^2)$")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()

# plt.plot(acc_kalman[:,2], color = 'r', label = '$a_x$ corregida')
# ## plt.plot(acel[:,2], color = 'b', label = '$a_x$ s/corregir')

# ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
# plt.xlabel("Tiempo (s)")
# plt.ylabel("Aceleracion $(m/s^2)$")

# ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
# plt.legend()

# ## Despliego la gráfica
## plt.show()

# print(pos_kalman[:,0])

# ## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
# plt.plot(acc_kalman[:,0], color = 'r', label = '$a_x$')
# plt.plot(vel_kalman[:,0], color = 'b', label = '$v_x$')
# plt.plot(pos_kalman[:,0], color = 'g', label = '$x_x$')

# ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
# plt.xlabel("Tiempo (s)")
# plt.ylabel("Aceleracion $(m/s^2)$")

# ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
# plt.legend()

# ## Despliego la gráfica
# plt.show()