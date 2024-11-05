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
import emd

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

## ------------------------------------- INTEGRACIÓN ACELERACIÓN ---------------------------------------

## Integro aceleración para obtener velocidad
vel_z = cumulative_trapezoid(acc_analytic[:,2], dx = periodoMuestreo, initial = 0)

## ----------------------------------- DESCOMPOSICIÓN EMD VELOCIDAD ------------------------------------

## Hago la descomposición EMD de la señal de velocidad
imfs_vel_z = emd.sift.sift(vel_z)

## Para compensar la deriva en la integración suprimo la IMF de menor frecuencia
vel_z_procesada = vel_z - imfs_vel_z[:,-1]

plt.plot(vel_z)
plt.show()

## Graficación de las IMFs de la señal de velocidad
plt.plot(tiempo, imfs_vel_z[:,0], label = 'Imf1')
plt.plot(tiempo, imfs_vel_z[:,1], label = 'Imf2')
plt.plot(tiempo, imfs_vel_z[:,2], label = 'Imf3')
plt.plot(tiempo, imfs_vel_z[:,3], label = 'Imf4')
plt.legend()
plt.show()

## -------------------------------------- INTEGRACIÓN VELOCIDAD ----------------------------------------

## Integro velocidad para obtener posición
pos_z = cumulative_trapezoid(vel_z_procesada, dx = periodoMuestreo, initial = 0)

## ----------------------------------- DESCOMPOSICIÓN EMD POSICIÓN -------------------------------------

## Hago la descomposición EMD de la señal de posición
imfs_pos_z = emd.sift.sift(pos_z)

## Por criterio me voy a quedar con los primeros tres IMFs de la señal de posición
pos_z_procesada = imfs_pos_z[:,0] + imfs_pos_z[:,1]

# ## Graficación de las IMFs de la señal de posición
# plt.plot(tiempo, imfs_pos_z[:,0], label = 'Imf1')
# plt.plot(tiempo, imfs_pos_z[:,1], label = 'Imf2')
# plt.plot(tiempo, imfs_pos_z[:,2], label = 'Imf3')
# plt.plot(tiempo, imfs_pos_z[:,3], label = 'Imf4')
# plt.plot(tiempo, imfs_pos_z[:,4], label = 'Imf5')
# plt.plot(tiempo, imfs_pos_z[:,5], label = 'Imf6')
# plt.legend()
# plt.show()

## Graficación de la señal
plt.plot(tiempo, pos_z_procesada)
plt.show()

TransformadaFourier(pos_z_procesada, periodoMuestreo)