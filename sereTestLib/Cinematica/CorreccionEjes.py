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
import numpy as np
from scipy.signal import filtfilt, butter
from ahrs import *
from pyquaternion import Quaternion
from Cuaterniones import *

def quaternion_to_euler_angle_vectorized1(w, x, y, z):

    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z

def estimate_orientation(a, w, t, alpha=0.9, g_ref=(0., 0., 1.),
                         theta_min=1e-6, highpass=.01, lowpass=.05):
    """ Estimate orientation with a complementary filter.
    Fuse linear acceleration and angular velocity measurements to obtain an
    estimate of orientation using a complementary filter as described in
    `Wetzstein 2017: 3-DOF Orientation Tracking with IMUs`_
    .. _Wetzstein 2017: 3-DOF Orientation Tracking with IMUs:
    https://pdfs.semanticscholar.org/5568/e2100cab0b573599accd2c77debd05ccf3b1.pdf
    Parameters
    ----------
    a : array-like, shape (N, 3)
        Acceleration measurements (in arbitrary units).
    w : array-like, shape (N, 3)
        Angular velocity measurements (in rad/s).
    t : array-like, shape (N,)
        Timestamps of the measurements (in s).
    alpha : float, default 0.9
        Weight of the angular velocity measurements in the estimate.
    g_ref : tuple, len 3, default (0., 0., 1.)
        Unit vector denoting direction of gravity.
    theta_min : float, default 1e-6
        Minimal angular velocity after filtering. Values smaller than this
        will be considered noise and are not used for the estimate.
    highpass : float, default .01
        Cutoff frequency of the high-pass filter for the angular velocity as
        fraction of Nyquist frequency.
    lowpass : float, default .05
        Cutoff frequency of the low-pass filter for the linear acceleration as
        fraction of Nyquist frequency.
    Returns
    -------
    q : array of quaternions, shape (N,)
        The estimated orientation for each measurement.
    """

    # initialize some things
    N = len(t)
    dt = np.diff(t)
    g_ref = np.array(g_ref)
    q = np.ones(N, dtype=Quaternion)

    # get high-passed angular velocity
    w = filtfilt(*butter(5, highpass, btype='high'), w, axis=0)
    w[np.linalg.norm(w, axis=1) < theta_min] = 0
    q_delta = from_rotation_vector(w[1:] * dt[:, None])

    # get low-passed linear acceleration
    a = filtfilt(*butter(5, lowpass, btype='low'), a, axis=0)

    for i in range(1, N):

        # get rotation estimate from gyroscope
        q_w = q[i - 1] * q_delta[i - 1]

        # get rotation estimate from accelerometer
        v_world = rotate_vectors(q_w, a[i])
        n = np.cross(v_world, g_ref)
        phi = np.arccos(np.dot(v_world / np.linalg.norm(v_world), g_ref))
        q_a = from_rotation_vector(
            (1 - alpha) * phi * n[None, :] / np.linalg.norm(n))[0]

        # fuse both estimates
        q[i] = q_a * q_w

    return q

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Ruta donde voy a abrir el archivo
ruta_registro = "C:/Yo/Tesis/sereData/sereData/Registros/MarchaLibre_Rodrigo.txt"

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

q = filters.complementary.Complementary(gyr = gyro, acc = acel, frequency = 1 /periodoMuestreo)

q_am = q.am_estimation(acc = acel)

angulos_gy = []

for i in range (q_analytic.shape[0]):

    angulos_gy.append(quaternion_to_euler_angle_vectorized1(q[i,0], q[i,1], q[i,2], q[i,3]))

plt.plot(angulos_gy)
plt.show()

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