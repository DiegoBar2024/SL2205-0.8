## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from parameters import *
from Muestreo import *
from ValoresMagnetometro import *
from LecturaDatos import *
import parameters
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import imumaster
import ahrs
from skinematics import imus

## ---------------------------------------- CORRECCIÓN DE EJES ------------------------------------------

def CorreccionEjes():

    return

## Ejecución principal del programa
if __name__== '__main__':

    ## Especifico la ruta en la cual se encuentra el registro a leer
    ruta_registro_completa = ruta_registro + 'MarchaLibre_Sabrina.txt'

    ## Hago la lectura de los valores del magnetómetro
    mag = ValoresMagnetometro(id_persona = 69, lectura_datos_propios = True, ruta = ruta_registro_completa)

    ## Hago la lectura de las aceleraciones y los giroscopios
    data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = 69, lectura_datos_propios = True, ruta = ruta_registro_completa)

    # ## Cálculo de orientaciones
    # orientation = imumaster.Orientation(sample_rate = 1 / periodoMuestreo, frame = 'ENU', method = 'EKF')

    # ## Genero una lista vacía donde voy a guardar las aceleraciones rotadas
    # acel_rotadas = []

    # ## MÉTODO I: MADGWICK
    # madgwick = ahrs.filters.Madgwick(gyr = np.array([gyro[:, 0], gyro[:, 1], gyro[:, 2]]).transpose(),
    #                             acc = np.array([acel[:, 0], acel[:, 1], acel[:, 2]]).transpose(),
    #                             frequency = 1 / periodoMuestreo)

    # ## Obtengo cuaterniones estimados
    # quaternions = madgwick.Q

    # rotated_vector = R.from_quat(quaternions).apply(np.array([acel[:, 0], acel[:, 1], acel[:, 2]]).transpose())

    # ## MÉTODO II: FILTRO DE KALMAN EXTENDIDO
    # ## Itero para cada una de las muestras que tengo
    # for i in range (acel.shape[0]):

    #     ## Estimacion del cuaternión de rotación en la muestra dada
    #     q_estimation = orientation.EKF(np.array([acel[:, 0], acel[:, 2], acel[:, 1]]).transpose()[i, :], 
    #                                 np.array([gyro[:, 0], gyro[:, 2], gyro[:, 1]]).transpose()[i, :],
    #                                 np.array([mag[:, 0], mag[:, 2], mag[:, 1]]).transpose()[i, :])

    #     ## Cálculo de ángulos de Euler para la muestra dada
    #     eulerangle = orientation.eulerangle(q_estimation)

    #     ## Hago la rotación de las aceleraciones usando Euler
    #     acel_rotada = R.from_euler('xyz', eulerangle, degrees = True).apply(np.array([acel[:, 0], acel[:, 2], acel[:, 1]]).transpose()[i, :])

    #     ## Agrego la aceleración rotada a la lista correspondiente
    #     acel_rotadas.append(acel_rotada)
    
    ## MÉTODO III
    q, pos, vel, acel_rotada = imus.analytical(omega = np.array([gyro[:, 0], gyro[:, 2], gyro[:, 1]]).transpose(), 
                                accMeasured = np.array([acel[:, 0], acel[:, 2], acel[:, 1]]).transpose(), 
                                rate = 1 / periodoMuestreo)

    rotated_vector = R.from_quat(q).apply(np.array([acel[:, 0], acel[:, 2], acel[:, 1]]).transpose())