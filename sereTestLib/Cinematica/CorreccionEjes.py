## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from parameters import *
from Muestreo import *
from ValoresMagnetometro import *
from LecturaDatos import *
import pandas as pd
import ahrs
from ahrs.common.quaternion import *
import numpy as np
from matplotlib import pyplot as plt
from skinematics.imus import *
from pyquaternion import *
from vqf import VQF

## ---------------------------------------- CORRECCIÓN DE EJES ------------------------------------------

## Función toma las señales de aceleración y cuaterniones de rotación y los rota
def Rotacion(acel, quat_rotacion):

    ## Construyo una lista en donde voy a almacenar las aceleraciones rotadas
    acel_rotada = []

    ## Itero para cada uno de los cuaterniones de orientacion
    for i in range (quat_rotacion.shape[0]):

        ## Inicializo un objeto conteniendo el cuaternión de rotación
        cuaternion_rotacion = Quaternion(quat_rotacion[i, :][0], quat_rotacion[i, :][1], quat_rotacion[i, :][2], quat_rotacion[i, :][3])

        ## MÉTODO I: Rotación del vector usando el cuaternión como objeto
        vector_rotado = cuaternion_rotacion.rotate(acel[i, :])

        ## MÉTODO II: Aplico la rotación del vector i de aceleración según el cuaternión i
        acel_rotada.append(vector.rotate_vector(acel[i, :], cuaternion_rotacion.q))
    
    ## MÉTODO III: Obtención de la aceleración rotada aplicando los cuaterniones en un único paso
    rotated_acc = np.array([Quaternion(q_np).rotate(acel[q_idx]) for q_idx, q_np in enumerate(quat_rotacion)])
    
    ## Retorno la aceleración rotada
    return np.array(acel_rotada)

## Función que me elimine las componentes de la gravedad de las señales de acelerometría
def EliminarGravedad(acel, quat_rotacion):

    ## Construyo una lista en donde voy a almacenar las aceleraciones sin tener en cuenta la gravedad
    acel_sin_g = []

    ## Defino el vector de la gravedad como vertical hacia arriba
    gravedad = np.array([0, 0, constants.g])

    ## Itero para cada uno de los cuaterniones de orientación
    for i in range (quat_rotacion.shape[0]):

        ## Inicializo un objeto conteniendo el cuaternión de rotación
        cuaternion_rotacion = Quaternion(quat_rotacion[i, :][0], quat_rotacion[i, :][1], quat_rotacion[i, :][2], quat_rotacion[i, :][3])

        ## Hago la rotación de la gravedad usando el cuaternión inverso
        ## Como resultado obtengo la gravedad en el sistema solidario al IMU
        gravedad_rotada = vector.rotate_vector(gravedad, cuaternion_rotacion.inverse.q)

        ## Almaceno la resta entre la aceleración original y la gravedad rotada
        acel_sin_g.append(acel[i, :] - gravedad_rotada)
    
    ## Retorno la aceleración sin gravedad en el sistema del IMU
    return np.array(acel_sin_g)
    
## Función que toma como entrada los valores de aceleración, velocidad angular e intensidad de campo magnético
## Se devuelve a la salida el objeto filtro dependiendo del modelo elegido
def Filtrado(acel, gyro, mag, modelo, fs):

    ## Método I: Filtro de Kalman Extendido
    if modelo == 'ekf':

        ## Procesamiento con EKF
        filtro = ahrs.filters.EKF(gyr = gyro, acc = acel, mag = mag, frame = 'ENU', frequency = fs)

    ## Método II: Filtro de Mahony
    elif modelo == 'mahony':

        ## Procesamiento con Mahony
        filtro = ahrs.filters.Mahony(acc = acel, gyr = gyro, mag = mag, frequency = fs)

    ## Método III: Filtro de Madgwick
    elif modelo == 'madgwick':

        ## Procesamiento con Madgwick
        filtro = ahrs.filters.Madgwick(gyr = gyro, acc = acel, mag = mag, frequency = fs)

    ## Método IV: Filtro Complementario
    elif modelo == "complementary":

        ## Procesamiento con complementario
        filtro = ahrs.filters.Complementary(gyr = gyro, acc = acel, mag = mag, frequency = fs, gain = 0.8)
    
    ## Método V: VQF
    elif modelo == 'vqf':

        ## Creación del objeto VQF
        filtro = VQF(gyrTs = 1 / fs, accTs = 1 / fs, magTs = 1 / fs)

        ## Cargo los datos de entrada
        filtro = filtro.updateBatch(gyr = np.ascontiguousarray(gyro), acc = np.ascontiguousarray(acel), mag = np.ascontiguousarray(mag))

    ## Retorno el filtro correspondiente
    return filtro

## Ejecución principal del programa
if __name__== '__main__':

    ## Especifico la ruta en la cual se encuentra el registro a leer
    ruta_registro_completa = ruta_registro + 'MarchaLibre_Sabrina.txt'

    ## Hago la lectura de los valores del magnetómetro
    mag = ValoresMagnetometro(id_persona = 69, lectura_datos_propios = True, ruta = ruta_registro_completa)

    ## Hago la lectura de las aceleraciones y los giroscopios
    data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = 69, lectura_datos_propios = True, ruta = ruta_registro_completa)

    ## Especifico el conjunto de filtros que tengo a disposicion
    filtros = ['complementary']

    ## Itero para cada uno de los filtros que tengo
    for nombre_filtro in filtros:

        ## Hago el filtrado de la señal para obtener orientacion
        filtro = Filtrado(acel, gyro, mag, modelo = nombre_filtro, fs = 1 / periodoMuestreo)
        
        ## En caso de que el filtro sea vqf
        if nombre_filtro == 'vqf':

            ## Accedo de distinta manera a los cuaterniones
            ## Ésta operación rota la aceleración del sistema solidario a la IMU al sistema solidario a la Tierra
            acel_rotada = Rotacion(acel, filtro['quat9D'])
        
        ## En caso de que tenga otro filtro
        else:

            ## Hago la rotación de la aceleración usando cuaterniones de orientacion
            ## Ésta operación rota la aceleración del sistema solidario a la IMU al sistema solidario a la Tierra
            acel_rotada = Rotacion(acel, filtro.Q)

        ## Graficación de la aceleración rotada
        plt.plot(acel_rotada[:, 0], label = '{}'.format(nombre_filtro))

    ## Despliego la gráfica
    plt.plot(acel[:, 0], label = 'Real')
    plt.legend()
    plt.show()

    ## Elimino la gravedad
    acel_sin_g = EliminarGravedad(acel, filtro.Q)

    ## Grafico aceleracion sin gravedad
    plt.plot(acel_sin_g[:, 2])
    plt.show()