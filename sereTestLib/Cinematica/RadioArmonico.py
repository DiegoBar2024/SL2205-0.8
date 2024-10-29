from Magnitud import Magnitud
from Normalizacion import Normalizacion
from DeteccionPicos import *
from Filtros import FiltroMediana
from DeteccionPicos import *
from Fourier import *
import numpy as np
from Muestreo import PeriodoMuestreo
import pandas as pd
from ValoresMagnetometro import ValoresMagnetometro
from matplotlib import pyplot as plt
from scipy import signal
import libf0
from harm_analysis import harm_analysis
from control import *
from skinematics.imus import analytical, IMU_Base
from skinematics.sensors import xsens
from skinematics.quat import *

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S267/3S267.csv"

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

## ----------------------------------------- DETECCIÓN DE FRECUENCIA FUNDAMENTAL ------------------------------------------

## Defino la señal a procesar
señal = acel[:,2]

## MÉTODO I: A MANO
## Se obtiene el espectro de toda la señal completa aplicando la transformada de Fourier
## Ya que es una señal real se cumple la simetría conjugada
(frecuencias, transformada) = TransformadaFourier(señal, periodoMuestreo)

## Cálculo de los coeficientes de Fourier en semieje positivo
coefs = (2 / señal.shape[0]) * np.abs(transformada)[:señal.shape[0]//2]

## Calculo de las frecuencias en el semieje positivo
frecs = frecuencias[:señal.shape[0]//2]

## Elimino la componente de contínua de la señal
coefs[0] = 0

## Determino la posición en la que se da el máximo. 
## ESTOY ASUMIENDO QUE EL MÁXIMO SE DA EN LA COMPONENTE FUNDAMNENTAL (no tiene porque ocurrir!)
## Ésta será considerada como la frecuencia fundamental de la señal
pos_maximo = np.argmax(coefs)

## Frecuencia fundamental de la señal
## La frecuencia fundamental de la señal en las aceleraciones CC (craneo-cervical) y AP (antero-posterior) son iguales.
## Se podría interpretar como la frecuencia fundamental de los pasos en la marcha de la persona
frec_fund = frecs[pos_maximo]

## MÉTODO II: USANDO HARM_ANALYSIS
## ¡Ojo! Éste método vale únicamente cuando el armónico fundamental es el de MAYOR amplitud (no siempre se cumple!)

## Defino cantidad de armónicos no fundamentales a calcular
n_harm = 19

## Análisis de armónicos
armonicos = harm_analysis(señal, FS = 1 / periodoMuestreo, n_harm = 19)

## Obtengo frecuencia fundamental de la señal
frec_fund = armonicos['fund_freq']

## Potencia de la componente fundamental de la señal
fund_db = armonicos['fund_db']

## Potencias de los armónicos no fundamentales
no_fund_db = np.array(armonicos['pot_armonicos'])

## Hago el pasaje del valor obtenido en decibeles a magnitud
fund_mag = db2mag(fund_db)

## Hago el pasaje de los valores obtenidos en decibeles a magnitud
no_fund_mag = db2mag(no_fund_db)

## Calculo la amplitud del armónico fundamental (como sinusoidales)
amplitud_fund = np.sqrt(2 * fund_mag)

## Calculo la amplitud de los armónicos no fundamentales (como sinusoidales)
amplitud_no_fund = np.sqrt(2 * no_fund_mag)

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

## ----------------------------- PROCESADO DE SEÑAL CON EJES CORREGIDOS -----------------------

## Defino la nueva señal
señal = acc_analytic[:,2]

## Transformada de Fourier
(frecuencias, transformada) = TransformadaFourier(señal, periodoMuestreo)

## ----------------------------------------- CÁLCULO DEL HARMONIC RATIO ------------------------------------------

# ## Numerador del HR (armónicos pares) suponiendo aceleración AP/CC
# num_HR = 0

# ## Itero para cada uno de los armónicos pares que tengo
# for i in range (0, n_harm, 2):
    
#     ## Sumo la magnitud del armónico par correspondiente
#     num_HR += no_fund_mag[i]

# ## Denominador del HR (armónicos impares) suponiendo aceleración AP/CC
# den_HR = fund_mag

# ## Itero para cada uno de los armónicos impares que tengo
# for i in range(1, n_harm, 2):

#     ## Sumo la magnitud del armónico impar correspondiente
#     den_HR += no_fund_mag[i]

# ## Calculo el HR de la persona
# HR = num_HR / den_HR

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

## ----------------------------------- EJEMPLO DE PRUEBA ------------------------------------------

# Cantidad de muestras
N = 600

# Período de muestreo
T = 1.0 / 1200.0

## Vector de tiempos
x = np.linspace(0.0, N * T, N, endpoint = False)

## Función
y = np.cos(50.0 * 2.0 * np.pi * x) + 0.5 * np.cos(100.0 * 2.0 * np.pi * x) + 0.25 * np.cos(150 * 2.0 * np.pi * x) + 0.125 * np.cos(200 * 2.0 * np.pi * x) +  0.1 * np.cos(300 * 2.0 * np.pi * x) + 0.05 * np.sin(500*2*np.pi*x)

## Transformada de Fourier
TransformadaFourier(y, T)

## Analisis armónico de la señal
armonicos = harm_analysis(y, FS = 1 / T, n_harm = 10)

## Obtengo frecuencia fundamental de la señal
frec_fund = armonicos['fund_freq']

## Potencia de la componente fundamental de la señal
fund_db = armonicos['fund_db']

## Hago el pasaje del valor obtenido en decibeles a magnitud
fund_mag = db2mag(fund_db)

## Calculo la amplitud del armónico fundamental
amplitud_fund = np.sqrt(2 * fund_mag)

## Potencias de los armónicos no fundamentales (como sinusoidales)
no_fund_db = np.array(armonicos['pot_armonicos'])

## Hago el pasaje de los valores obtenidos en decibeles a magnitud
no_fund_mag = db2mag(no_fund_db)

## Calculo la amplitud de los armónicos no fundamentales (como sinusoidales)
amplitud_no_fund = np.sqrt(2 * no_fund_mag)