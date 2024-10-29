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
from filterpy.kalman import *

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S300/3S300.csv"

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

## MÉTODO I: USANDO HARM_ANALYSIS
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

print(amplitud_fund)
print(amplitud_no_fund)

## MÉTODO II: A MANO
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

## -------------------------------------- CORRECCIÓN DE EJES ----------------------------------

# ## Armamos el diccionario con los datos a ingresar
# dict_datos = {'acc': acel, 'omega' : gyro, 'rate' : 1 / periodoMuestreo}

# ## Matriz de rotación inicial
# ## Es importante tener en cuenta la orientación inicial de la persona
# orient_inicial = np.array([np.array([1,0,0]), np.array([0,0,1]), np.array([0,1,0])])

# ## Creacion de instancia IMU_Base
# imu_analytic = IMU_Base(in_data = dict_datos, q_type = 'analytical', R_init = orient_inicial, calculate_position = False)

# ## Accedo a los cuaterniones resultantes
# q_analytic = imu_analytic.quat

# ## Accedo a la velocidad resultante
# vel_analytic = imu_analytic.vel

# ## Accedo a la posición resultante
# pos_analytic = imu_analytic.pos

# ## Accedo a la aceleración corregida
# acc_analytic = imu_analytic.accCorr

## ----------------------------- PROCESADO DE SEÑAL CON EJES CORREGIDOS -----------------------

# ## Defino la nueva señal
# señal = acc_analytic[:,2]

# ## Transformada de Fourier
# (frecuencias, transformada) = TransformadaFourier(señal, periodoMuestreo)

## ----------------------------------------- CÁLCULO DEL HARMONIC RATIO ------------------------------------------

## Numerador del HR (armónicos pares) suponiendo aceleración AP/CC
num_HR = 0

## Itero para cada uno de los armónicos pares que tengo
for i in range (0, n_harm, 2):
    
    ## Sumo la magnitud del armónico par correspondiente
    num_HR += amplitud_no_fund[i]

## Denominador del HR (armónicos impares) suponiendo aceleración AP/CC
den_HR = amplitud_fund

## Itero para cada uno de los armónicos impares que tengo
for i in range(1, n_harm, 2):

    ## Sumo la magnitud del armónico impar correspondiente
    den_HR += amplitud_no_fund[i]

## Calculo el HR de la persona
HR = num_HR / den_HR

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

## PRUEBA SIN ADICIÓN DE RUIDO
# Cantidad de muestras
N = 600

# Período de muestreo
T = 1.0 / 1500.0

## Vector de tiempos
x = np.linspace(0.0, N * T, N, endpoint = False)

## Función
y = np.cos(50.0 * 2.0 * np.pi * x) + 0.5 * np.cos(100.0 * 2.0 * np.pi * x) + 0.75 * np.cos(150 * 2.0 * np.pi * x) + 0.125 * np.cos(200 * 2.0 * np.pi * x) +  0.1 * np.cos(300 * 2.0 * np.pi * x) + 0.05 * np.sin(500*2*np.pi*x)

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

## Potencias de los armónicos no fundamentales (como sinusoidales)
no_fund_db = np.array(armonicos['pot_armonicos'])

## Hago el pasaje de los valores obtenidos en decibeles a magnitud
no_fund_mag = db2mag(no_fund_db)

## Calculo la amplitud del armónico fundamental
amplitud_fund = np.sqrt(2 * fund_mag)

## Calculo la amplitud de los armónicos no fundamentales (como sinusoidales)
amplitud_no_fund = np.sqrt(2 * no_fund_mag)

## Impresión de amplitudes normales
print(amplitud_fund)
print(amplitud_no_fund)

## PRUEBA CON ADICIÓN DE RUIDO

## Desviación estándar del ruido
desv_estandar = 5

## Genero un vector de ruido normal que tenga las mismas dimensiones que el vector de la función
## El parámetro <<loc>> me indica el valor medio del ruido gaussiano
## El parámetro <<scale>> me indica la desviación estándar del ruido gaussiano
## El parámetro <<size>> me indica la longitud del vector de ruido (quiero que tenga la misma longitud que mi señal)
ruido = np.random.normal(loc = 0, scale = desv_estandar, size = len(y))

## Construyo la señal ruidosa
y_ruido = y + ruido

## Transformada de Fourier de señal ruidosa
TransformadaFourier(y_ruido, T)

## Analisis armónico de la señal ruidosa
armonicos_ruido = harm_analysis(y_ruido, FS = 1 / T, n_harm = 10)

## Obtengo frecuencia fundamental de la señal
frec_fund_ruido = armonicos_ruido['fund_freq']

## Potencia de la componente fundamental de la señal
fund_db_ruido = armonicos_ruido['fund_db']

## Hago el pasaje del valor obtenido en decibeles a magnitud
fund_mag_ruido = db2mag(fund_db_ruido)

## Potencias de los armónicos no fundamentales (como sinusoidales)
no_fund_db_ruido = np.array(armonicos_ruido['pot_armonicos'])

## Hago el pasaje de los valores obtenidos en decibeles a magnitud
no_fund_mag_ruido = db2mag(no_fund_db_ruido)

## Calculo la amplitud del armónico fundamental
amplitud_fund_ruido = np.sqrt(2 * fund_mag_ruido)

## Calculo la amplitud de los armónicos no fundamentales (como sinusoidales)
amplitud_no_fund_ruido = np.sqrt(2 * no_fund_mag_ruido)

## Impresión de amplitudes de señal con ruido
print(amplitud_fund_ruido)
print(amplitud_no_fund_ruido)

## CONCLUSIÓN: A medida que aumenta el nivel de ruido los valores de amplitud de armónicos obtenidos son cada vez menos precisos
## Para poder tener una buena precisión en los valores de amplitudes de armónicos es necesario reducir el ruido lo más posible
## La presencia de ruido hace mucho la diferencia porque introduce armónicos en frecuencia que inicialmente no están y deforma los picos ya existentes

## PRUEBA CON REDUCCIÓN DE RUIDO
filtro = KalmanFilter(dim_x = len(y), dim_z = len(y))

## Matriz de transición de estados
filtro.F = 1

## Matriz de medida
filtro.H = 1

## Ruido de medición
filtro.R = 0

## Ruido de proceso
filtro.Q = 0.1

## Matriz de varianza de estado
filtro.P = 1

## Inicializo el índice que voy a utilizar para iterar
i = 1

## Inicializo el estado inicial en el valor inicial de la señal
xt = y[0]

## Creo una lista en donde me voy a guardar la señal filtrada
y_denoised = [xt]

## Mientras no se termine de iterar en toda la señal
while i < len(y):
    
    ## Predicción
    x_predict, filtro.P = predict(xt, filtro.P , filtro.F, filtro.Q)

    ## Corrección
    x_t, filtro.P  = update(x_predict, filtro.P , y[i], filtro.R, filtro.H)

    ## Aumento el valor del indice en una unidad
    i += 1

    ## Agrego el valor corregido a un vector
    y_denoised.append(x_t)

## Hago el pasaje de tipo <<list>> a tipo <<ndarray>>
y_denoised = np.array(y_denoised)

TransformadaFourier(y_denoised, T)

## Analisis armónico de la señal ruidosa
armonicos_denoised = harm_analysis(y_denoised, FS = 1 / T, n_harm = 10)

## Obtengo frecuencia fundamental de la señal
frec_fund_denoised = armonicos_denoised['fund_freq']

## Potencia de la componente fundamental de la señal
fund_db_denoised = armonicos_denoised['fund_db']

## Hago el pasaje del valor obtenido en decibeles a magnitud
fund_mag_denoised = db2mag(fund_db_denoised)

## Potencias de los armónicos no fundamentales (como sinusoidales)
no_fund_db_denoised = np.array(armonicos_denoised['pot_armonicos'])

## Hago el pasaje de los valores obtenidos en decibeles a magnitud
no_fund_mag_denoised = db2mag(no_fund_db_denoised)

## Calculo la amplitud del armónico fundamental
amplitud_fund_denoised = np.sqrt(2 * fund_mag_denoised)

## Calculo la amplitud de los armónicos no fundamentales (como sinusoidales)
amplitud_no_fund_denoised = np.sqrt(2 * no_fund_mag_denoised)

## Impresión de amplitudes denoised
print(amplitud_fund_denoised)
print(amplitud_no_fund_denoised)

## Comparo amplitudes relativas
print(amplitud_no_fund_denoised[1]/amplitud_no_fund_denoised[2])
print(amplitud_no_fund[1]/amplitud_no_fund[2])
print(amplitud_no_fund_ruido[1]/amplitud_no_fund_ruido[2])

## Graficación de la señal de error
plt.plot(x,y - y_denoised)
plt.show()