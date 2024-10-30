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

ruta = "C:/Yo/Tesis/sereData/sereData/Dataset/dataset/S268/3S268.csv"

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

## Graficación como valores de dispersión de las señales de separación de tiempos
plt.axhline(y = tiempo_medio + desv_tiempos, linestyle = '-', color = 'r')
plt.axhline(y = tiempo_medio, linestyle = '-', color = 'b')
plt.axhline(y = tiempo_medio - desv_tiempos, linestyle = '-', color = 'g')
plt.scatter(x = np.arange(start = 0, stop = len(sep_tiempos)), y = sep_tiempos)
plt.show()

## La idea es ahora en base a la dispersión eliminar ciertos valores y quedarme con aquellos que estén más concentrados (próximos a la media)
## O sea se deciden eliminar aquellos valores que estén muy alejados interpretándose como errores en el momento de hacer la detección de picos
## Se explicita la condición de filtrado de que me quedo con aquellos tiempos que tengan una separación menor a la media + 0.5 * desviacion estándar
condicion_filtrado_1 = sep_tiempos < tiempo_medio + 0.5 * desv_tiempos

## Se hace el filtrado de aquellos tiempos que estén en el intervalo definido antes
filtrado_tiempos_1 = sep_tiempos[condicion_filtrado_1]

## Se explicita la condición de filtrado de que me quedo con aquellos tiempos que tengan una separación mayor a la media - 0.5 * desviacion estándar
condicion_filtrado_2 = filtrado_tiempos_1 > tiempo_medio - 0.5 * desv_tiempos

## Se hace el filtrado de aquellos tiempos que estén en el intervalo definido antes
filtrado_tiempos = filtrado_tiempos_1[condicion_filtrado_2]

## Se calcula el valor medio luego de realizar el filtrado
media_filtrado = np.mean(filtrado_tiempos)

## Se calcula la desviación estándar luego de realizar el filtrado
desv_filtrado = np.std(filtrado_tiempos)

## Obtengo la proporción de separaciones que se encuentran dentro del rango (media - desv_estandar, media + desv_estandar)
## Cuanto mayor sea éste índice mejor ya que va a ser más constante el tiempo de separación de los pasos y es más consistente
## Se puede poner como valor aceptable para interpretar ésto un valor de índice mínimo de 0.75
muestras_interior = filtrado_tiempos[media_filtrado - desv_filtrado < filtrado_tiempos]

## Hago la segunda parte del filtrado para evitar que no se contabilicen datos repetidos
muestras_interior = muestras_interior[media_filtrado + desv_filtrado > muestras_interior]

## Calculo la proporción de muestras en el interior del rango comparado con el exterior
proporcion_int_ext = len(muestras_interior) / len(filtrado_tiempos)

print(proporcion_int_ext)
print(media_filtrado)
print(desv_filtrado)

## Graficación de la dispersión de los puntos de la señal filtrada
plt.axhline(y = media_filtrado + desv_filtrado, linestyle = '-', color = 'r')
plt.axhline(y = media_filtrado, linestyle = '-', color = 'b')
plt.axhline(y = media_filtrado - desv_filtrado, linestyle = '-', color = 'g')
plt.scatter(x = np.arange(start = 0, stop = len(filtrado_tiempos)), y = filtrado_tiempos)
plt.show()

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