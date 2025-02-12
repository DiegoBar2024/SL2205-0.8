## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from LecturaDatos import *
from Magnitud import Magnitud

## --------------------------------------- CÁLCULO DE MAGNITUD -----------------------------------------

## El criterio de detección de giros involucra mirar la magnitud de las señales de giroscopios
## Hago el cálculo de la magnitud de las señales de giroscopio
magnitud_gyro = Magnitud(gyro)

## ---------------------------------- DETECCIÓN DE INSTANTE DE GIRO ------------------------------------

## El instante de giro será aquel en que se produzca el pico mayor de la magnitud de los giroscopios
## Hallo el índice correspondiente a dicho valor, correspondiente al numero de muestra donde se da el giro
muestra_giro =  np.argmax(magnitud_gyro)

## Determino el tiempo en que se produce el giro dividiendo por la frecuencia de muestreo
tiempo_giro = muestra_giro * periodoMuestreo

## ------------------------------------- GRAFICACIÓN DE MAGNITUD ---------------------------------------

## Grafico la magnitud del vector de giroscopio indicando el instante del giro
plt.plot(magnitud_gyro)
plt.plot(muestra_giro, magnitud_gyro[muestra_giro], "o")
plt.show()
