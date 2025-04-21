## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from Normalizacion import Normalizacion
from DeteccionPicos import *
from Fourier import TransformadaFourier
import numpy as np
import sys
import pandas as pd
from scipy import signal
from harm_analysis import *
from control import *
from LecturaDatos import *

## ------------------------------------- CENTRADO DE ACELERACIONES -------------------------------------

## Obtengo la aceleración mediolateral centrada
acc_ML = acel[:, 0] - np.mean(acel[:, 0])

## Obtengo la aceleración vertical centrada
acc_VT = acel[:, 1] - np.mean(acel[:, 1])

## Obtengo la aceleración anteroposterior centrada
acc_AP = - acel[:, 2] - np.mean( - acel[:, 2])

## --------------------------------------- GRAFICACIÓN DE DATOS ----------------------------------------

## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
plt.plot(tiempo, acc_ML, color = 'r', label = '$a_{ML}$')
plt.plot(tiempo, acc_VT, color = 'b', label = '$a_{VT}$')
plt.plot(tiempo, acc_AP, color = 'g', label = '$a_{AP}$')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleracion $(m/s^2)$")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()

## ---------------------------------------- FILTRADO DE MEDIANA ----------------------------------------

## Aplico filtro de mediana de la aceleración mediolateral
acc_ML = signal.medfilt(acc_ML, 5)

## Aplico filtro de mediana de la aceleración vertical
acc_VT = signal.medfilt(acc_VT, 5)

## Aplico filtro de mediana de la aceleración anteroposterior
acc_AP = signal.medfilt(acc_AP, 5)

## --------------------------------------- GRAFICACIÓN DE DATOS ----------------------------------------

## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
plt.plot(tiempo, acc_ML, color = 'r', label = '$a_{ML}$')
plt.plot(tiempo, acc_VT, color = 'b', label = '$a_{VT}$')
plt.plot(tiempo, acc_AP, color = 'g', label = '$a_{AP}$')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleracion $(m/s^2)$")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()

## ------------------------------------ NORMALIZACIÓN EN AMPLITUD --------------------------------------

## Hago normalización en amplitud de la aceleración mediolateral
acc_ML = acc_ML / np.max(np.abs(acc_ML))

## Hago normalización en amplitud de la aceleración vertical
acc_VT = acc_VT / np.max(np.abs(acc_VT))

## Hago normalización en amplitud de la aceleración anteroposterior
acc_AP = acc_AP / np.max(np.abs(acc_AP))

## -------------------------------------- GRAFICACIÓN DE DATOS -----------------------------------------

## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
plt.plot(tiempo, acc_ML, color = 'r', label = '$a_{ML}$')
plt.plot(tiempo, acc_VT, color = 'b', label = '$a_{VT}$')
plt.plot(tiempo, acc_AP, color = 'g', label = '$a_{AP}$')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
plt.xlabel("Tiempo (s)")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()

plt.plot(acc_AP)
plt.show()