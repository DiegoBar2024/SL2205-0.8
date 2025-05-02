## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from sereTestLib.Cinematica.ParametrosGaitPy import *
from matplotlib import pyplot as plt

## ------------------------------------- GRÁFICA LONGITUD DE PASOS -------------------------------------

## Gráfico de la longitud de los pasos en función de cada paso
plt.plot(long_pasos_m1)
plt.xlabel('Número de paso')
plt.ylabel('Longitud de paso (m)')
plt.show()

## ------------------------------------- GRÁFICA DURACIÓN DE PASOS -------------------------------------

## Gráfico de la duración de los pasos en función de cada paso
plt.plot(duraciones_pasos)
plt.xlabel('Número de paso')
plt.ylabel('Duración de paso (s)')
plt.show()

## ----------------------------------------- GRÁFICA VELOCIDAD -----------------------------------------

## Gráfico de la velocidad instantánea en función del número de pasos
plt.plot(velocidades)
plt.xlabel('Número de paso')
plt.ylabel('Velocidad instantánea (m/s)')
plt.show()

## ----------------------------------------- GRÁFICA FRECUENCIA ----------------------------------------

## Gráfico de la velocidad instantánea en función del número de pasos
plt.plot(frecuencias)
plt.xlabel('Número de paso')
plt.ylabel('Frecuencia instantánea (pasos/s)')
plt.show()