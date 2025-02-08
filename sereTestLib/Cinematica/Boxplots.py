## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import numpy as np
from matplotlib import pyplot as plt
from LongitudPasoM1 import long_pasos_m1
from LongitudPasoM2 import long_pasos_m2

## -------------------------------------- GRÁFICOS DE COMPARACIÓN --------------------------------------

## Creo un diccionario conteniendo los nombres y las listas con las longitudes calculadas de los pasos
pasos = {'M1': long_pasos_m1, 'M2': long_pasos_m2}

## Hago el boxplot comparando éstos dos métodos
plt.boxplot(pasos.values(), tick_labels = pasos.keys())
plt.show()