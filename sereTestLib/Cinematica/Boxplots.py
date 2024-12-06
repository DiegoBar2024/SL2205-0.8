## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import numpy as np
from matplotlib import pyplot as plt
from LongitudPasoM1 import long_pasos_m1
from LongitudPasoM2 import long_pasos_m2
from LongitudPasoM3 import long_pasos_m3

## -------------------------------------- GRÁFICOS DE COMPARACIÓN --------------------------------------

pasos = {'M1': long_pasos_m1, 'M2': long_pasos_m2, 'M3' : long_pasos_m3}

plt.boxplot(pasos.values(), labels = pasos.keys())
plt.show()