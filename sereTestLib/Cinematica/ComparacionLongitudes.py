## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from Segmentacion import *
from LongitudPaso import long_pasos_m1 as long_pasos11, long_pasos_m2 as long_pasos12
from LongitudPasoEMD import long_pasos_m1 as long_pasos21, long_pasos_m2 as long_pasos22
from LongitudPasoVertical import long_pasos as long_pasosVERT

## -------------------------------------- GRAFICACIÓN DE ERRORES ---------------------------------------

plt.scatter(x = np.arange(start = 0, stop = len(long_pasos11)), y = np.subtract(long_pasos11, long_pasos21))
plt.show()