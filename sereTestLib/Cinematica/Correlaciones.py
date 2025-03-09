## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from scipy.signal import *
from LecturaDatos import *

## ------------------------------------- CÁLCULO DE CORRELACIONES --------------------------------------

## La idea es poder calcular correlaciones existentes entre señales de aceleración en distintos ejes
## Calculo de correlación entre las señales de aceleración en el eje x con el eje y
corr_ACx_ACy = correlate(acel[:,0], acel[:,1], mode = 'same')

## Calculo de correlación entre las señales de aceleración en el eje x con el eje z
corr_ACx_ACz = correlate(acel[:,0], acel[:,2], mode = 'same')

## Calculo de correlacion entre las señales de aceleración en el eje y con el eje z
corr_ACy_ACz = correlate(acel[:,1], acel[:,2], mode = 'same')

## ------------------------------------- GRÁFICA DE CORRELACIONES --------------------------------------

## Graficacion de correlaciones
plt.plot(tiempo, corr_ACx_ACy, label = 'Correlacion X-Y')
plt.plot(tiempo, corr_ACx_ACz, label = 'Correlacion X-Z')
plt.plot(tiempo, corr_ACy_ACz, label = 'Correlacion Y-Z')
plt.legend()
plt.show()