## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import numpy as np
from matplotlib import pyplot as plt
from LongitudPasoM1 import long_pasos_m1, coeficientes_m1
from LongitudPasoM2 import long_pasos_m2, coeficientes_m2, long_pie

## ------------------------------------ CÁLCULO DE ERROR (MÉTODO I) ------------------------------------

## Genero una variable que me guarde la longitud de los pasos esperada expresada en metros (valor de control)
longitud_pasos = 0.50

## Construyo un vector del tamaño de la cantidad de pasos detectados cuyos valores sean igual a la longitud especificada anteriormente
pasos_control_m1 = longitud_pasos * np.ones(np.size(long_pasos_m1))

## Calculo la señal de error de la longitud de los pasos calculada con la longitud de control
error_m1 = abs(pasos_control_m1 - long_pasos_m1)

## Grafico la señal de error en un diagrama de dispersión
plt.scatter(x = np.arange(start = 0, stop = len(long_pasos_m1)), y = error_m1)
plt.legend(["Metodo I"])
plt.show()

## ------------------------------------- OPTIMIZACIÓN (MÉTODO I) ---------------------------------------

## Calculo el error cuadrático medio actual de los pasos tomados
error_medio_m1 = np.sum(np.square(error_m1)) / len(error_m1)

## Calculo el valor óptimo del factor de corrección usando minimización de error de minimos cuadrados (ver análisis teórico)
optimo_m1 = (np.dot(pasos_control_m1, coeficientes_m1)) / (np.sum(np.square(coeficientes_m1)))

## Impresión en pantalla con el valor óptimo para el método M1
print("Valor óptimo para el método M1: {}".format(optimo_m1))

## ----------------------------------- CÁLCULO DE ERROR (MÉTODO II) ------------------------------------

## Construyo un vector del tamaño de la cantidad de pasos detectados cuyos valores sean igual a la longitud especificada anteriormente
pasos_control_m2 = longitud_pasos * np.ones(np.size(long_pasos_m2))

## Calculo la señal de error de la longitud de los pasos calculada con la longitud de control
error_m2 = abs(pasos_control_m2 - long_pasos_m2)

## Grafico la señal de error en un diagrama de dispersión
plt.scatter(x = np.arange(start = 0, stop = len(long_pasos_m2)), y = error_m2)
plt.legend(["Metodo II"])
plt.show()

## ------------------------------------- OPTIMIZACIÓN (MÉTODO II) --------------------------------------

## Igual a la optimización con el método I aplico un modelo de mínimización de error cuadrático medio
## Calculo el error cuadrático medio actual de los pasos tomados
error_medio_m2 = np.sum(np.square(error_m2)) / len(error_m2)

## Calculo el valor óptimo del factor de corrección usando minimización de error de minimos cuadrados (ver análisis teórico)
optimo_m2 =  (np.sum(np.subtract(pasos_control_m2, coeficientes_m2))) / (long_pie * len(pasos_control_m2))

## Impresión en pantalla con el valor óptimo para el método M2
print("Valor óptimo para el método M2: {}".format(optimo_m2))