## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from SegmentacionM1 import *
import pandas as pd

## -------------------------------------- FILTRADO ACELERACIÓN -----------------------------------------

## Defino la señal de aceleración vertical como la señal medida en el eje z menos la gravedad (aproximación)
acel_vert = acel[:,1] - constants.g

## Filtro de Butterworth de orden 4, pasaaltos, frecuencia de corte 0.5Hz
## La idea es aplicar un filtro a la aceleración vertical antes de integrar para obtener velocidad
sos = signal.butter(N = 4, Wn = 0.5, btype = 'highpass', fs = 1 / periodoMuestreo, output = 'sos')

## Velocidad vertical luego de haber aplicado la etapa de filtrado pasaaltos
acel_vert_filtrada = signal.sosfiltfilt(sos, acel_vert)

## ------------------------------------- INTEGRACIÓN ACELERACIÓN ---------------------------------------

## Integro aceleración para obtener velocidad
vel_z = cumulative_trapezoid(acel_vert_filtrada, tiempo, dx = periodoMuestreo, initial = 0)

## --------------------------------------- FILTRADO VELOCIDAD ------------------------------------------

## Filtro de Butterworth de orden 4, pasaaltos, frecuencia de corte 0.5Hz
## La idea es aplicar un filtro intermedio a la velocidad vertical antes de volver a integrar para obtener posición
sos = signal.butter(N = 4, Wn = 0.5, btype = 'highpass', fs = 1 / periodoMuestreo, output = 'sos')

## Velocidad vertical luego de haber aplicado la etapa de filtrado pasaaltos
vel_z_filtrada = signal.sosfiltfilt(sos, vel_z)

## -------------------------------------- INTEGRACIÓN VELOCIDAD ----------------------------------------

## Integro velocidad para obtener posición
pos_z = cumulative_trapezoid(vel_z_filtrada, tiempo, dx = periodoMuestreo, initial = 0)

## ---------------------------------------- FILTRADO POSICIÓN ------------------------------------------

## Con el fin de eliminar la deriva hago una etapa de filtrado pasaaltos
## Etapa de filtrado pasaaltos de Butterworth con frecuencia de corte 0.5Hz de orden 4
sos = signal.butter(N = 4, Wn = 0.5, btype = 'highpass', fs = 1 / periodoMuestreo, output = 'sos')

## Calculo la posición en el eje vertical luego de hacer el filtrado
## La cantidad de tiempo que transcurre entre dos valles debe ser igual al tiempo de paso
pos_z_filtrada = signal.sosfiltfilt(sos, pos_z)

## ------------------------------------ PARÁMETROS DE OPTIMIZACIÓN -------------------------------------

## Hago la lectura del dataframe donde tengo guardado el historial de parámetros optimizados
param_optimizado = pd.read_csv("Archivo_optimización")

## Indexo la columna donde tengo guardados los registros existentes de los parámetros obtenidos
param = param_optimizado["Parametro_M2"]

## ----------------------------- CÁLCULO DE LA LONGITUD DEL PASO (MÉTODO II) ---------------------------

## Especifico la longitud de la pierna del individuo en metros
## Ésto debe considerarse como una entrada al sistema. Es un parámetro que puede medirse
## ¡IMPORTANTE: ÉSTE PARÁMETRO CAMBIA CON CADA PERSONA! SINO EL RESULTADO DA CUALQUIER COSA
long_pierna = 1

## Longitud del pie de la persona. Dato a medir y que puede variar el resultado
## Éste valor es necesario para estimar el desplazamiento en la fase de doble estancia
## Obtengo de las pruebas el valor óptimo como el promedio de los parámetros optimizados
long_pie = np.mean(param)

## Creo una lista donde voy a almacenar la longitud de los pasos de la persona
long_pasos_m2 = []

## Creo un vector de coeficientes guardando el sumando independiente de la longitud del pie
coeficientes_m2 = []

## Especifico el coeficiente multiplicativo que uso para ponderar la longitud del pie
## Los estudios sugieren usar un factor de corrección multiplicativo de 0.75 para la longitud del pie
## La idea es poder usar éste coeficiente para optimizar el modelo
factor_correccion_pie = 0.14

## Itero para cada uno de los segmentos de pasos detectados (IC a IC)
for i in range (len(pasos)):

    ## Hago la segmentación del paso en el tramo IC-TC
    tramo_IC_TC = pos_z_filtrada[pasos[i]['IC'][0] : pasos[i]['TC']]

    ## Hago la segmentación del paso en el tramo TC-IC (donde el IC es el contacto inicial opuesto)
    tramo_TC_IC = pos_z_filtrada[pasos[i]['TC'] : pasos[i]['IC'][1]]

    ## Calculo la variación de altura del centro de masa en base a la diferencia entre el desplazamiento vertical máximo y mínimo
    d_step = abs(max(tramo_TC_IC) - min(tramo_TC_IC))

    ## Calculo longitud del paso sumando la componente single stance y double stance
    long_paso = 2 * np.sqrt(2 * long_pierna * d_step - d_step ** 2) + factor_correccion_pie * long_pie

    ## Guargo la longitud de paso en la lista
    long_pasos_m2.append(long_paso)

    ## Agrego el coeficiente correspondiente
    coeficientes_m2.append(2 * np.sqrt(2 * long_pierna * d_step - d_step ** 2))

## ----------------------------- CÁLCULO DE PARÁMETROS DE MARCHA (MÉTODO II) ---------------------------

## Se calcula la longitud de paso promedio
long_paso_promedio = np.mean(long_pasos_m2)

## Se calcula la duración de paso promedio como el inverso de la cadencia
tiempo_paso_promedio = 1 / frec_fund

## Se calcula la velocidad de marcha como el cociente entre éstas cantidades
velocidad_marcha = long_paso_promedio / tiempo_paso_promedio

## Calculo la proporción de la doble estancia por paso
doble_estancia_media = np.mean(doble_estancia)

print('\nMÉTODO II')
print("Longitud de paso (m)\n  Promedio: {}\n  Desviación Estándar: {}\n  Mediana: {}".format(np.mean(long_pasos_m2), np.std(long_pasos_m2), np.median(long_pasos_m2)))
print("Duración de paso (s)\n  Promedio: {}\n  Desviación Estándar: {}\n  Mediana: {}".format(np.mean(duraciones_pasos), np.std(duraciones_pasos), np.median(duraciones_pasos)))
print("Velocidad de marcha (m/s): ", velocidad_marcha)
print("Cadencia (pasos/s): ", frec_fund)