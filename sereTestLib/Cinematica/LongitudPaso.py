## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from Segmentacion import *

## ------------------------------------- INTEGRACIÓN ACELERACIÓN ---------------------------------------

## Defino la señal de aceleración vertical como la señal medida en el eje z menos la gravedad (aproximación)
acel_vert = acel[:,1] - constants.g

## Integro aceleración para obtener velocidad
vel_z = cumulative_trapezoid(acel_vert, tiempo, dx = periodoMuestreo, initial = 0)

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

## -------------------------------------- SEGMENTACIÓN DE PASOS ----------------------------------------

## Creo una lista vacía en donde voy a guardar los pasos segmentados
segmentada = []

## Itero para cada uno de los pasos que tengo detectados
for i in range (len(pasos)):

    ## Hago la segmentación de la señal
    segmento = pos_z_filtrada[pasos[i]['IC'][0] : pasos[i]['IC'][1]]

    plt.plot(segmento)
    plt.show()

    ## Luego lo agrego a la señal segmentada
    segmentada.append(segmento)

## -------------------------------- VARIACIÓN DE ALTURA CENTRO DE MASA ---------------------------------

## Creo una lista donde voy a almacenar las variaciones de altura del centro de masa en cada tramo
desp_vert_COM = []

## Itero para cada uno de los segmentos de pasos detectados
for i in range (len(segmentada)):

    ## Calculo la variación de altura del centro de masa en base a la diferencia entre el desplazamiento vertical máximo y mínimo
    d_step = abs(max(segmentada[i]) - min(segmentada[i]))

    ## Agrego el desplazamiento máximo del COM calculado a la lista
    desp_vert_COM.append(d_step)

## ---------------------------- GRAFICACIÓN VARIACIÓN ALTURA CENTRO DE MASA ----------------------------

# plt.scatter(x = np.arange(start = 0, stop = len(desp_vert_COM)), y = desp_vert_COM)
# plt.show()

## ----------------------------- CÁLCULO DE LA LONGITUD DEL PASO (MÉTODO I) ----------------------------

## Especifico la longitud de la pierna del individuo en metros
## Ésto debe considerarse como una entrada al sistema. Es un parámetro que puede medirse
## ¡IMPORTANTE: ÉSTE PARÁMETRO CAMBIA CON CADA PERSONA! SINO EL RESULTADO DA CUALQUIER COSA
long_pierna = 0.9

## Creo una lista donde voy a almacenar la longitud de los pasos de la persona
long_pasos_m1 = []

## Itero para cada uno de los segmentos de pasos detectados (IC a IC)
for i in range (len(segmentada)):
    
    ## Calculo la longitud del paso con la fórmula sugerida por Zijlstra
    ## Aplico Factor de Corrección multiplicativo de 1.25 (recomendación para no subestimar longitud pasos)
    long_paso = 1.25 * 2 * np.sqrt(2 * long_pierna * desp_vert_COM[i] - desp_vert_COM[i] ** 2)

    ## Agrego el paso a la lista de longitud de pasos
    long_pasos_m1.append(long_paso)

## ----------------------------- CÁLCULO DE PARÁMETROS DE MARCHA (MÉTODO I) ----------------------------

## Se calcula la longitud de paso promedio
long_paso_promedio = np.mean(long_pasos_m1)

## Se calcula la duración de paso promedio como el inverso de la cadencia
tiempo_paso_promedio = 1 / frec_fund

## Se calcula la velocidad de marcha como el cociente entre éstas cantidades
velocidad_marcha = long_paso_promedio / tiempo_paso_promedio

## Calculo la proporción de la doble estancia por paso
doble_estancia_media = np.mean(doble_estancia)

print('MÉTODO I')
print("Longitud de paso (m)\n  Promedio: {}\n  Desviación Estándar: {}\n  Mediana: {}".format(np.mean(long_pasos_m1), np.std(long_pasos_m1), np.median(long_pasos_m1)))
print("Duración de paso (s)\n  Promedio: {}\n  Desviación Estándar: {}\n  Mediana: {}".format(np.mean(duraciones_pasos), np.std(duraciones_pasos), np.median(duraciones_pasos)))
print("Velocidad de marcha (m/s): ", velocidad_marcha)
print("Cadencia (pasos/s): ", frec_fund)

## ----------------------------------- GRAFICACIÓN LONGITUD DE PASOS -----------------------------------

plt.scatter(x = np.arange(start = 0, stop = len(long_pasos_m1)), y = long_pasos_m1)
plt.show()

## ----------------------------- CÁLCULO DE LA LONGITUD DEL PASO (MÉTODO II) ---------------------------

## Longitud del pie de la persona. Dato a medir y que puede variar el resultado
## Éste valor es necesario para estimar el desplazamiento en la fase de doble estancia
## Si no tengo un valor concreto tomo por defecto 30cm
long_pie = 0.25

## Creo una lista donde voy a almacenar la longitud de los pasos de la persona
long_pasos_m2 = []

## Itero para cada uno de los segmentos de pasos detectados (IC a IC)
for i in range (len(pasos)):

    ## Hago la segmentación del paso en el tramo IC-TC
    tramo_IC_TC = pos_z_filtrada[pasos[i]['IC'][0] : pasos[i]['TC']]

    ## Hago la segmentación del paso en el tramo TC-IC (donde el IC es el contacto inicial opuesto)
    tramo_TC_IC = pos_z_filtrada[pasos[i]['TC'] : pasos[i]['IC'][1]]

    ## Calculo la variación de altura del centro de masa en base a la diferencia entre el desplazamiento vertical máximo y mínimo
    d_step = abs(max(tramo_TC_IC) - min(tramo_TC_IC))

    ## Calculo longitud del paso sumando la componente single stance y double stance
    long_paso = 2 * np.sqrt(2 * long_pierna * d_step - d_step ** 2) + 0.75 * long_pie

    ## Guargo la longitud de paso en la lista
    long_pasos_m2.append(long_paso)

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