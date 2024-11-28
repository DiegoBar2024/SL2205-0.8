## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from sereTestLib.Cinematica.SegmentacionM3 import *

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

## -------------------------------------- SEGMENTACIÓN DE PASOS ----------------------------------------

## Creo una lista vacía en donde voy a guardar los pasos segmentados PICO_MAXIMO - PÍCO_MINIMO
segmentada1 = []

## Creo una lista vacía en donde guardo los pasos segmentados PICO_MINIMO - PICO_MAXIMO
segmentada2 = []

## Itero para cada uno de los pasos que tengo detectados
for i in range (len(pasos)):

    ## Hago la segmentación de la señal en el tramo PICO_MAXIMO - PÍCO_MINIMO
    segmento1 = vel_z_filtrada[pasos[i]['IC'][0] : pasos[i]['TC']]

    ## Hago la segmentación de la señal en el tramo PICO_MINIMO - PICO_MAXIMO
    segmento2 = vel_z_filtrada[pasos[i]['TC'] : pasos[i]['IC'][1]]

    ## Luego lo agrego a la señal segmentada
    segmentada1.append(segmento1)

    ## Hago lo mismo con la otra parte de la señal
    segmentada2.append(segmento2)

## ------------------------------------ CALCULO LONGITUD DE PASO ---------------------------------------

## Especifico la longitud de la pierna del individuo en metros
## Ésto debe considerarse como una entrada al sistema. Es un parámetro que puede medirse
## ¡IMPORTANTE: ÉSTE PARÁMETRO CAMBIA CON CADA PERSONA! SINO EL RESULTADO DA CUALQUIER COSA
long_pierna = 0.9

## Creo una lista donde voy a almacenar la longitud de los pasos de la persona
long_pasos = []

## Itero para cada uno de los segmentos de pasos detectados
for i in range (len(pasos)):

    ## En caso de que no se haya detectado ningún pico mínimo, digo que continúe
    if pasos[i]['TC'] == 0:

        continue

    ## Hago la integral de la velocidad en el intervalo (PICO_MAXIMO, PICO_MINIMO) considerando condiciones iniciales nulas
    desp_1 = cumulative_trapezoid(segmentada1[i], tiempo[pasos[i]['IC'][0] : pasos[i]['TC']], dx = periodoMuestreo, initial = 0)

    ## Hago la integral de la velocidad en el intervalo (PICO_MINIMO, PICO_MAXIMO) considerando condiciones iniciales nulas
    desp_2 = cumulative_trapezoid(segmentada2[i], tiempo[pasos[i]['TC'] : pasos[i]['IC'][1]], dx = periodoMuestreo, initial = 0)

    ## Aplico la fórmula para obtener la longitud del paso
    ## Aplico mismo factor de corrección de 1.25 para corregir las longitudes del paso (conclusión paper Zijlstra)
    long_paso = 1.25 * (np.sqrt(long_pierna ** 2 - (long_pierna - abs(desp_1[-1])) ** 2) + np.sqrt(long_pierna ** 2 - (long_pierna - abs(desp_2[-1])) ** 2))

    ## Agrego el paso calculado a la lista de pasos
    long_pasos.append(long_paso)

## --------------------------------- CÁLCULO DE PARÁMETROS DE MARCHA  ----------------------------------

## Se calcula la longitud de paso promedio
long_paso_promedio = np.mean(long_pasos)

## Se calcula la duración de paso promedio como el inverso de la cadencia
tiempo_paso_promedio = 1 / frec_fund

## Se calcula la velocidad de marcha como el cociente entre éstas cantidades
velocidad_marcha = long_paso_promedio / tiempo_paso_promedio

## Calculo la proporción de la doble estancia por paso
doble_estancia_media = np.mean(doble_estancia)

print("Longitud de paso (m)\n  Promedio: {}\n  Desviación Estándar: {}\n  Mediana: {}".format(np.mean(long_pasos), np.std(long_pasos), np.median(long_pasos)))
print("Duración de paso (s)\n  Promedio: {}\n  Desviación Estándar: {}\n  Mediana: {}".format(np.mean(duraciones_pasos), np.std(duraciones_pasos), np.median(duraciones_pasos)))
print("Velocidad de marcha (m/s): ", velocidad_marcha)
print("Cadencia (pasos/s): ", frec_fund)

## ----------------------------------- GRAFICACIÓN LONGITUD DE PASOS -----------------------------------

plt.scatter(x = np.arange(start = 0, stop = len(long_pasos)), y = long_pasos)
plt.show()