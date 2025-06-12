## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from Segmentacion import *
import pandas as pd
from scipy.integrate import cumulative_trapezoid
import json

## ---------------------------------- CÁLCULO DE PARÁMETROS DE MARCHA ----------------------------------

def LongitudPasoM2(pasos, acel, tiempo, periodoMuestreo, frec_fund, duraciones_pasos, id_persona, long_pierna = 1, long_pie = 0.03):

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

    ## ------------------------------------- SEGMENTACIÓN DE PASOS -------------------------------------

    ## Creo una lista vacía en donde voy a guardar los pasos segmentados
    segmentada = []

    ## Itero para cada uno de los pasos que tengo detectados
    for i in range (len(pasos)):

        ## Hago la segmentación de la señal
        segmento = pos_z_filtrada[pasos[i]['IC'][0] : pasos[i]['IC'][1]]

        ## Luego lo agrego a la señal segmentada
        segmentada.append(segmento)

    ## ----------------------------- CARGADO DEL PARÁMETRO DE LONGITUD DE PASO -----------------------------

    ## Hago la lectura del archivo JSON previamente existente
    with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/OptimizacionM2.json", 'r') as openfile:

        # Cargo el diccionario el cual va a ser un objeto JSON
        dicc_optimizacion = json.load(openfile)
    
    ## En caso de que el paciente tenga parámetros ya optimizados
    if str(id_persona) in list(dicc_optimizacion.keys()):
        
        ## Obtengo la lista de parámetros optimizados asociados al ID de la persona
        parametros = dicc_optimizacion[str(id_persona)]
    
    ## En caso de que el paciente no tenga parámetros correspondientes
    else:

        ## Seteo el factor de corrección a 0.75 por defecto
        parametros = 0.75

    ## ----------------------------- CÁLCULO DE LA LONGITUD DEL PASO (MÉTODO II) ---------------------------

    ## Especifico la longitud de la pierna del individuo en metros
    ## Ésto debe considerarse como una entrada al sistema. Es un parámetro que puede medirse
    ## ¡IMPORTANTE: ÉSTE PARÁMETRO CAMBIA CON CADA PERSONA! SINO EL RESULTADO DA CUALQUIER COSA
    long_pierna = long_pierna

    ## Longitud del pie de la persona. Dato a medir y que puede variar el resultado
    ## Éste valor es necesario para estimar el desplazamiento en la fase de doble estancia
    ## Obtengo de las pruebas el valor óptimo como el promedio de los parámetros optimizados
    long_pie = long_pie

    ## Creo una lista donde voy a almacenar la longitud de los pasos de la persona
    long_pasos_m2 = []

    ## Creo un vector de coeficientes guardando el sumando independiente de la longitud del pie
    coeficientes_m2 = []

    ## Especifico el coeficiente multiplicativo que uso para ponderar la longitud del pie
    ## Los estudios sugieren usar un factor de corrección multiplicativo de 0.75 para la longitud del pie
    ## La idea es poder usar éste coeficiente para optimizar el modelo
    factor_correccion_pie = np.mean(parametros)

    ## Itero para cada uno de los segmentos de pasos detectados (IC a IC)
    for i in range (len(pasos)):

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

    ## Obtengo un vector que tenga los pasos numerados
    pasos_numerados = np.arange(0, len(pasos), 1)

    ## Se calcula la longitud de paso promedio
    long_paso_promedio = np.mean(long_pasos_m2)

    ## Se calcula la duración de paso promedio como el inverso de la cadencia
    tiempo_paso_promedio = 1 / frec_fund

    ## Se calcula la velocidad de marcha como el cociente entre éstas cantidades
    velocidad_marcha = long_paso_promedio / tiempo_paso_promedio

    print('\nMÉTODO II')
    print("Longitud de paso (m)\n  Promedio: {}\n  Desviación Estándar: {}\n  Mediana: {}".format(np.mean(long_pasos_m2), np.std(long_pasos_m2), np.median(long_pasos_m2)))
    print("Duración de paso (s)\n  Promedio: {}\n  Desviación Estándar: {}\n  Mediana: {}".format(np.mean(duraciones_pasos), np.std(duraciones_pasos), np.median(duraciones_pasos)))
    print("Velocidad de marcha (m/s): ", velocidad_marcha)
    print("Cadencia (pasos/s): ", frec_fund)

    ## --------------------------------- CÁLCULO DE VELOCIDAD INSTANTÁNEA ----------------------------------

    ## Creo un vector donde voy a almacenar las velocidades instantáneas con una resolución de un paso
    velocidades = []

    ## Itero para cada uno de los segmentos de pasos detectados (IC a IC)
    for i in range (len(segmentada)):

        ## Obtengo la velocidad instantánea como el cociente entre la longitud estimada del paso y la duración del paso
        ## Se entiende que ésto es un estimador de la velocidad instantánea con una resolución de un paso
        vel_instantanea = long_pasos_m2[i] / duraciones_pasos[i]

        ## Agrego la velocidad instantánea asociada al i-ésimo paso a la lista de velocidades
        velocidades.append(vel_instantanea)

    ## --------------------------------- CÁLCULO DE FRECUENCIA INSTANTÁNEA ---------------------------------

    ## Creo un vector donde voy a almacenar las frecuencias instantáneas de todos los pacientes
    frecuencias = []

    ## Itero para cada uno de los segmentos de pasos detectados (IC a IC)
    for i in range (len(segmentada)):

        ## Obtengo la frecuencia instantánea de la marcha como el inverso de la duración del paso
        frec_instantanea = 1 / duraciones_pasos[i]

        ## Agregeo la frecuencia instantanea a la lista de frecuencias
        frecuencias.append(frec_instantanea)

    ## Retorno los parámetros de marcha calculados
    return pasos_numerados, frecuencias, velocidades, long_pasos_m2, coeficientes_m2