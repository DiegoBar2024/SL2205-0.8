## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Wavelets')
from LecturaDatos import *
from LecturaDatosPacientes import *
from ContactosIniciales import *
from ContactosTerminales import *
from Segmentacion import *
from Escalogramas import *
from Escalado import *
from tsfel import *

## -------------------------------------------- ETIQUETADO ---------------------------------------------

## Construyo una lista con todos aquellos pacientes denominados estables no añosos
id_estables_no_añosos = np.array([114, 127, 128, 129, 130, 133, 213, 224, 226, 44, 294])

## Construyo una lista con todos aquellos pacientes denominados estables añosos
## En principio estos pacientes se consideran como estables pero se van a mantener por separado del análisis
id_estables_añosos = np.array([67, 77, 111, 112, 115, 216, 229, 271, 273])

## Obtengo una lista con los identificadores de todos los pacientes estables
id_estables = np.concatenate([id_estables_añosos, id_estables_no_añosos])

## Construyo una lista con aquellos pacientes denominados inestables
id_inestables = np.array([69, 72, 90, 122, 137, 139, 142, 144, 148, 149, 158, 167, 178, 221, 223, 232, 256])

## Construyo una lista con los IDs de aquellos pacientes los cuales yo sé que están etiquetados
id_etiquetados = np.concatenate([id_estables, id_inestables])

## --------------------------------------- PROCESAMIENTO DE GIROS --------------------------------------

## Hago la lectura de los datos generales de los pacientes
pacientes, ids_existentes = LecturaDatosPacientes()

## Obtengo todos aquellos pacientes para los cuales yo tenga registros de marcha y además estén etiquetados
etiquetados_existentes = np.intersect1d(id_etiquetados, ids_existentes)

## Itero para cada uno de los identificadores de los pacientes
for id_persona in etiquetados_existentes:

    ## Creo un bloque try catch en caso de que ocurra algún error en el procesamiento
    try:

        ## Hago la lectura de los datos del registro de marcha del paciente
        data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona)

        ## Cálculo de contactos iniciales
        contactos_iniciales, muestras_paso, acc_AP_norm, frec_fund = ContactosIniciales(acel, cant_muestras, periodoMuestreo, graficar = False)

        ## Cálculo de contactos terminales
        contactos_terminales = ContactosTerminales(acel, cant_muestras, periodoMuestreo, graficar = False)

        ## Hago la segmentación de la marcha
        pasos, duraciones_pasos, giros = Segmentacion(contactos_iniciales, contactos_terminales, muestras_paso, periodoMuestreo, acc_AP_norm, gyro)

        ## Genero una lista en donde voy a guardar las subsecuencias asociadas a los pasos
        subsec_pasos = [[]]

        ## Itero para cada uno de los pasos que tengo detectados
        for paso in pasos:

            ## Agrego el paso a la subsecuencia de pasos
            subsec_pasos[-1].append(paso)

            ## Itero para cada uno de los giros detectados
            for giro in giros:

                ## En caso de que el paso pertenezca a un giro
                if paso['IC'][0] >= giro[0] and paso['IC'][1] <= giro[1]:

                    ## Elimino el paso que coloqué en la lista ya que pertenece a un giro
                    subsec_pasos[-1].pop(-1)

                    ## En caso de que el paso sea el primero que está incluído en un giro
                    if paso['IC'][0] == giro[0]:

                        ## Agrego una lista vacía al final de la lista de subsecuencias de pasos
                        subsec_pasos.append([])

                    ## Termino la ejecución del bucle ya que encontré un giro en el que está incluído
                    break

        ## Especifico la ruta en la cual yo voy a hacer el guardado de los escalogramas
        ruta = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_sin_giros'

        ## Itero para cada una de las subsecuencias de pasos que segmenté
        for i in range (len(subsec_pasos)):

            ## Impresión de pantalla avisando numero de segmento                    
            print("Segmento {} de {}".format(i + 1, len(subsec_pasos)))

            ## Hago el cálculo de los escalogramas correspondientes
            escalogramas_segmentos, directorio_muestra, nombre_base_segmento = Escalogramas(id_persona, tiempo, subsec_pasos[i], cant_muestras, acel, gyro, periodoMuestreo)

            ## Hago el escalado y el guardado correspondiente de los escalogramas
            Escalado(escalogramas_segmentos, directorio_muestra + 'Tramo{}/'.format(i), nombre_base_segmento, ruta)
        
        ## Impresión en pantalla avisando el identificador del paciente que está siendo procesado
        print("ID del paciente que está siendo procesado: {}".format(id_persona))

    ## En caso de que ocurra un error en el procesamiento
    except:

        ## Sigo con el paciente que sigue
        continue