## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from LecturaDatos import *
from LecturaDatosPacientes import *
from ContactosIniciales import *
from ContactosTerminales import *
from Segmentacion import *
from tsfel import *

## ------------------------------------ GENERACIÓN DE ESCALOGRAMAS --------------------------------------

## Hago la lectura de los datos generales de los pacientes
pacientes, ids_existentes = LecturaDatosPacientes()

## Especifico la ruta en la cual yo voy a hacer el guardado de los escalogramas
ruta = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_giros_tot'

## Itero para cada uno de los identificadores de los pacientes
for id_persona in ids_existentes:

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

        ## Segmento las aceleraciones en el tramo de giro
        acel_giro = acel[giros[0][0] : giros[0][1]]

        ## Segmento los giroscopios en el tramo de giro
        gyro_giro = gyro[giros[0][0] : giros[0][1]]

    ## En caso de que ocurra un error en el procesamiento
    except:

        ## Sigo con el paciente que sigue
        continue