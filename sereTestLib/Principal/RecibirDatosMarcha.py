## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
import pandas as pd
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Presentacion')
import os
from LecturaDatos import *
from ContactosIniciales import *
from ContactosTerminales import *
from Segmentacion import *
from LongitudPasoM1 import *
from GeneracionReporte import *

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Se pide al usuario el nombre del archivo que se va a cargar
nombre_archivo = input("Nombre del archivo: ")

## Se pide al usuario el identificador del paciente
id_persona = int(input("ID del paciente: "))

## Se pide al usuario el nombre del paciente
nombre_persona = input("Nombre del paciente (<<Nombre>> <<Apellido>>): ")

## Se pide al usuario la fecha de nacimiento del paciente
nacimiento_persona = input("Fecha de nacimiento del paciente: (<<DD/MM/YYYY>>): ")

## Se pide al usuario la longitud de la pierna de la persona (en metros)
longitud_pierna = float(input("Longitud de la pierna (m): "))

## Se pide al usuario la longitud del pie de la persona (en centímetros)
## Divido entre 100 para poder pasar los centímetros a metros
longitud_pie = float(input("Longitud del pie (cm): ")) / 100

## Especifico la ruta donde se van a encontrar todos los registros (se puede cambiar)
ruta_registro = "C:/Yo/Tesis/sereData/sereData/Registros/{}.txt".format(nombre_archivo)

## Hago la lectura del registro de datos asociados a la persona
data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = None, lectura_datos_propios = True, ruta = ruta_registro)

## ----------------------------------------- ANÁLISIS DE MARCHA -----------------------------------------

## Cálculo de contactos iniciales
contactos_iniciales, muestras_paso, acc_AP_norm, frec_fund = ContactosIniciales(acel, cant_muestras, periodoMuestreo, graficar = False)

## Cálculo de contactos terminales
contactos_terminales = ContactosTerminales(acel, cant_muestras, periodoMuestreo, graficar = False)

## Hago la segmentación de la marcha
pasos, duraciones_pasos = Segmentacion(contactos_iniciales, contactos_terminales, muestras_paso, periodoMuestreo, acc_AP_norm, gyro)

## Cálculo de parámetros de marcha
pasos_numerados, frecuencias, velocidades, long_pasos_m1, coeficientes_m1 = LongitudPasoM1(pasos, acel, tiempo, periodoMuestreo, frec_fund, duraciones_pasos, longitud_pierna)

## --------------------------------------- GENERACIÓN DEL REPORTE -----------------------------------------

## Especifico la ruta en la cual se va a guardar el PDF con el reporte generado
ruta_guardado = "C:/Yo/Tesis/sereData/sereData/Reportes/{}".format(id_persona)

## Especifico el nombre con el que se va a guardar el reporte
nombre_reporte = 'Reporte'

## En caso de que la ruta donde se va a guardar el reporte no exista
if not os.path.exists(ruta_guardado):

    ## Creo la ruta correspondiente
    os.makedirs(ruta_guardado)

## Hago la creación del reporte en PDF con los resultados del análisis de marcha
CreacionReporte(id_persona, nombre_persona, nacimiento_persona, tiempo, long_pasos_m1, duraciones_pasos, velocidades, frecuencias, pasos_numerados, ruta_guardado + '/{}.pdf'.format(nombre_reporte))