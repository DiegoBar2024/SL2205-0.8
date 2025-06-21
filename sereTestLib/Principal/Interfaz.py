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

## ------------------------------------- GENERACIÓN DE INTERFAZ --------------------------------------

import tkinter as tk
import tkinter.filedialog

root = tk.Tk(baseName = "Programa")

# setting the windows size
root.geometry("600x400")

## Defino el objeto correspondiente a la ID de la persona
ID_var = tk.StringVar()

## Defino el objeto correspondiente al nombre de la persona
nombre_var = tk.StringVar()

## Defino el objeto correspondiente al nacimiento de la persona
nacimiento_var = tk.StringVar()

## Defino el objeto correspondiente a la longitud de la pierna de la persona
pierna_var = tk.StringVar()

## Defino el objeto correspondiente a la longitud del pie de la persona
pie_var = tk.StringVar()

def PedirDirectorio():

    global ruta_registro

    ruta_registro = tkinter.filedialog.askopenfile().name

def RealizarAnalisis():

    ## Obtengo el valor de la ID de la persona
    id_persona = ID_var.get()

    ## Obtengo el nombre de la persona
    nombre_persona = nombre_var.get()

    ## Obtengo la fecha de nacimiento de la persona
    nacimiento_persona = nacimiento_var.get()

    ## Obtengo la longitud de la pierna de la persona
    longitud_pierna = float(pierna_var.get())

    ## Hago la lectura del registro de datos asociados a la persona
    data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = None, lectura_datos_propios = True, ruta = ruta_registro)

    ## Cálculo de contactos iniciales
    contactos_iniciales, muestras_paso, acc_AP_norm, frec_fund = ContactosIniciales(acel, cant_muestras, periodoMuestreo, graficar = False)

    ## Cálculo de contactos terminales
    contactos_terminales = ContactosTerminales(acel, cant_muestras, periodoMuestreo, graficar = False)

    ## Hago la segmentación de la marcha
    pasos, duraciones_pasos = Segmentacion(contactos_iniciales, contactos_terminales, muestras_paso, periodoMuestreo, acc_AP_norm, gyro)

    ## Cálculo de parámetros de marcha usando el método I
    pasos_numerados, frecuencias, velocidades, long_pasos_m1, coeficientes_m1 = LongitudPasoM1(pasos, acel, tiempo, periodoMuestreo, frec_fund, duraciones_pasos, id_persona, longitud_pierna)

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

## Variable de etiqueta para la ID del paciente
etiqueta_ID = tk.Label(root, text = 'ID del paciente: ', font = ('calibre', 10, 'bold'))

## Variable de entrada para la ID del paciente
entrada_ID = tk.Entry(root,textvariable = ID_var, font = ('calibre', 10, 'normal'))

## Variable de etiqueta para el nombre del paciente
etiqueta_nombre = tk.Label(root, text = 'Nombre del paciente (<<Nombre>> <<Apellido>>): ', font = ('calibre',10, 'bold'))

entrada_nombre = tk.Entry(root,textvariable = nombre_var, font=('calibre',10,'normal'))

etiqueta_nacimiento = tk.Label(root, text = 'Fecha de nacimiento del paciente: (<<DD/MM/YYYY>>): ', font=('calibre',10, 'bold'))

entrada_nacimiento = tk.Entry(root,textvariable = nacimiento_var, font=('calibre',10,'normal'))

etiqueta_pierna = tk.Label(root, text = 'Longitud de la pierna (m): ', font=('calibre',10, 'bold'))

entrada_pierna = tk.Entry(root,textvariable = pierna_var, font=('calibre',10,'normal'))

etiqueta_pie = tk.Label(root, text = 'Longitud del pie (cm): ', font=('calibre',10, 'bold'))

entrada_pie = tk.Entry(root,textvariable = pie_var, font=('calibre',10,'normal'))

boton_directorio = tk.Button(root, text = 'Seleccionar Archivo', command = PedirDirectorio)

# creating a button using the widget 
# Button that will call the submit function 
boton_comenzar = tk.Button(root,text = 'Comenzar Análisis', command = RealizarAnalisis)

## Botones
etiqueta_ID.grid(row = 0,column = 0)
entrada_ID.grid(row = 0,column = 1)
etiqueta_nombre.grid(row = 1,column = 0)
entrada_nombre.grid(row = 1,column = 1)
etiqueta_nacimiento.grid(row = 2,column = 0)
entrada_nacimiento.grid(row = 2,column = 1)
etiqueta_pierna.grid(row = 3,column = 0)
entrada_pierna.grid(row = 3,column = 1)
etiqueta_pie.grid(row = 4,column = 0)
entrada_pie.grid(row = 4,column = 1)
boton_directorio.grid(row = 5, column = 0)
boton_comenzar.grid(row = 6, column = 0)

## Hago un bucle infinito para el display de la ventana
root.mainloop()