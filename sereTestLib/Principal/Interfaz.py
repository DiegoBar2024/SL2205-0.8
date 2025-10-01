## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
import pandas as pd
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Presentacion')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
from parameters import *
import os
from joblib import load
import numpy as np
from LecturaDatos import *
from ContactosIniciales import *
from ContactosTerminales import *
from Segmentacion import *
from LongitudPasoM1 import *
from GeneracionReporte import *
from DeteccionActividades import DeteccionActividades
from EliminacionGirosTransitorios import *
from tkinter import messagebox
import tkinter as tk
import tkinter.filedialog

## ------------------------------------- GENERACIÓN DE INTERFAZ --------------------------------------

## Función que al cerrar la ventana termina la ejecución del programa
def Terminar():
    root.quit()
    root.destroy()

## Función que me permite pedir el directorio
def PedirDirectorio():
    global ruta_registro
    ruta_registro = tkinter.filedialog.askopenfile().name

def RealizarAnalisis():

    ## Genero un bloque try donde voy a guardar toda la interfaz
    try:

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

        ## Especifico la cantidad de muestras que va a tener la ventana de analisis
        muestras_ventana = 100

        ## Especifico la cantidad de muestras de solapamiento entre ventanas
        muestras_solapamiento = 50

        ## Hago el cálculo del vector de SMA para dicha persona
        vector_SMA, features, ventanas, nombres_features = DeteccionActividades(acel, tiempo, muestras_ventana, muestras_solapamiento, periodoMuestreo, cant_muestras, actividad = None, CalcFeatures = False)

        ## Cargo el modelo del clasificador ya entrenado según la ruta del clasificador
        clf_entrenado = load(ruta_SVM)

        ## Determino la predicción del clasificador ante mi muestra de entrada
        ## Etiqueta 0: Reposo
        ## Etiqueta 1: Movimiento
        pat_predictions = clf_entrenado.predict(np.array((vector_SMA)).reshape(-1, 1))

        ## Me quedo sólo con las ventanas en las cuales se ha detectado marcha
        ventanas_marcha = ventanas[np.where(pat_predictions == 1)]

        ## Tomo la hipótesis de que el reposo puede estar al inicio y al final únicamente
        ## Selecciono entonces el tramo de aceleración en la cual se ha detectado marcha
        acel_marcha = acel[ventanas_marcha[0][0] : ventanas_marcha[-1][1], :]

        ## Selecciono entonces el tramo de giroscopios en la cual se ha detectado marcha
        gyro_marcha = gyro[ventanas_marcha[0][0] : ventanas_marcha[-1][1], :]

        ## Obtengo la cantidad total de muestras únicamente del tramo de marcha
        cant_muestras_marcha = acel_marcha.shape[0]

        ## Obtengo el vector de tiempos correspondiente durante el tramo de marcha
        tiempo_marcha = np.arange(start = 0, stop = cant_muestras_marcha * periodoMuestreo, step = periodoMuestreo)

        ## Cálculo de contactos iniciales
        contactos_iniciales, muestras_paso, acc_AP_norm, frec_fund = ContactosIniciales(acel_marcha, cant_muestras_marcha, periodoMuestreo, graficar = False)

        ## Cálculo de contactos terminales
        contactos_terminales = ContactosTerminales(acel_marcha, cant_muestras_marcha, periodoMuestreo, graficar = False)

        ## Hago la segmentación de la marcha
        pasos, duraciones_pasos, giros = Segmentacion(contactos_iniciales, contactos_terminales, muestras_paso, periodoMuestreo, acc_AP_norm, gyro_marcha)

        ## Elimino giros y transitorios donde paso como parámetro la cantidad de pasos que tiene mi transitorio
        pasos, duraciones_pasos = EliminarGirosTransitorios(pasos, duraciones_pasos, giros, cant_pasos_transitorio = 3)

        ## Cálculo de parámetros de marcha usando el método I
        pasos_numerados, frecuencias, velocidades, long_pasos_m1, coeficientes_m1 = LongitudPasoM1(pasos, acel_marcha, tiempo_marcha, periodoMuestreo, frec_fund, duraciones_pasos, id_persona, giros, longitud_pierna)

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

        ## Despliego un mensaje comunicando que el análisis se terminó de realizar correctamente
        messagebox.showinfo('Info', 'El análisis se ha realizado correctamente. Resultados disponibles en la carpeta correspondiente.')

    ## En caso de que ocurra algún error al ingresar los datos
    except ValueError:

        ## Despliego una ventana que me diga el error correspondiente
        messagebox.showerror("Error", "Error al ingresar los datos. Asegurarse que el formato sea correcto.")
    
    ## En caso de que no se haya seleccionado ningún archivo
    except PermissionError:

        ## Despliego una ventana que me diga el error correspondiente
        messagebox.showerror("Error", "No se ha seleccionado ningún registro para el análisis.")
    
    ## En caso de que ocurra alguna excepción
    except Exception as e:

        ## Despliego una ventana que me muestre el mensaje de error correspondiente
        messagebox.showerror("Error", "Ocurrió un error inesperado: \n {}.".format(e))

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

## Variable de etiqueta para la ID del paciente
etiqueta_ID = tk.Label(root, text = 'ID del paciente: ', font = ('calibre', 10, 'bold'))

## Variable de entrada para la ID del paciente
entrada_ID = tk.Entry(root,textvariable = ID_var, font = ('calibre', 10, 'normal'))

## Variable de etiqueta para el nombre del paciente
etiqueta_nombre = tk.Label(root, text = 'Nombre del paciente (<<Nombre>> <<Apellido>>): ', font = ('calibre', 10, 'bold'))

## Variable de entrada para el nombre del paciente
entrada_nombre = tk.Entry(root,textvariable = nombre_var, font = ('calibre', 10, 'normal'))

## Variable de etiqueta para la fecha de nacimiento del paciente
etiqueta_nacimiento = tk.Label(root, text = 'Fecha de nacimiento del paciente: (<<DD/MM/YYYY>>): ', font = ('calibre', 10, 'bold'))

## Variable de entrada para la fecha de nacimiento del paciente
entrada_nacimiento = tk.Entry(root,textvariable = nacimiento_var, font = ('calibre', 10,'normal'))

## Variable de etiqueta para la longitud de la pierna del paciente
etiqueta_pierna = tk.Label(root, text = 'Longitud de la pierna (m): ', font = ('calibre', 10, 'bold'))

## Variable de entrada para la longitud de la pierna del paciente
entrada_pierna = tk.Entry(root,textvariable = pierna_var, font = ('calibre', 10, 'normal'))

## Variable de etiqueta para la longitud del pie del paciente
etiqueta_pie = tk.Label(root, text = 'Longitud del pie (cm): ', font = ('calibre', 10, 'bold'))

## Variable de entrada para la longitud del pie del paciente
entrada_pie = tk.Entry(root,textvariable = pie_var, font = ('calibre', 10, 'normal'))

## Botón que permite al usuario elegir el directorio en el cual se encuentra el registro de marcha a analizar
boton_directorio = tk.Button(root, text = 'Seleccionar Archivo', command = PedirDirectorio)

## Botón que permite al usuario comenzar a realizar el análisis
boton_comenzar = tk.Button(root,text = 'Comenzar Análisis', command = RealizarAnalisis)

## Configuro un botón para terminar la ejecución
terminar = tk.Button(root, text = "Cerrar", command = Terminar)

## Configuro grilla de botones y de entradas en la interfaz gráfica
etiqueta_ID.grid(row = 0, column = 0)
entrada_ID.grid(row = 0, column = 1)
etiqueta_nombre.grid(row = 1, column = 0)
entrada_nombre.grid(row = 1, column = 1)
etiqueta_nacimiento.grid(row = 2, column = 0)
entrada_nacimiento.grid(row = 2, column = 1)
etiqueta_pierna.grid(row = 3, column = 0)
entrada_pierna.grid(row = 3, column = 1)
etiqueta_pie.grid(row = 4, column = 0)
entrada_pie.grid(row = 4, column = 1)
boton_directorio.grid(row = 5, column = 0)
boton_comenzar.grid(row = 5, column = 1)
terminar.grid(row = 6, column = 0)

## Hago un bucle infinito para el display de la ventana
root.mainloop()