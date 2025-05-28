## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import numpy as np
from matplotlib import pyplot as plt
from LecturaDatos import *
from ContactosIniciales import *
from ContactosTerminales import *
from Segmentacion import *
from LongitudPasoM1 import *
from LongitudPasoM2 import *
import json

## ------------------------------------- ÓPTIMIZACION DEL MODELO ---------------------------------------

def Optimizacion():

    ## Especifico el ID de la persona para la cual voy a hacer el procesamiento de los datos
    id_persona = 2

    ## Hago la lectura de los datos especificando que estoy leyendo datos propios
    data, acel, gyro, cant_muestras, periodoMuestreo, nombre_persona, nacimiento_persona, tiempo = LecturaDatos(lectura_datos_propios = True)

    ## Cálculo de contactos iniciales
    contactos_iniciales, muestras_paso, acc_AP_norm, frec_fund = ContactosIniciales(acel, cant_muestras, periodoMuestreo, graficar = True)

    ## Cálculo de contactos terminales
    contactos_terminales = ContactosTerminales(acel, cant_muestras, periodoMuestreo, graficar = True)

    ## Hago la segmentación de la marcha
    pasos, duraciones_pasos = Segmentacion(contactos_iniciales, contactos_terminales, muestras_paso, periodoMuestreo, acc_AP_norm, gyro)

    ## Cálculo de parámetros de marcha para el método I
    pasos_numerados_m1, frecuencias_m1, velocidades_m1, long_pasos_m1, coeficientes_m1 = LongitudPasoM1(pasos, acel, tiempo, periodoMuestreo, frec_fund, duraciones_pasos)

    ## Cálculo de parámetros de marcha para el método I
    pasos_numerados_m2, frecuencias_m2, velocidades_m2, long_pasos_m2, coeficientes_m2 = LongitudPasoM2(pasos, acel, tiempo, periodoMuestreo, frec_fund, duraciones_pasos)

    ## Especifico la longitud del pie de la persona
    long_pie = 0.28

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

    ## Retorno los valores óptimos de los coeficientes correspondientes a cada método
    return optimo_m1, optimo_m2, id_persona

## Rutina principal del programa
if __name__== '__main__':

    ## Hago la lectura del archivo JSON previamente existente
    with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/Optimizacion.json", 'r') as openfile:

        # Cargo el valor existente del JSON
        dicc_optim = json.load(openfile)
    
    ## Obtengo las constantes de optimización asociadas a las pruebas realizadas para ambos métodos
    optimo_m1, optimo_m2, id_persona = Optimizacion()

    ## La clave en principio va a ser el ID de la persona. En principio se toma un registro controlado por persona
    dicc_optim[str(id_persona)] = (optimo_m1, optimo_m2)
    
    ## Especifico la ruta del archivo JSON sobre la cual voy a reescribir
    with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/Optimizacion.json", "w") as outfile:

        ## Escribo el diccionario actualizado
        json.dump(dicc_optim, outfile)