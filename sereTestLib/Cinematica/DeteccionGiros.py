## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from scipy.signal import *
from LecturaDatos import *
from matplotlib import pyplot as plt
from scipy.signal import *
import pywt
from scipy.integrate import cumulative_trapezoid
from Fourier import *

## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

## Ruta donde voy a abrir el archivo
ruta_registro = "C:/Yo/Tesis/sereData/sereData/Registros/MarchaEstandar_Rodrigo.txt"

## Abro el fichero correspondiente
fichero = open(ruta_registro, "r")

## Hago la lectura de todas las lineas correspondientes al fichero
lineas = fichero.readlines()

## Creo un array vacío en donde voy a guardar los datos
data = []

## Itero para todas aquellas lineas que tengan información útil
for linea in lineas[3:]:

    ## Hago la traducción de la línea de datos a una lista de numeros flotantes, segmentando la línea por tabulación
    lista_datos = list(map(float,linea.split("\t")[:-1]))

    ## Agrego la lista de datos como renglón de la matriz de datos
    data.append(lista_datos)

## Hago una lista con todos los headers de los datos tomados
headers = lineas[1].split("\t")[:-1]

## Hago el pasaje de los datos en forma de matriz a forma de dataframe
data = pd.DataFrame(data, columns = headers)

## Creo una lista con las columnas deseadas
columnas_deseadas = ['Time', 'AC_x', 'AC_y', 'AC_z', 'GY_x', 'GY_y', 'GY_z']

## Creo un diccionario con los nombres originales de las columnas y sus nombres nuevos
nombres_columnas = {'Timestamp': 'Time', 'Accel_LN_X_CAL' : 'AC_x', 'Accel_LN_Y_CAL' : 'AC_y', 'Accel_LN_Z_CAL' : 'AC_z'
                    ,'Gyro_X_CAL' : 'GY_x', 'Gyro_Y_CAL' : 'GY_y', 'Gyro_Z_CAL' : 'GY_z'}

## Itero para cada una de las columnas del dataframe
for columna in data.columns:

    ## Itero para cada uno de los nombres posibles
    for nombre in nombres_columnas.keys():

        ## En caso de que un nombre esté en la columna
        if nombre in columna:

            ## Renombro la columna
            data = data.rename(columns = {columna : nombres_columnas[nombre]})

## Selecciono las columnas deseadas
data = data[columnas_deseadas]

## Armamos una matriz donde las columnas sean las aceleraciones
acel = np.array([np.array(data['AC_x']), np.array(data['AC_y']), np.array(data['AC_z'])]).transpose()

## Armamos una matriz donde las columnas sean los valores de los giros
gyro = np.array([np.array(data['GY_x']), np.array(data['GY_y']), np.array(data['GY_z'])]).transpose()

## Separo el vector de tiempos del dataframe
tiempo = np.array(data['Time'])

## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
tiempo = (tiempo - tiempo[0]) / 1000

## Obtengo el período de muestreo
periodoMuestreo = PeriodoMuestreo(data)

## Obtengo la cantidad total de muestras
cant_muestras = len(tiempo)

data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = 299)

## ---------------------------------------- DETECCIÓN DE GIROS -----------------------------------------

## Obtengo el valor del giroscopio en el eje x
gyro_x = gyro[:, 0]

## Obtengo el valor del giroscopio en el eje y
gyro_y = gyro[:, 1]

## Obtengo el valor dek giroscopio en el eje z
gyro_z = gyro[:, 2]

## Grafico los datos. En mi caso las tres velocidades angulares
plt.plot(tiempo, gyro_x, color = 'r', label = '$w_x$')
plt.plot(tiempo, gyro_y, color = 'b', label = '$w_y$')
plt.plot(tiempo, gyro_z, color = 'g', label = '$w_z$')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la velocidad angular (rad/s)
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad angular (rad/s)")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()

## Etapa de filtrado pasabajos de Butterworth con frecuencia de corte 0.5Hz de orden 4
sos = butter(N = 4, Wn = 0.5, btype = 'lowpass', fs = 1 / periodoMuestreo, output = 'sos')

## La cantidad de tiempo que transcurre entre dos valles debe ser igual al tiempo de paso
gyro_y_filtrada = sosfiltfilt(sos, gyro_y)

## Grafico los datos. En mi caso las tres velocidades angulares
plt.plot(tiempo, gyro_y_filtrada, color = 'b', label = '$w_y$')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la velocidad angular (rad/s)
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad angular (rad/s)")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Despliego la gráfica
plt.show()

## Defino la variable que contiene la posición de la muestra en la que estoy parado, la cual se inicializa en 0
ubicacion_muestra = 0

## Defino un tamaño de muestras por ventana
muestras_ventana = 400

## Defino un tamaño de muestras de solapamiento entre ventanas
muestras_solapamiento = 0

## Integro aceleración angular para obtener el ángulo de giro
angulos_y = cumulative_trapezoid(gyro_y, dx = periodoMuestreo, initial = 0)

## Pregunto si ya terminó de recorrer todo el vector de muestras
while ubicacion_muestra < cant_muestras:

    ## En caso que la última ventana no pueda tener el tamaño predefinido, la seteo manualmente
    if (cant_muestras - ubicacion_muestra < muestras_ventana):    

        ## Me quedo con el segmento del giroscopio GY correspondiente
        segmento_GY = angulos_y[ubicacion_muestra :]

    ## En otro caso, digo que la ventana tenga el tamaño predefinido
    else:

        ## Me quedo con el segmento del giroscopio GY correspondiente
        segmento_GY = angulos_y[ubicacion_muestra : ubicacion_muestra + muestras_ventana]

    ## Actualizo el valor de la ubicación de la muestra para que me guarde la posición en la que debe comenzar la siguiente ventana
    ubicacion_muestra += muestras_ventana - muestras_solapamiento

    print(np.max(np.abs(segmento_GY)) - np.min(np.abs(segmento_GY)))