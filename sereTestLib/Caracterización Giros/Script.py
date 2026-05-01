## Configuro nombre de la carpeta raíz
## (modificar si se utiliza en otro equipo)
root = "C:/Yo/Tesis"

## Importación de librerías
import sys
sys.path.append("{}/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica".format(root))
from LecturaDatosPacientes import *
from LecturaDatos import *
from Utils import *

## Programa principal
if __name__== '__main__':

    ## Opcion 1: Graficar histograma caracterizando la distribución de edades de la población
    ## Opcion 2: Detector de giros en un registro de marcha a partir de la velocidad angular
    opcion = 2

    ## En caso de que quiera caracterizar la población según la edad
    if opcion == 1:

        ## Obtengo la información correspondiente a todos los pacientes en la base de datos
        pacientes, ids_existentes = LecturaDatosPacientes()

        ## Obtengo el histograma por edad
        graficar_histograma_edades(pacientes)
    
    ## En caso que quiera hacer la detección de giros en el registro de marcha
    elif opcion == 2:

        ## Seteo el sistema inercial que voy a usar de referencia para el cálculo de orientación
        sist_inercial = 'ENU'

        ## Hago la lectura de las mediciones de la IMU del individuo
        ## Las medidas del Shimmer3 vienen en m/s2 para el acelerómetro y grados/s para el giroscopio
        data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = '299', 
        lectura_datos_propios = False, ruta = '{}/sereData/sereData/Registros/MarchaLibre_Rodrigo.txt'.format(root))

        ## Hago la conversión de los valores de velocidad angular de grados/s a rad/s
        gyro = gyro * np.pi / 180

        ## Defino la frecuencia de muestreo del sistema
        frec_muestreo = 1 / periodoMuestreo

        ## Hago la estimación de la orientación del sistema de la IMU con respecto al sistema de referencia inercial
        imu_quat = estimar_orientacion_ekf(acel, gyro, frec_muestreo, sist_inercial)

        ## Hago la rotación de la velocidad angular del sistema de la IMU al sistema inercial
        ang_vel_inercial = rotate_body_to_world(gyro, imu_quat)

        ## Hago el suavizado de la señal de velocidad angular en el eje vertical usando filtro de promedios
        ## móviles con el fin de remover picos no deseados de ruido en la señal
        wz_suav = moving_average(ang_vel_inercial[:,2], frec_muestreo)

        ## Hago la detección de los giros en base a la velocidad angular en el eje vertical
        giros = detect_turns_windowed(wz_suav, frec_muestreo)

        ## Hago la graficación de los tramos en los que se detectan giros de los que no
        plot_signal_with_events(ang_vel_inercial[:,2], giros)