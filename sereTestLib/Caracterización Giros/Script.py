## Configuro nombre de la carpeta raíz
## (modificar si se utiliza en otro equipo)
root = "C:/Yo/Tesis/SL2205-0.8"

## Importación de librerías
import sys
sys.path.append("{}/SL2205-0.8/sereTestLib/Cinematica".format(root))
from LecturaDatosPacientes import *
from LecturaDatos import *
from Utils import *

## Programa principal
if __name__== '__main__':

    ## Opcion 1: Graficar histograma caracterizando la distribución de edades de la población
    opcion = 2

    ## En caso de que quiera caracterizar la población según la edad
    if opcion == 1:

        ## Obtengo la información correspondiente a todos los pacientes en la base de datos
        pacientes, ids_existentes = LecturaDatosPacientes()

        ## Obtengo el histograma por edad
        graficar_histograma_edades(pacientes)
    
    elif opcion == 2:

        ## Seteo el sistema inercial que voy a usar de referencia para el cálculo de orientación
        sist_inercial = 'ENU'

        ## Hago la lectura de las mediciones de la IMU del individuo
        data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = '299')

        ## Hago la estimación de la orientación del sistema de la IMU con respecto al sistema de referencia inercial
        imu_quat = estimar_orientacion_ekf(acel, gyro, 1 / periodoMuestreo, sist_inercial)

        ## Hago la rotación de la velocidad angular del sistema de la IMU al sistema inercial
        ang_vel_inercial = rotate_body_to_world(gyro, imu_quat)