## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from scipy.signal import *
from LecturaDatos import *
from matplotlib import pyplot as plt
from scipy.signal import *
from Fourier import *
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/KielMAT-main/kielmat/modules/ptd')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/KielMAT-main/kielmat')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
from parameters import *
from _pham import *

## ------------------------------------- DETECCION TRANSICIONES --------------------------------------

## Construyo una función que me permita detectar transiciones posturales
def TransicionesPosturales(acel, gyro, periodoMuestreo):

    ## Creo un objeto el cual me permita determinar transiciones posturales
    pham = PhamPosturalTransitionDetection()

    ## Hago la detección de transiciones posturales usando el método correspondiente
    pham.detect(accel_data = pd.DataFrame(acel), gyro_data = pd.DataFrame(gyro), sampling_freq_Hz = 200,
                tracking_system = "imu", tracked_point = "LowerBack", plot_results = True)
    
    ## Retorno las transiciones posturales
    return pham
    
## Ejecución principal del programa
if __name__== '__main__':

    ## Especifico la ruta en la cual se encuentra el registro a leer
    ruta_registro_completa = ruta_registro + 'MarchaEstandar_Rodrigo.txt'

    ## Hago la lectura de los datos
    data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = 69, lectura_datos_propios = True, ruta = ruta_registro_completa)

    ## Hago la detección de eventos de transición
    transiciones = TransicionesPosturales(acel, gyro, periodoMuestreo)