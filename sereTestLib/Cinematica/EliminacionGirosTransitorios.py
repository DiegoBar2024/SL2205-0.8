## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from Segmentacion import *
from scipy.integrate import cumulative_trapezoid
import pathlib
sys.path.append(str(pathlib.Path().resolve()).replace('\\','/') + '/sereTestLib')
from parameters import *
import json

## ---------------------------------- ELIMINACIÓN DE GIROS Y TRANSITORIOS ----------------------------------

def EliminarGirosTransitorios(pasos, duraciones_pasos, giros):

    ## Creo una lista vacía donde guardo los pasos de la marcha en régimen
    pasos_regimen = []

    ## Creo una lista vacía donde guardo las duraciones de los pasos en régimen
    duraciones_pasos_reg = []

    ## Itero para cada uno de los pasos detectados eliminando transitorios
    ## Tomo tres pasos como el transitorio para poder llegar a régimen
    for i in range (3, len(pasos) - 2):

        ## En caso de que en dicho paso haya un giro
        if (pasos[i]['IC'][0] in np.array(giros)[:, 0]) or (pasos[i]['IC'][1] in np.array(giros)[:, 1]):

            ## Sigo iterando en el bucle
            continue

        ## En caso de que sea un paso en régimen, lo agrego a la lista de pasos en régimen
        pasos_regimen.append(pasos[i])

        ## En caso de que sea un paso en régimen, lo agrego a la lista de pasos en régimen
        duraciones_pasos_reg.append(duraciones_pasos[i])

    ## Retorno la lista de pasos en régimen
    return pasos_regimen, duraciones_pasos_reg