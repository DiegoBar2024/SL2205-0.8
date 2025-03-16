## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
from pypdf import *
from PyPDF2 import *
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Wavelets')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')

from LongitudPasoM1 import *

## -------------------------------------- CREACIÓN DEL DOCUMENTO ---------------------------------------

## Especifico la ruta del archivo en donde voy a guardar los resultados
ruta_fichero = "C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Presentacion/Archivos"

## Especifico la terminación del fichero según el paciente en particular usando su ID (la S es de muestra corta)
nombre_fichero = 'S{}'.format(id_paciente)