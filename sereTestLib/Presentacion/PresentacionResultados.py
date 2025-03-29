## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
from pypdf import *
from PyPDF2 import *
from fpdf import FPDF, HTMLMixin
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Wavelets')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')

from LongitudPasoM1 import *

## -------------------------------------- CREACIÓN DEL DOCUMENTO ---------------------------------------

# Creo una clase heredada tanto de FPDF como de HTMLMixin
class MyFPDF(FPDF, HTMLMixin):
	pass

# Creo una instancia de la clase MyFPDF
pdf = MyFPDF()

# Agrego una página al PDF
pdf.add_page()

# Abro el archivo html
file = open("file.html", "r")

# Extraigo los datos de html como una cadena
Data = file.read()

# HTMLMixin write_html method
pdf.write_html(Data)

#saving the file as a pdf
pdf.output('Python_fpdf.pdf', 'F')