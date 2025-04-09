## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
from fpdf import FPDF
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')

from LongitudPasoM1 import *

## ------------------------------------- DEFINICIÓN DE FUNCIONES ---------------------------------------

def create_title(title, pdf):
    
    # Add main title
    pdf.set_font('Helvetica', 'b', 20)  
    pdf.ln(10)
    #pdf.write(5, title)
    pdf.cell(0, 10, title, 0, 0, 'C')
    pdf.ln(10)
    
    # # Add date of report
    # pdf.set_font('Helvetica', '', 14)
    # pdf.set_text_color(r=128,g=128,b=128)
    # today = time.strftime("%d/%m/%Y")
    # pdf.write(4, f'{today}')
    
    # # Add line break
    # pdf.ln(10)

def write_to_pdf(pdf, words):
    
    # Set text colour, font size, and font type
    pdf.set_text_color(r=0,g=0,b=0)
    pdf.set_font('Helvetica', '', 12)
    
    pdf.write(5, words)

def write_to_pdf_bold(pdf, words):
    
    # Set text colour, font size, and font type
    pdf.set_text_color(r=0,g=0,b=0)
    pdf.set_font('Helvetica', 'B', 12)
    
    pdf.write(5, words)

class PDF(FPDF):

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

## ------------------------------------ CREACIÓN DEL DOCUMENTO -----------------------------------------

TITLE = "Reporte"
WIDTH = 210
HEIGHT = 297

# Creación de un PDF por defecto se crea con un tamaño A4 (210 by 297 mm)
pdf = PDF()

# Agrego una página
pdf.add_page()

# Creo un título para el PDF
create_title(TITLE, pdf)

# Datos del paciente
write_to_pdf_bold(pdf, "ID Paciente: ")
write_to_pdf(pdf, "{}".format(id_persona))
pdf.ln(6)
write_to_pdf_bold(pdf, "Nombre Paciente: ")
write_to_pdf(pdf, nombre_persona)
pdf.ln(6)
write_to_pdf_bold(pdf, "Fecha de Nacimiento: ")
write_to_pdf(pdf, nacimiento_persona)
pdf.ln(6)
write_to_pdf_bold(pdf, "Fecha de la sesión: ")
write_to_pdf(pdf, "{}".format(time.strftime("%d/%m/%Y")))
pdf.ln(6)
write_to_pdf_bold(pdf, "Duración de la sesión: ")
write_to_pdf(pdf, "{}m {}s".format(int(tiempo[-1] // 60), int(tiempo[-1]) - 60 * int(tiempo[-1] // 60)))
pdf.ln(10)

## Tablas con los resultados del análisis de marcha
TABLE_DATA = (
    ("Indicador", "Media", "Mediana", "Desviación estándar"),
    ("Longitud del paso", "{} m".format(round(np.mean(long_pasos_m1), 2)), "{} m".format(round(np.median(long_pasos_m1), 2)), "{} m".format(round(np.std(long_pasos_m1), 2))),
    ("Duración del paso", "{} s".format(round(np.mean(duraciones_pasos), 2)),"{} s".format(round(np.median(duraciones_pasos), 2)), "{} m".format(round(np.std(duraciones_pasos), 2))),
    ("Velocidad de marcha", "{} m/s".format(round(np.mean(velocidades), 2)), "{} m/s".format(round(np.median(velocidades), 2)), "{} m/s".format(round(np.std(velocidades), 2))),
    ("Cadencia", "{} pasos/s".format(round(np.mean(frecuencias), 2)), "{} pasos/s".format(round(np.median(frecuencias), 2)), "{} pasos/s".format(round(np.std(frecuencias), 2)))
)

with pdf.table(text_align = "CENTER") as table:
    for data_row in TABLE_DATA:
        row = table.row()
        for datum in data_row:
            row.cell(datum)

pdf.ln(10)

## Impresión de diagramas
## Gráfico de la longitud de los pasos en función de cada paso
DATA = {
    "Número de paso": pasos_numerados,
    "Longitud de paso (m)": long_pasos_m1,
}
COLUMNS = tuple(DATA.keys())

# Create a new figure object
plt.figure()
df = pd.DataFrame(DATA, columns = COLUMNS)
df.plot(x = COLUMNS[0], y = COLUMNS[1], kind = 'scatter')

# Converting Figure to an image:
img_buf = BytesIO()  # Create image object
plt.savefig(img_buf, dpi = 200)  # Save the image
pdf.image(img_buf, w = pdf.epw / 2, x = 0.05 * pdf.epw, y = 0.40 * pdf.eph)

## Impresión de diagramas
## Gráfico de la duración de los pasos en función de cada paso
DATA = {
    "Número de paso": pasos_numerados,
    "Duración del paso (s)": duraciones_pasos,
}
COLUMNS = tuple(DATA.keys())

# Create a new figure object
plt.figure()
df = pd.DataFrame(DATA, columns = COLUMNS)
df.plot(x = COLUMNS[0], y = COLUMNS[1], kind = 'scatter')

# Converting Figure to an image:
img_buf = BytesIO()  # Create image object
plt.savefig(img_buf, dpi = 200)  # Save the image
pdf.image(img_buf, w = pdf.epw / 2, x = 0.55 * pdf.epw, y = 0.40 * pdf.eph)

## Impresión de diagramas
## Gráfico de la duración de los pasos en función de cada paso
DATA = {
    "Número de paso": pasos_numerados,
    "Velocidad instantánea (m/s)": velocidades,
}
COLUMNS = tuple(DATA.keys())

# Create a new figure object
plt.figure()
df = pd.DataFrame(DATA, columns = COLUMNS)
df.plot(x = COLUMNS[0], y = COLUMNS[1], kind = 'scatter')

# Converting Figure to an image:
img_buf = BytesIO()  # Create image object
plt.savefig(img_buf, dpi = 200)  # Save the image
pdf.image(img_buf, w = pdf.epw / 2, x = 0.05 * pdf.epw, y = 0.65 * pdf.eph)

## Impresión de diagramas
## Gráfico de la duración de los pasos en función de cada paso
DATA = {
    "Número de paso": pasos_numerados,
    "Cadencia instantánea (pasos/s)": frecuencias,
}
COLUMNS = tuple(DATA.keys())

# Create a new figure object
plt.figure()
df = pd.DataFrame(DATA, columns = COLUMNS)
df.plot(x = COLUMNS[0], y = COLUMNS[1], kind = 'scatter')

# Converting Figure to an image:
img_buf = BytesIO()  # Create image object
plt.savefig(img_buf, dpi = 200)  # Save the image
pdf.image(img_buf, w = pdf.epw / 2, x = 0.55 * pdf.epw, y = 0.65 * pdf.eph)

# Generate the PDF
pdf.output("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Presentacion/Reporte.pdf", 'F')
img_buf.close()