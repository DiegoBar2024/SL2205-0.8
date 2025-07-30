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
from datetime import date

## ------------------------------------- DEFINICIÓN DE FUNCIONES ---------------------------------------

def calcularEdad(birthDate):
    today = date.today()
    age = today.year - birthDate.year - ((today.month, today.day) < (birthDate.month, birthDate.day))
    return age

def create_title(title, pdf):
    
    # Agrego el título principal del reporte
    pdf.set_font('Helvetica', 'b', 20)  
    pdf.ln(10)
    pdf.cell(0, 10, title, 0, 0, 'C')
    pdf.ln(10)

def write_to_pdf(pdf, words):
    
    # Seteo los parámetros del texto que voy a escribir en el PDF
    pdf.set_text_color(r = 0, g = 0, b = 0)
    pdf.set_font('Helvetica', '', 12)
    
    pdf.write(5, words)

def write_to_pdf_bold(pdf, words):
    
    # Seteo los parámetros del texto que voy a escribir en el PDF
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

def CreacionReporte(id_persona, nombre_persona, nacimiento_persona, tiempo, long_pasos, duraciones_pasos, velocidades, frecuencias, pasos_numerados, ruta_guardado):

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

    ## Genero una lista con el dia, el mes y el año de nacimiento EN ESE ORDEN
    ## MUY IMPORTANTE PARA QUE FUNCIONE EL CÁLCULO DE LA EDAD QUE LOS DATOS INGRESADOS TENGAN EL FORMATO: <<DD/MM/YYYY>>
    edad_param = nacimiento_persona.split("/")

    ## Hago el cálculo de la edad de la persona correspondiente
    edad = calcularEdad(date(int(edad_param[2]), int(edad_param[1]), int(edad_param[0])))

    ## Genero una tabla con la media y una desviación estándar aproximada para parámetros de marcha segun banda etaria
    ## Los valores numéricos se extraen del estudio "Gait analysis with the Kinect v2: normative study with healthy individuals and 
    ## comprehensive study of its sensitivity, validity, and reliability in individuals with stroke"
    ## Nomenclatura: LP (Longitud Paso[m]), DP (Duración Paso[s]), VEL (Velocidad [m/s]), CAD (Cadencia [pasos/s])
    parametros = {
        '10-19': {'LP' : [0.63, 0.05], 'DP': [0.54, 0.05], 'VEL': [1.16, 0.15], 'CAD': [111.87 / 60, 9.27 / 60]},
        '20-29': {'LP' : [0.67, 0.06], 'DP': [0.56, 0.05], 'VEL': [1.18, 0.15], 'CAD': [107.43 / 60, 9.60 / 60]},
        '30-39': {'LP' : [0.65, 0.05], 'DP': [0.56, 0.05], 'VEL': [1.16, 0.12], 'CAD': [107.87 / 60, 10.25 / 60]},
        '40-49': {'LP' : [0.64, 0.07] , 'DP': [0.54, 0.05], 'VEL': [1.19, 0.15], 'CAD': [112.37 / 60, 9.36 / 60]},
        '50-59': {'LP' : [0.62, 0.07], 'DP': [0.55, 0.06], 'VEL': [1.16, 0.19], 'CAD': [111.15 / 60, 12.03 / 60]},
        '60-69': {'LP' : [0.60, 0.08], 'DP': [0.56, 0.05], 'VEL': [1.06, 0.17], 'CAD': [107.56 / 60, 9.10 / 60]},
        '70-120': {'LP' : [0.52, 0.08], 'DP': [0.59, 0.07], 'VEL': [0.94, 0.19], 'CAD': [102.04 / 60, 9.99 / 60]}
    }

    ## Itero para cada una de las franjas etarias que tengo disponibles
    for franja in list(parametros.keys()):

        ## Hago la conversión de la cadena que tiene la franja etaria en una lista
        lista_franja = franja.split('-')

        ## En caso de que la edad calculada de la persona pertenezca a la franja etaria
        if int(lista_franja[0]) <= edad <= int(lista_franja[1]):

            ## Finalizo la ejecución del bucle ya que me quedo con la franja resultante que voy a usar para indexar
            break

    ## Tablas con los resultados del análisis de marcha
    TABLE_DATA = (
        ("Indicador", "Media", "Mediana", "Desviación estándar"),
        ("Longitud del paso", "{} m".format(round(np.mean(long_pasos), 2)), "{} m".format(round(np.median(long_pasos), 2)), "{} m".format(round(np.std(long_pasos), 2))),
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
        "Longitud de paso (m)": long_pasos,
    }
    COLUMNS = tuple(DATA.keys())

    # Create a new figure object
    plt.figure()
    df = pd.DataFrame(DATA, columns = COLUMNS)
    df.plot(x = COLUMNS[0], y = COLUMNS[1], kind = 'scatter', ylim = (min(long_pasos + [parametros[franja]['LP'][0] - 2 * parametros[franja]['LP'][1]]) * 0.8,
                                                                      max(long_pasos + [parametros[franja]['LP'][0] + 2 * parametros[franja]['LP'][1]]) * 1.2))
    plt.axhline(y = parametros[franja]['LP'][0] + 2 * parametros[franja]['LP'][1], color = 'red', linestyle = '--', linewidth = 2)
    plt.axhline(y = parametros[franja]['LP'][0] - 2 * parametros[franja]['LP'][1], color = 'red', linestyle = '--', linewidth = 2)

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
    df.plot(x = COLUMNS[0], y = COLUMNS[1], kind = 'scatter', ylim = (min(duraciones_pasos + [parametros[franja]['DP'][0] - 2 * parametros[franja]['DP'][1]]) * 0.8, 
                                                                      max(duraciones_pasos + [parametros[franja]['DP'][0] + 2 * parametros[franja]['DP'][1]]) * 1.2))
    plt.axhline(y = parametros[franja]['DP'][0] + 2 * parametros[franja]['DP'][1], color = 'red', linestyle = '--', linewidth = 2)
    plt.axhline(y = parametros[franja]['DP'][0] - 2 * parametros[franja]['DP'][1], color = 'red', linestyle = '--', linewidth = 2)

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
    df.plot(x = COLUMNS[0], y = COLUMNS[1], kind = 'scatter', ylim = (min(velocidades + [parametros[franja]['VEL'][0] - 2 * parametros[franja]['VEL'][1]]) * 0.8, 
                                                                      max(velocidades + [parametros[franja]['VEL'][0] + 2 * parametros[franja]['VEL'][1]]) * 1.2))
    plt.axhline(y = parametros[franja]['VEL'][0] + 2 * parametros[franja]['VEL'][1], color = 'red', linestyle = '--', linewidth = 2)
    plt.axhline(y = parametros[franja]['VEL'][0] - 2 * parametros[franja]['VEL'][1], color = 'red', linestyle = '--', linewidth = 2)

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
    df.plot(x = COLUMNS[0], y = COLUMNS[1], kind = 'scatter', ylim = (min(frecuencias + [parametros[franja]['CAD'][0] - 2 * parametros[franja]['CAD'][1]]) * 0.8, 
                                                                      max(frecuencias + [parametros[franja]['CAD'][0] + 2 * parametros[franja]['CAD'][1]]) * 1.2))
    plt.axhline(y = parametros[franja]['CAD'][0] + 2 * parametros[franja]['CAD'][1], color = 'red', linestyle = '--', linewidth = 2)
    plt.axhline(y = parametros[franja]['CAD'][0] - 2 * parametros[franja]['CAD'][1], color = 'red', linestyle = '--', linewidth = 2)

    # Converting Figure to an image:
    img_buf = BytesIO()  # Create image object
    plt.savefig(img_buf, dpi = 200)  # Save the image
    pdf.image(img_buf, w = pdf.epw / 2, x = 0.55 * pdf.epw, y = 0.65 * pdf.eph)

    # Generate the PDF
    pdf.output(ruta_guardado, 'F')
    img_buf.close()