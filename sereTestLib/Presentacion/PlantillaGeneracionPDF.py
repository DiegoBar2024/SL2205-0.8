import fpdf
from fpdf import FPDF
import time
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi

def create_title(title, pdf):
    
    # Add main title
    pdf.set_font('Helvetica', 'b', 20)  
    pdf.ln(40)
    pdf.write(5, title)
    pdf.ln(10)
    
    # Add date of report
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(r=128,g=128,b=128)
    today = time.strftime("%d/%m/%Y")
    pdf.write(4, f'{today}')
    
    # Add line break
    pdf.ln(10)

def write_to_pdf(pdf, words):
    
    # Set text colour, font size, and font type
    pdf.set_text_color(r=0,g=0,b=0)
    pdf.set_font('Helvetica', '', 12)
    
    pdf.write(5, words)

class PDF(FPDF):

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

# Global Variables
TITLE = "Monthly Business Report"
WIDTH = 210
HEIGHT = 297

# Create PDF
pdf = PDF() # A4 (210 by 297 mm)

'''
First Page of PDF
'''
# Add Page
pdf.add_page()

# Add lettterhead and title
create_title(TITLE, pdf)

# Add some words to PDF
write_to_pdf(pdf, "1. The table below illustrates the annual sales of Heicoders Academy:{} ".format(1))
pdf.ln(15)

# Add some words to PDF
write_to_pdf(pdf, "2. The visualisations below shows the trend of total sales for Heicoders Academy and the breakdown of revenue for year 2016:")

# Generate the PDF
pdf.output("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Presentacion/ReporteEjemplo.pdf", 'F')