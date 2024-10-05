from sereTestLib.parameters import static_path , modo_webservice
from pathlib import Path


##############################################
#BASE DE DATOS
server='localhost'
user='sere'
password='Maggiolo_590_sere'
database='sere'
port=1433
###############################################
#PATHS
path_muestras=static_path + "Muestras/"
zip_path = path_muestras +"zips/"
to_unzip_path= path_muestras+"unzip/"
calibs= path_muestras+"calibs/"
home = str(Path.home())
cal_path=static_path+ "Calibraciones/"
ruta_pub_key= home+ "/seresensys.pub"

###############################################
#MAIL
mail_sere= "serelabs.uy@gmail.com"
passw = "hxuqxugnzmwwmqqo"
attachment_path= static_path+"/attachments/"
###############################################
#PDF
from fpdf import FPDF 
from fpdf.enums import XPos, YPos
class PDF(FPDF):
    def header(self):
        # Logo
        self.image(attachment_path+'logo.png', 10, 8, 33)

###############################################
path_reportes=static_path+"/reportes/"

secret_password = b'Maggiolo_590'

################################################
#                   WEBSERVICE
################################################
if modo_webservice=="modo_testing":
   server='sql-server-sere-SL2205-testing'
elif modo_webservice=="modo_prod":
       server='sql-server-sere-SL2205-prod'



