from distutils.log import info
import os
from random import sample
import numpy as np
from hashlib import md5
from fpdf.enums import XPos, YPos
import plotly.graph_objects as go
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders
import pyminizip as pyzip
from pyzip import PyZip
from pyfolder import PyFolder
import plotly.express as px
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta 
import shutil

from SereSensysLib.procesamiento.procesamiento_SereSensys import DataSereSensys
from SereSensysLib.procesamiento.parameters_procesamiento import ruta_priv_key

from sereTestLib.utils.acceleration import average_ac3
from sereTestLib.parameters import dir_in_original_sample,date
from sereTestLib.webservice.parameters_ws import attachment_path, PDF, path_reportes, secret_password, mail_sere, passw, home, ruta_pub_key
from sereTestLib.Inferencia import Inferencia
from sereTestLib.webservice.consultas import insertar_resultado, datos_muestras_pac

def agregar_dato_dict(dict1, dato, dict2):
    """
    Si la key "dato" existe en el dict1, agrega su contenido en el dict 2, de lo contrario agrega un campo vacio con la key "dato".
    
    Args:
        dict1 (dict): diccionario con el dato que queremos copiar
        dato (str): key que queremos copiar del dict1
        dict2 (dict): diccionario al que le queremos agregar un dato

    Returns:
        dict2 (dict): diccionario con el dato agregado.

    """
    if dato in dict1:
        dict2[dato]=dict1[dato]
    else:
        dict2[dato]=""
    return(dict2)


def success_to_save(fpath):
    """
    Returns True if exists a file with name fpath
    """
    return(os.path.isfile(fpath))

def is_non_zero_file(fpath): 
    """
    Returns True if the size of the file is greater than 0
    """ 
    return (os.path.getsize(fpath) > 0)

def verificar_checksum(checksum, path):
    """
    Checks if checksum is equal to the md5 of the zip file

    Args:
        checksum (str): the checksum
        path (str): the path to the zip file

    Returns:
        (bool): True if its equal
    """
    m = md5()
    with open(path, "rb") as f:
        data = f.read() 
        m.update(data)
        
    return (m.hexdigest()==checksum)

def comprobar_AC_y(sample_id, lb,ub):
    """
    Returns True if the mean AC Y of the sample_id (avg) is lb<avg<ub
    """
    avg=average_ac3(dir_in_original_sample,sample_id)[0]
    return np.abs(avg)>lb and np.abs(avg)<ub

def obtener_sample_id(filename):
    """
    Returns the name of the file without the extension.
    It expects that the filename is the id of the sample
    """
    return(os.path.splitext(filename)[0])

def sendEmail(smtpHost, smtpPort, mailUname, mailPwd, fromEmail, mailSubject, mailContentHtml, recepientsMailList, attachmentFpaths):
    """
    Function that sends a mail.

    Args:
        smtpHost (str): Host of stmp server
        smtpPort (int): Port of the stmp server
        mailUname (str): Account of the sender's email
        mailPwd (str): Password of the sender's email
        fromEmail (str): Sender's email
        mailSubject (str): Subject
        mailContentHtml (str): Message
        recepientsMailList (list): list of the recipients' emails
        attachmentFpaths (list): path of the attachment files
    """
    # create message object
    msg = MIMEMultipart()
    msg['From'] = fromEmail
    msg['To'] = ','.join(recepientsMailList)
    msg['Subject'] = mailSubject
    msg.attach(MIMEText(mailContentHtml, 'html'))

 # create file attachments
    for aPath in attachmentFpaths:
        # check if file exists
        part = MIMEBase('application', "octet-stream")
        part.set_payload(open(aPath, "rb").read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment; filename="{0}"'.format(os.path.basename(aPath)))
        msg.attach(part)
    # Send message object as email using smptplib
    s = smtplib.SMTP(smtpHost, smtpPort)
    s.starttls()
    s.login(mailUname, mailPwd)
    msgText = msg.as_string()
    sendErrs = s.sendmail(fromEmail, recepientsMailList, msgText)
    s.quit()

    # check if errors occured and handle them accordingly
    if not len(sendErrs.keys()) == 0:
        raise Exception("Errors occurred while sending email", sendErrs)

def mandar_mail(send_to, path_pdf,sample_id=''):
    """ 
    Calls sendEmail to mail the results report.

    Args:
        send_to (str): recipient
        path_pdf (str): path to the results report
    """
    # mail server parameters
    smtpHost = "smtp.gmail.com"
    smtpPort = 587
    mailUname = mail_sere
    mailPwd = passw
    fromEmail = mail_sere

    # mail body, recepients, attachment files
    attachmentFpaths = [path_pdf]
    mailSubject = "Reporte Estabilidad SereTest "+sample_id
    mailContentHtml = ""
    recepientsMailList = [send_to]
    sendEmail(smtpHost, smtpPort, mailUname, mailPwd, fromEmail,
          mailSubject, mailContentHtml, recepientsMailList,attachmentFpaths)

def generar_plot_estabilidad(estabilidad,sample_id):
    """
    Generates the stability graph and saves it.

    Args:
        estabilidad (float): stability percentage
        sample_id (str): Sample id
    """
    plot_bgcolor = "white"
    quadrant_colors = [plot_bgcolor,"#2bad4e" , "#85e043", "#eff229", "#f2a529", "#f25829" ] 
    quadrant_text = ["", "<b>Muy Estable</b>", "<b>Estable</b>", "<b>Moderado</b>", "<b>Instable</b>", "<b>Muy Inestable</b>"]
    n_quadrants = len(quadrant_colors) - 1

    current_value = float(f'{estabilidad:.2f}')
    min_value = 0
    max_value = 100
    hand_length = np.sqrt(2) / 4
    hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))
    fig = go.Figure(
    data=[
        go.Pie(
            values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
            rotation=90,
            hole=0.5,
            marker_colors=quadrant_colors,
            text=quadrant_text,
            textinfo="text",
            hoverinfo="skip",
        ),
    ],
    layout=go.Layout(
        showlegend=False,
        margin=dict(b=0,t=10,l=10,r=10),
        width=450,
        height=450,
        paper_bgcolor=plot_bgcolor,
        annotations=[
            go.layout.Annotation(
                text=f"<b>Índice Sere:</b><br>{current_value}%",
                x=0.5, xanchor="center", xref="paper",
                y=0.25, yanchor="bottom", yref="paper",
                showarrow=False,
            )
        ],
        shapes=[
            go.layout.Shape(
                type="circle",
                x0=0.48, x1=0.52,
                y0=0.48, y1=0.52,
                fillcolor="#333",
                line_color="#333",
            ),
            go.layout.Shape(
                type="line",
                x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                line=dict(color="#333", width=4)
            )
        ]
    )
    )
    #fig.show()
    fig.write_image(attachment_path+str(sample_id)+'_plot_estabilidad.png','png')

def generar_datos(result,json_patient,json_sensor,cal_id=None, anonimo=None):
    """
    Given the inference results, creates a dictionary with the neccesary information for the report and calls "generar_plot_estabilidad"
    Args:
        result (results): inference results
        json_patient (json): sample patient information
        json_sensor (json): sample sensor information
        cal_id (str): calibration id
        anonimo (bool): if True, some data is anonimyzed. 
    """
    if json_sensor['buena_sentado'] != '-1':
        buena_sentado = int(json_sensor['buena_sentado'])
        
        inicio_sentado_min, inicio_sentado_seg, _ = json_sensor['inicio_sentado'][buena_sentado].split(':')
        final_sentado_min, final_sentado_seg, _ = json_sensor['final_sentado'][buena_sentado].split(':')
        inicio_sentado = timedelta(hours=0, minutes=int(inicio_sentado_min), seconds=int(inicio_sentado_seg))
        final_sentado = timedelta(hours=0, minutes=int(final_sentado_min), seconds=int(final_sentado_seg))
        
        tiempo_sentado = final_sentado - inicio_sentado
    else:
        tiempo_sentado = timedelta(hours=0, minutes=0, seconds=0)
    
    if json_sensor['buena_parado'] != '-1':
        buena_parado = int(json_sensor['buena_parado'])
        
        inicio_parado_min, inicio_parado_seg, _ = json_sensor['inicio_parado'][buena_parado].split(':')
        final_parado_min, final_parado_seg, _ = json_sensor['final_parado'][buena_parado].split(':')
        inicio_parado = timedelta(hours=0, minutes=int(inicio_parado_min), seconds=int(inicio_parado_seg))
        final_parado = timedelta(hours=0, minutes=int(final_parado_min), seconds=int(final_parado_seg))
        
        tiempo_parado = final_parado - inicio_parado
    else:
        tiempo_parado = timedelta(hours=0, minutes=0, seconds=0)
        
    tiempo_inactivo = str(tiempo_sentado + tiempo_parado)
    
    dict={
        "sample_id":json_patient["numero_muestra"],
        "fecha":json_patient["fecha"],
        "paciente_id":json_patient["cedula"],
        "paciente":json_patient["nombre"]+" "+json_patient["apellido"],
        "medico":json_patient["identificador_medico_tratante"],
        "estabilidad":result.stable_percentage["".join("Caminando")],
        "tiempo_activo":result.activities_time_format["Caminando"],
        "tiempo_inactivo":tiempo_inactivo,
        "modelo autoencoder": result.ae_name,
        "modelo clasificador": result.clf_name,
        "commit":result.git_version_commit,
        "institucion":json_patient["institucion"],
        "lugar":json_patient["lugar"],
        "pasa_umbral": result.activities_time_secs["Caminando"]>70,
        "observaciones":json_patient['anotaciones_individuales_sobre_la_prueba'],
        "intervencion":json_patient['intervencion_terapeutica'],
        "cal_sensor":cal_id,
        "sensor": json_patient["sensor_utilizado"],
        "talla":json_patient["talla"],
        "peso":json_patient["peso"],
        "sexo":json_patient["sexo"],
        "sicofarmacos":json_patient["sicofarmacos"],
        "caida":json_patient["caida"],
        "inestabilidad":json_patient["inestabilidad"],
        "auxiliar_mov":json_patient["auxiliar_de_movilidad"],
        "vertigos": json_patient["vertigos"],
        "patologias_previas":json_patient["patologias_previas_conocidas"],
        "clasificacion_tecnico":json_patient["clasificacion_del_tecnico"],
        "calzado":json_patient["calzado"],
        "uso_auxiliar_mov":json_patient["uso_aux_mov"],
        "nombre_farmaco": json_patient["nombre_farmaco"],
        "fecha_nac":json_patient["fecha_nac"],
        "hora_del_dia":json_patient["hora_del_dia"],

        # "sample_id":"101",
        # "fecha":"2022-09-20",
        # "hora_del_dia": "10:50",
        # "paciente_id":"4564654",
        # "paciente":"Juan Perez",
        # "medico": "7894565",
        # "estabilidad": 40,
        # "tiempo_activo": "0:01:10",
        # "tiempo_inactivo":"0:00:15",
        # "modelo autoencoder": "dsadsadsfa",
        # "modelo clasificador": "dsaffas",
        # "commit":"commit",
        # "institucion":"SereLabs",
        # "lugar": "Oficina",
        # "pasa_umbral":True,
        # "observaciones": "fsadkhjfkadsfj dsakj sdfka jhadsfkjhdf kjhsfdak djhsfkjsfkjdsaf asjdhfkasdhfj skhjdskajdsaj haskhdsajksdakh shshshhshdsa jh daskj dhsakjdhsa skjdhs aksdjha sdkjhas ",
        # "intervencion": "Audifonos",
        # "sensor": 1  ,
        # "cal_sensor":cal_id,
        
        # "talla":"1.85",
        # "peso":"87",
        # "sexo":"Fem",
        # "sicofarmacos":"Si",
        # "caida":"No",
        # "inestabilidad":"No",
        # "auxiliar_mov":"No",
        # "vertigos": "No",
        # "patologias_previas":"Ninguna",
        # "clasificacion_tecnico":"Estable",
        # "calzado":"Championes",
        # "uso_auxiliar_mov":"No",
        # "nombre_farmaco":" - ",
        # "fecha_nac":"1944-02-05",
        # "caida_ultimo_testeo":"",
        # "piso":"",
        # "interior_exterior":"",
        # "distancia": "",
        # "escala_downton": "",


    }
    dict=agregar_dato_dict(json_patient, "caida_ultimo_testeo", dict)
    dict=agregar_dato_dict(json_patient, "piso", dict)    
    dict=agregar_dato_dict(json_patient, "interior_exterior", dict)
    dict=agregar_dato_dict(json_patient, "distancia", dict)
    dict=agregar_dato_dict(json_patient, "escala_downton", dict)

    if anonimo:
        dict["paciente_id"]="1234657"
        dict["fecha"]="2022-07-29"
        dict["paciente"]="Juan Perez"
        dict["medico"]="7894565"
        dict["institucion"]="SereLabs"
        dict["lugar"]="PB"
        dict["fecha_nac"]="2000-01-01"
    fecha_hoy = datetime.strptime(dict["fecha"],'%Y-%m-%d').date()
    fecha_nac=datetime.strptime(dict["fecha_nac"], '%Y-%m-%d').date()
    difference = relativedelta(fecha_hoy, fecha_nac)
    dict["edad"]=difference.years
    generar_plot_estabilidad(dict["estabilidad"], dict["sample_id"])
    if dict["sexo"]==0:
        dict["sexo"]="Masc"
    else:
        dict["sexo"]="Fem"
    return(dict)

def generar_pdf(result,json_patient,json_sensor,cal_id=None,anonimo=False):
    """
    Generates the results report and saves it.

    Args:
        result (results): inference results
        json_patient (json): sample patient information
        json_sensor (json): sample sensor information
        cal_id (str): calibration id
        anonimo (bool): if True, personal data will be anonimized 
    """
    dict=generar_datos(result,json_patient,json_sensor, cal_id=cal_id,anonimo=anonimo)
    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.set_font('helvetica', 'B', 16)
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 15)
    pdf.cell(80)
    pdf.cell(30, 30, 'Reporte Estabilidad', 0, align='C' ,new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln(20)
    pdf.set_font('helvetica', '', 12)
    pdf.set_font('helvetica', '', 12)
    pdf.cell(60, 10, '**Paciente**: '+dict["paciente_id"]+" - "+dict["paciente"], 0,align= 'L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(60, 10, '**Lugar del registro**: '+str(dict["institucion"])+" - "+str(dict["lugar"]), 0, align= 'L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(60, 10, '**Fecha de la muestra**: '+dict["hora_del_dia"] + " " + dict["fecha"], 0, align='L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(60, 10, '**Nro de muestra**: '+str(dict["sample_id"]), 0, align= 'L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT) 
    pdf.cell(60, 10, '**Médico**: '+dict["medico"], 0, align= 'L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.image(name=attachment_path+str(dict["sample_id"])+'_plot_estabilidad.png',x=50,w=100)
    pdf.cell(w=60,h=10, txt='**Tiempo de Actividad**: '+dict["tiempo_activo"],border= 0, align='L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(w=60,h=10, txt='**Tiempo de Inactividad**: '+dict["tiempo_inactivo"],border= 0, align='L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.multi_cell(w=0,h=10, txt='**Observaciones de la muestra:** '+dict["observaciones"],border= 0, align='L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if dict["intervencion"]:
        pdf.cell(w=60,h=10, txt='**Intervención terapéutica:** '+dict["intervencion"],border= 0, align='L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if not dict["pasa_umbral"]:
        pdf.cell(w=60,h=10, txt='**Atención: El tiempo de actividad es menor a 1:10 minutos.**',border= 0, align='L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(w=60,h=10, txt='**El resultado puede no ser certero, se sugiere realizar la muestra nuevamente.**',border= 0, align='L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font('helvetica', '', 8)
    pdf.cell(w=60,h=15,border= 0, align='L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(w=60,h=5, txt='**Versión del autoencoder**: '+dict["modelo autoencoder"],border= 0, align='L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(w=60,h=5, txt='**Versión del clasificador **: '+dict["modelo clasificador"],border= 0,align='L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(w=60,h=5, txt='**Versión del código **: '+dict["commit"],border= 0, align='L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(w=60,h=5, txt='**Sensor Utilizado **: '+str(dict["sensor"]),border= 0, align='L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if dict["cal_sensor"]:
        pdf.cell(w=60,h=5, txt='**Calibración Utilizada **: '+str(dict["sensor"]),border= 0, align='L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.add_page()

    pdf.set_font('helvetica', 'B', 15)
    pdf.cell(80)
    pdf.cell(30, 30, 'Datos del paciente ', 0, align='C' ,new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln(30)
    pdf.set_font('helvetica', '', 12)

    data = (
    ('Fecha Nacimiento', str(dict["fecha_nac"])+ " ("+str(dict["edad"])+" Años)"), 
    ('Altura', str(dict["talla"])),
    ("Peso", str(dict["peso"])),
    ("Género", dict["sexo"]),
    ("Caídas", str(dict["caida"])),
    ("Prescripción de auxiliar de movilidad", str(dict["auxiliar_mov"])),
    ("Patologías previas",  str(dict["patologias_previas"])),
    ("Prescripción de psicofármacos",  str(dict["sicofarmacos"])), #+" "+ dict["nombre_farmaco"]),
    ('Autopercepción de inestabilidad',str(dict["inestabilidad"] )),
    ('Vértigos',str(dict["vertigos"])),
    ('Caídas desde la última muestra', str(dict["caida_ultimo_testeo"])),
    ('Escala Downton',str(dict["escala_downton"]))
)
    y = list(data)
    if dict["sicofarmacos"] != "No":
        y[7]=("Prescripción de psicofármacos",  str(dict["sicofarmacos"])+" "+ str(dict["nombre_farmaco"]))
    data = tuple(y)

    line_height = pdf.font_size * 2.5
    col_width = pdf.epw / 2  # distribute content evenly
    for row in data:
        for datum in row:
            pdf.multi_cell(col_width, line_height, datum, border=1,new_x="RIGHT", new_y="TOP", max_line_height=pdf.font_size)
        pdf.ln(line_height)

    pdf.set_font('helvetica', 'B', 15)
    pdf.cell(80)


    pdf.cell(30, 30, 'Datos de la muestra', 0, align='C' ,new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln(30)
    pdf.set_font('helvetica', '', 12)



    data = (
    ('Piso', dict["piso"]),
    ('Distancia recorrida', dict["distancia"]),
    ('Interior/Exterior',dict["interior_exterior"]),
    ('Se utilizó auxiliar de movilidad en la prueba', dict["uso_auxiliar_mov"]),
    ("Calzado utilizado en la prueba",  dict["calzado"]),
    ('Clasificación del médico',dict["clasificacion_tecnico"]),
)
    line_height = pdf.font_size * 2.5
    col_width = pdf.epw / 2  # distribute content evenly
    for row in data:
        for datum in row:
            pdf.multi_cell(col_width, line_height, datum, border=1,new_x="RIGHT", new_y="TOP", max_line_height=pdf.font_size)
        pdf.ln(line_height)
    

    pdf.output(attachment_path+str(dict["sample_id"])+'_seretest.pdf')
    return(attachment_path+str(dict["sample_id"])+'_seretest.pdf')

#generar_pdf("","")


def generar_encriptar_zip(pdf, id):
    """
    Given the paths to the reports generates a encrypted zip.

    Args:
        pdfs (list): list with the paths to the reports
        ids (list): list with the sample ids
    """
    pyzip.compress_multiple(pdf,[],path_reportes+str(id[0])+'.zip',secret_password,8)
    return(path_reportes+str(id[0])+'.zip')

def Inferir(sample_id,json_patient,mail,json_sensor):
    result=Inferencia(sample_id)
    path_pdf=generar_pdf(result, json_patient, json_sensor)
    insertar_resultado(sample_id, path_pdf,json_patient,result.__dict__)
    mandar_mail(mail,path_pdf,sample_id)
    print(result.activities_time_secs)


def Inferir_test(sample_id,json_patient,mail,json_sensor):
    result=Inferencia(sample_id)
    path_pdf=generar_pdf(result, json_patient, json_sensor)
    mandar_mail(mail,path_pdf)
    print(result.activities_time_secs)


def write_mail_generico(subject, content):
    """
    Envia un mail genérico a "victoria.tournier@serelabs.com" y "juanmateo.ferrari@serelabs.com"

    Args:
        subject (str): Asunto
        sample_id (str): Mensaje

    """
    # mail server parameters
    smtpHost = "smtp.gmail.com"
    smtpPort = 587
    mailUname = mail_sere
    mailPwd = passw
    fromEmail = mail_sere

    # mail body, recepients, attachment files
    mailSubject = subject
    mailContentHtml = content
    recepientsMailList = ["victoria.tournier@serelabs.com", "juanmateo.ferrari@serelabs.com"]
    # create message object
    msg = MIMEMultipart()
    msg['From'] = fromEmail
    msg['To'] = ','.join(recepientsMailList)
    msg['Subject'] = mailSubject
    msg.attach(MIMEText(mailContentHtml, 'html'))

    # Send message object as email using smptplib
    s = smtplib.SMTP(smtpHost, smtpPort)
    s.starttls()
    s.login(mailUname, mailPwd)
    msgText = msg.as_string()
    sendErrs = s.sendmail(fromEmail, recepientsMailList, msgText)
    s.quit()



def write_info_mail_cal(sample_id, nro_cal):
    """Envia un mail de error de calibración a  "victoria.tournier@serelabs.com" y "juanmateo.ferrari@serelabs.com"

    Args:
        sample_id (str): identificador de la muestra
        nro_cal (str):número de calibración del sensor
    """
    write_mail_generico( "ERROR DE CALIBRACIÓN","Error de calibración, la muestra "+str(sample_id)+" no tiene la calibración "+ str(nro_cal)+".")

def write_mail_val_LN_WR(id):
    """Envia un mail de error en el sensor a  "victoria.tournier@serelabs.com" y "juanmateo.ferrari@serelabs.com"
    Args:
        id (str): id de calibración
    """
    write_mail_generico( "ERROR EN EL SENSOR","Error en el sensor, la validación "+ str(id)+" dio mal con WR y LN.")



def generar_zip(path):
    """
    Given the paths to the reports generates a encrypted zip.

    Args:
        pdfs (list): list with the paths to the reports
        ids (list): list with the sample ids
    """
    pyzip = PyZip(PyFolder(path, interpret=False))
    pyzip.save(path+".zip")
    return(path+'.zip')



def generar_plots_historicos(patient_id, fechas, estabilidades, samples, intervencion):
    """
    Generates the history graphs and saves it.

    Args:
        patient_id (str): id of the patient
        fechas (list): list of the sample dates
        estabilidades (list): list of the stability percentages
        samples (list): list of the samples ids
        intervencion (list): list of the interventions
    """
    dEst=[0]
    for i in range (1,len(estabilidades)):
        dEst.append(estabilidades[i]-estabilidades[i-1])

    dEstabilidad=[]
    for i in range (0,len(estabilidades)):
        dEstabilidad.append(estabilidades[i]-estabilidades[0])

    fig = px.scatter(x=fechas, y=estabilidades, text=estabilidades)
    fig.add_trace(go.Scatter(x=fechas, y=estabilidades, line_shape='spline' , showlegend=False,line=dict(color='royalblue', width=2.5)))
    fig.add_hline( y= 50,line_dash="dot",opacity=0.25)
    fig.add_hline( y= 75,line_dash="dot",opacity=0.25)
    fig.add_hline( y= 25,line_dash="dot",opacity=0.25)
    fig.add_hline( y= 100,line_dash="dot",opacity=0.25)
    fig.add_hline( y= 0,line_dash="dot",opacity=0.25)
    fig.add_hrect(
        y0=0, y1=25, line_width=0, 
        fillcolor="#f25829", opacity=0.3)
    fig.add_hrect(
        y0=25, y1=50, line_width=0, 
        fillcolor="#f2a529", opacity=0.3)    
    fig.add_hrect(
        y0=50, y1=75, line_width=0, 
        fillcolor="#eff229", opacity=0.3)
    fig.add_hrect(
        y0=75, y1=100, line_width=0, 
        fillcolor="#85e043", opacity=0.3)        
#"#2bad4e" , "#85e043", "#eff229", "#f2a529", "#f25829"
    fig.update_yaxes(tickvals=[0,25,50,75,100])

    fig.update_traces(textposition="top center")
    fig.update_layout(
    title={
        'text': "Histórico de estabilidad según el número de muestra",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Fecha",
    yaxis_title="Estabilidad (%)",
    font=dict(
        size=15,
    ),
    plot_bgcolor='white',
    width=1500,
    height=500,
    
)
    fig.write_image(attachment_path+str(patient_id)+'_historico.png','png')


    fig = px.scatter(x=fechas, y=dEst, text=estabilidades)
    fig.add_trace(go.Scatter(x=fechas, y=dEst, line_shape='spline' , showlegend=False,line=dict(color='royalblue', width=2.5)))
    fig.update_traces(textposition="bottom center")
    fig.update_layout(
    title={
        'text': "Diferencia de estabilidad con la última muestra",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Fecha",
    yaxis_title="Diferencia de estabilidad",
    font=dict(
        size=15,
    ),
    width=1500,
    height=500,
    
)
    fig.write_image(attachment_path+str(patient_id)+'_historico_diferencias_1.png','png')



    fig = px.scatter(x=fechas, y=dEstabilidad, text=estabilidades)
    fig.add_trace(go.Scatter(x=fechas, y=dEstabilidad, line_shape='spline' , showlegend=False,line=dict(color='royalblue', width=2.5)))
    fig.update_traces(textposition="bottom center")
    fig.update_layout(
    title={
        'text': "Diferencia de estabilidad con la primera muestra",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Fecha",
    yaxis_title="Diferencia de estabilidad",
    font=dict(
        size=15,
    ),
    #plot_bgcolor='white',
    width=1500,
    height=500,
    
)
    fig.write_image(attachment_path+str(patient_id)+'_historico_diferencias_2.png','png')


def generar_datos_extendidos(pat_id, anonimo=False):
    """
    Genera los datos de las últimas 6 muestras del mismo paciente. 
    También llama a la función generar_plots_historicos para generar los gráficos correspondientes.

    Args:
        pat_id (str): identificador del paciente
        anonimo (bool): si es True, el nombre, cédula y médico del paciente se anonimizan y se ponen datos por defecto.
    Returns:
        data (tuple): datos de las últimas 6 muestras del mismo paciente. Si no hay muestras, retorna un string vacío.
        dict (dict): diccionario con datos del paciente.  Si no hay muestras, retorna un string vacío.
        largo (int): cantidad de muestras del paciente (minimo 0, máximo 6).
    """

    datos=datos_muestras_pac(pat_id)
    l=len(datos)
    if l>6:
        datos=datos[:6]
    elif l==0:
        return("","",0)
    datos.reverse()

    fecha=('Fecha',)
    sample_id=('Número de muestra',)
    indice=('Índice de Estabilidad',)
    caidas=("Caídas",)
    aux_mov=("Prescripción de auxiliar de movilidad",)
    clas_medico=('Clasificación del médico',)
    psicofarmacos=("Prescripción de psicofármacos",)
    inest=('Autopercepción de inestabilidad',)
    vertigos=('Vértigos',)
    intervencion=('Intervencion terapéutica',)
    aux_mov_prueba=('Se utilizó auxiliar de movilidad en la prueba',)
    lugar=('Lugar de la prueba',)
    calzado=('Calzado',)
    edad=('Edad',)
    altura=('Altura',)
    peso=("Peso",)
    genero=("Género",)
    fechas=[]
    estabilidades=[]
    samples=[]
    intervenciones=[]
    for dato in datos:
        dictionary=dato[1]
        dictionary = json.loads(dictionary)   
        sample_id=sample_id+(dictionary["sample_id"], )
        samples.append(dictionary["sample_id"])
        est=(dictionary["stable_percentage"]['Caminando'])
        indice=indice+(str(float(f'{est:.2f}')),)
        estabilidades.append( (float(f'{est:.2f}')))
        path=dato[0]
        if path != 0:
        
            json_decrypt=DataSereSensys.decrypt(ruta_priv_key , home +path)
            json_dict=json.loads(json_decrypt)
            json_dict=json_dict[1]
            date=json_dict["fecha"] +" "+ json_dict["hora_del_dia"]
            fecha=fecha+(date,)
            fechas.append(date)
            caidas=caidas+(str(json_dict["caida"]),)
            aux_mov=aux_mov+ (json_dict["auxiliar_de_movilidad"],)
            clas_medico=clas_medico+(json_dict["clasificacion_del_tecnico"],)
            psicofarmacos=psicofarmacos + (json_dict["sicofarmacos"],)
            inest=inest + (json_dict["inestabilidad"],)
            vertigos=vertigos + (json_dict["vertigos"],)
            intervencion=intervencion+(json_dict["intervencion_terapeutica"],)
            intervenciones.append(json_dict["intervencion_terapeutica"])
            aux_mov_prueba=aux_mov_prueba+ (json_dict["uso_aux_mov"],)
            lugar_ext=json_dict["institucion"]+" "+json_dict["lugar"]
            lugar= lugar+(lugar_ext,)
            calzado=calzado + (json_dict["calzado"],)
            fecha_hoy = datetime.strptime(json_dict["fecha"],'%Y-%m-%d').date()
            fecha_nac=datetime.strptime(json_dict["fecha_nac"], '%Y-%m-%d').date()
            difference = relativedelta(fecha_hoy, fecha_nac)
            edad=edad+ (str(difference.years),)
            altura=altura + (json_dict["talla"],)
            peso=peso + (json_dict["peso"],)
            genero = genero +(json_dict["sexo"],)

    data=(fecha,sample_id,indice,clas_medico,lugar,caidas,aux_mov,psicofarmacos,inest,vertigos,intervencion,aux_mov_prueba,calzado,edad,altura,peso,genero)
    dict={
        "paciente_id":json_dict["cedula"],
        "paciente":json_dict["nombre"]+" "+json_dict["apellido"],
        "medico":json_dict["identificador_medico_tratante"],
    }
    if anonimo:
        dict["paciente_id"]="1234657"
        dict["paciente"]="Juan Perez"
        dict["medico"]="7894565"
    largo=len(datos)
    generar_plots_historicos(dict["paciente_id"], fechas, estabilidades, samples, intervenciones)
    return(data,dict,largo)

def generar_historico(pat_id, anonimo=False):
    """
    Genera reporte histórico de las últimas 6 muestras (máximo) del paciente.

    Args:
        pat_id (str): identificador del paciente
        anonimo (bool): indica si es necesario anonimizar los datos sensibles del paciente. Por defecto es Falso.

    Returns:
        (str): path al reporte histórico. 
    """
    data,dict,largo=generar_datos_extendidos(pat_id,anonimo=anonimo)
    if largo ==0:
        return("e0")
    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.set_font('helvetica', 'B', 16)
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 15)
    pdf.cell(80)
    pdf.cell(30, 30, 'Histórico de Estabilidad', 0, align='C' ,new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln(20)
    pdf.set_font('helvetica', '', 12)
    pdf.set_font('helvetica', '', 12)
    pdf.cell(60, 10, '**Paciente**: '+dict["paciente_id"]+" - "+dict["paciente"], 0,align= 'L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(60, 10, '**Médico**: '+dict["medico"], 0, align= 'L',markdown=True,new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.image(name=attachment_path+dict["paciente_id"]+'_historico.png',x=7,w=200)

    pdf.image(name=attachment_path+dict["paciente_id"]+'_historico_diferencias_1.png',x=7,w=200)
    pdf.image(name=attachment_path+dict["paciente_id"]+'_historico_diferencias_2.png',x=7,w=200)

    pdf.add_page()

    pdf.set_font('helvetica', 'B', 15)
    pdf.cell(80)
    pdf.cell(30, 30, 'Datos de las muestras', 0, align='C' ,new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln(30)
    pdf.set_font('helvetica', '', 12)
    
    line_height = pdf.font_size * 2.5
    
    col_width = [pdf.epw / 3]
    for i in range(largo):
        col_width.append(pdf.epw / (3*largo))
    for row in data:
        i=0
        for datum in row:
            #print(datum)
            pdf.multi_cell(col_width[i], line_height, datum, border=1,new_x="RIGHT", new_y="TOP", max_line_height=pdf.font_size)
            i=i+1
        pdf.ln(line_height)

    pdf.set_font('helvetica', 'B', 15)
    pdf.cell(80)

    pdf.output(attachment_path+str(dict["paciente_id"])+'_historico_seretest.pdf')
    return(attachment_path+str(dict["paciente_id"])+'_historico_seretest.pdf')


def reemplazar_info(json_dict, campo,valor):
    json_dict[1][campo]=valor
    return(json_dict)
def encriptar(json_dict,numero_muestra,ruta_destino):
    app_json=json.dumps(json_dict, indent=4, sort_keys=False)
    DataSereSensys.encrypt(ruta_destino , ruta_pub_key, app_json, numero_muestra)

def zipear(path_entrada, path_salida):
    shutil.make_archive(path_salida, 'zip', path_entrada)
#print(generar_historico("00000000"))
