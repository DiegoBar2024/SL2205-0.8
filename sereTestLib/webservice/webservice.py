from random import sample
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List, Optional
from zipfile import ZipFile
import sys,os


sys.path.append(os.getcwd())

from sereTestLib.webservice.consultas import *
from sereTestLib.webservice.parameters_ws import zip_path, to_unzip_path, calibs, cal_path , cal_path
from SereSensysLib.procesamiento.procesamiento_SereSensys import procesamiento_SS, validacion_calibraciones
from sereTestLib.webservice.functions import *
from sereTestLib.parameters import dir_in_original_sample
import sereTestLib.parameters

app = FastAPI(title="SereTest WebApp", description="Serelabs",
    version="0.0.1",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "Serelabs ML Team",
        "email": "francisco.deizaguirre@serelabs.com",
    }
)



@app.get("/")
async def root():
    return {"message": "Serelabs"}




@app.get("/id/{id}")
async def searchPatientWS(id:str):
    """
   Función que recibe un identificador de paciente.
    Se realiza la busqueda del mismo, y se reserva el número de muestra con la secuencia SECUENCIA_MUESTRAS.
    (El número de muestra cambia según el contenido de la secuencia SECUENCIA_MUESTRAS)
    Agrega también información de la última muestra realizada del paciente (si la hay).

    Parameters:
    ----------
    ci(str): identificador del paciente

    Returns:
    -------
    (json): Si existe el paciente retorna el contenido de la tabla PACIENTES y el número de muestra.
    Si no existe el paciente retorna el número de muestra.

    Ejemplo1:
    -------
    Para un paciente que existe en la tabla PACIENTES y existe una última muestra disponible, pac_sample_id(12345678) retorna:

    {"IDENTIFICACION": "12345678", "NOMBRE": "victoria", "NOMBRE2": "vicky123", 
    "APELLIDO": "Vic", "APELLIDO2": "Tour", "SEXO": 1, "FECHANAC": "5/5/2001", 
    "talla": "210", "peso": "78", "sicofarmacos": "No",
    "caida": "No", "inestabilidad": "No", "auxiliar_de_movilidad": "No", 
    "vertigos": "No", "patologias_previas_conocidas": "ninguna", "sample_id": 73}

    Ejemplo2:
    ---------
    Para un paciente que no existe en la tabla PACIENTES, pac_sample_id(49924590) retorna:

    {"IDENTIFICACION": "", "NOMBRE": "", "NOMBRE2": "", 
    "APELLIDO": "", "APELLIDO2": "", "SEXO": "", "FECHANAC": "", 
    "talla": "", "peso": "", "sicofarmacos": "",
    "caida": "", "inestabilidad": "", "auxiliar_de_movilidad": "", 
    "vertigos": "", "patologias_previas_conocidas": "",'sample_id': 72}

    Ejemplo3:
    ---------

    Para un paciente que existe en la tabla PACIENTES pero no existe una última muestra disponible 
    o no se pudo acceder a ella, pac_sample_id(12345678) retorna:
    
    {"IDENTIFICACION": "12345678", "NOMBRE": "victoria", "NOMBRE2": "vicky123", 
    "APELLIDO": "Vic", "APELLIDO2": "Tour", "SEXO": 1, "FECHANAC": "5/5/2001", 
    "talla": "", "peso": "", "sicofarmacos": "",
    "caida": "", "inestabilidad": "", "auxiliar_de_movilidad": "", 
    "vertigos": "", "patologias_previas_conocidas": "", "sample_id": 73}

    """
    return({'result':pac_sample_id(id)})

@app.post("/upload")
async def sendSampleWS(background_tasks: BackgroundTasks,checksum:str,no_coincide:int=0,nro_cal:str= "",file: UploadFile= File(...)):
    
    """
    Función que recibe un checksum y un archivo zip con un json  encriptado y los datos binarios.
    Verifica la integridad del zip con el checksum y desencripta el json.
    Se procesa la muestra, se almacena en sample_path, se comprueba la aceleración en Y.
    Si no hubo errores, si el paciente no existía en la tabla de pacientes se lo agrega.
    Se guarda el contenido del json en la tabla de muestras. 
    Se infirere el tiempo de actividad y se realiza la inferencia de la muestra.
    Se manda un mail con los resultados al médico.
    Retorna si no hubo errores un mensaje, de lo contrario un código de error.

    La función espera que el nombre del zip sea el número de muestra.

    Parameters: 

        checksum (str): checksum para verificar la integridad del zip.
        file (bytes): Zip con un json encriptado y los datos binarios.

    Returns:

        si no hubo errores:

            (message): Inference in the background

        si hubo errores:
        
            e0(str): fallo al guardar los datos
            e1(str): el servidor recibió un msj vacío
            e2(str): falló verificación de checksum
            e3(str): fallo al desencriptar
            e4(str): fallo al crear CSV
            e5(str): fallo al comprobar aceleración en Y
            e6(str): no se pudo mandar el mail al médico
    """

    content = file.file.read()
    with open(zip_path+file.filename, "wb") as new_file:
        new_file.write(content)
        new_file.close()

    if not success_to_save(zip_path+file.filename):
        return("e0")
    if not is_non_zero_file(zip_path+file.filename):
        return("e1")
    #if not verificar_checksum(checksum, zip_path+file.filename):
    #    return("e2")
    sample_id=obtener_sample_id(file.filename)

    if no_coincide:
        write_info_mail_cal(sample_id,nro_cal)
    unzip_folder=to_unzip_path+str(sample_id)+"/"
    zip_archive = ZipFile(zip_path+str(sample_id)+".zip")
    zip_archive.extractall(f"{unzip_folder}")
    output_procesamiento=procesamiento_SS(sample_id, to_unzip_path,dir_in_original_sample)
    if isinstance(output_procesamiento, tuple):  # si procesamiento_SS devuelve una tupla asigno los diccionarios,
                                                 # si no devuelvo el mensaje de error
        json_sensor, json_patient = output_procesamiento
    else:
        if output_procesamiento == ("e3") or output_procesamiento == ("e4"):
            return output_procesamiento
    existe_agregar_paciente(json_patient)
    if not comprobar_AC_y(sample_id,8,10):
        sereTestLib.parameters.accel_type="Accel_"+"WR"
    mail=obtener_mail(json_patient)
    if not mail:
        return("e6")
    existe_agregar_paciente(json_patient)
    insertar_muestra(json_patient,unzip_folder+str(sample_id)+".bin",nro_cal)
    
    background_tasks.add_task(Inferir,sample_id,json_patient,mail,json_sensor)
    return {"message": "Inference in the background"}



@app.get("/idHistory/{id}")
async def queryForHistoryWS(id:str):
    """
    Función que dado un id de paciente consulta en la tabla MUESTRAS para encontrar las muestas asociadas a ese paciente.
    Luego consulta a la tabla RESULTADOSMUESTRA para obtener los numeros de muestra que están disponibles los resultados(ya que puede haber muestras sin resultados).
    

    Args:
        id (str): id del paciente
    Returns:
        (list): lista con el número de muestra, identificador de paciente, médico  y fecha de las muestras que están disponibles los resultados.

    Ejemplo:
    --------
    pac_results(49) retorna: [{'NROMUESTRA': 247, 'PACIENTE': '49', 'MEDICO': 'JMF', 'FECHAMUESTRA': datetime.date(2022, 7, 4)}]
    """
    return(pac_results(id))

@app.get("/reportes/{id}")
def generateReportsWS(id: str):
    """
    Función que busca el reporte de la muestra y los comprime en un zip con contraseña

    Args:

        ids (str): identificador de la muestra.

    Returns:

        (zip)
    """
    pdfs=reporte(id)
    zip=generar_encriptar_zip([pdfs], [id])

    return FileResponse(zip, media_type='application/zip', filename= 'archive.zip')


@app.get("/historico/{id}")
def generateHistoricalReportsWS(id: str):
    """
    Función que busca el reporte historico del paciente y los comprime en un zip con contraseña

    Args:

        id (str): identificador del paciente.

    Returns:
        (zip): con el historico del paciente
        e0 (str): si no existe un historico del paciente

    """
    pdfs= generar_historico(id)
    if pdfs=="e0":
        return(pdfs)
    zip=generar_encriptar_zip([pdfs], [id])

    return FileResponse(zip, media_type='application/zip', filename= 'archive.zip')



@app.get("/id_doc/{id}")
async def searchDoctorWS(id:str):
    """
    Función que busca si existe el médico en la base de datos.

    Parameters:

        id (str): identificador del médico

    Returns:

        (bool): Retorna True si existe el médico en la tabla MEDICOS y False si no.
    """
    return({'result':doc_id(id)})


@app.get("/feedback/{id}")
async def searchFeedbackWS(id:str):
    """
    Función que busca el feedback de la muestra en la base de datos.

    Parameters:

        id (str): identificador de la muestra

    """
    return(devolver_feedback(id))

@app.get("/insertfeedback/{sample_id}")
def insertFeedbackWS(sample_id: str,feedback:str):
    """_
    Inserta el feedback de la muestra en la base de datos

    Args:
        sample_id (str): identificador de la muestra
        feedback (str): feedback del médico
    """
    insertar_feedback(feedback,sample_id)

@app.get("/inst_medico/{id}")
async def searchInstitucionesWS(id:str):
    """
    Función que busca las instituciones asociadas al médico en la base de datos.

    Parameters:

        id (str): identificador del médico

    """
    return(devolver_instituciones_medico(id))

@app.get("/lugar_inst/{inst}")
async def searchLugaresWS(inst:str):
    """
    Función que busca las lugares asociados a las instituciones en la base de datos.

    Parameters:

        inst (str): institución

    """
    return(devolver_lugares_instituciones(inst))

@app.post("/calibracion")
def calibracionWS(id:str,file: UploadFile= File(...)):
    content = file.file.read()
    with open(zip_path+file.filename, "wb") as new_file:
        new_file.write(content)
        new_file.close()
    zip_archive = ZipFile(zip_path+file.filename)
    zip_archive.extractall(f"{to_unzip_path}"+"/"+id)
    result1, result2, flag_ambas_mal=validacion_calibraciones(id, to_unzip_path,calibs+"/"+id )
    if flag_ambas_mal:
        write_mail_val_LN_WR(id)
    print("LN: ")
    print(result1)
    print("WR: ")
    print(result2)
    return(result1, result2, flag_ambas_mal)




@app.post("/upload_test")
async def sendSampleWS_test(background_tasks: BackgroundTasks,checksum:str,file: UploadFile= File(...)):
    """ Funcion para pruebas
    """

    content = file.file.read()
    with open(zip_path+file.filename, "wb") as new_file:
        new_file.write(content)
        new_file.close()
    sample_id=obtener_sample_id(file.filename)
    unzip_folder=to_unzip_path+"/"+str(sample_id)
    zip_archive = ZipFile(zip_path+str(sample_id)+".zip")
    zip_archive.extractall(f"{unzip_folder}")
    output_procesamiento=procesamiento_SS(sample_id, to_unzip_path,dir_in_original_sample)
    if isinstance(output_procesamiento, tuple):
        json_sensor, json_patient = output_procesamiento
    else:
        if output_procesamiento == ("e3") or output_procesamiento == ("e4"):
            return output_procesamiento
    mail=obtener_mail(json_patient)
    existe_agregar_paciente(json_patient)
    if not comprobar_AC_y(sample_id,8,10):
        sereTestLib.parameters.accel_type="Accel_"+"WR"
    background_tasks.add_task(Inferir_test,sample_id,json_patient,mail,json_sensor)
    return {"message": "Inference in the background"}


@app.get("/ultimacalibracion/{sensor_id}")
async def ultimaCalibracionWS(sensor_id:str):
    """
    Recibe el id del sensor y devuelve el número de la última calibración válida del sensor
    Args:
        sensor_id (str): id del sensor

    Returns:
        dict: número de la última calibración válida
    
    Ejemplo:
    -------

    con el request http://127.0.0.1:3000/ultimacalibracion/2001  (sensor_id=2001)
    retorna {"NROCALIBRACION": 12}
 

    """

    return(ultima_calibracion(sensor_id))


@app.get("/returncalibracion/{cal_id}")
async def returnCalibracionWS(cal_id:str):
    """
    Recibe un número de calibración y devuelve la calibración correspondiente
    Args:
        cal_id (str): número de calibración
    Returns:
        (zip)
    """
    #busco el path de la calibracion
    path=calibracion(cal_id)

    return FileResponse(path, media_type='application/zip', filename= str(cal_id)+'.zip')

@app.post("/uploadcalibracion/")
async def uploadCalibracionWS(tecnico:str,checksum:str,file: UploadFile= File(...)):
    """
    Recibe la calibracion encriptada con el checksum y la agrega a la base de datos. 
    Args:

        checksum (str): checksum para verificar la integridad del zip.
        file (bytes): Zip con la calibración

    Returns:

        si no hubo errores:

            0 (str)

        si hubo errores:
        
            e0(str): fallo al guardar los datos
            e1(str): el servidor recibió un msj vacío
            e2(str): falló verificación de checksum
    """
    content = file.file.read()
    with open(cal_path+file.filename, "wb") as new_file:
        new_file.write(content)
        new_file.close()
    if not success_to_save(cal_path+file.filename):
        return("e0")
    if not is_non_zero_file(cal_path+file.filename):
        return("e1")
    #if not verificar_checksum(checksum, zip_path+file.filename):
    #    return("e2")
    sensor=obtener_sample_id(file.filename)
    cal_id=return_cal_id()
    zip_path_name= cal_path+str(cal_id)+'.zip'
    os.rename(cal_path+file.filename, zip_path_name)
    insert_cal(cal_id, sensor,zip_path_name.split(str(home))[-1] , tecnico)
    return(0)




from uvicorn import run
if __name__== '__main__':
    #run("webservice:app", host="192.9.200.90", port=5000,reload=True)
    run("sereTestLib.webservice.webservice:app", port=3000,reload=True)


    
