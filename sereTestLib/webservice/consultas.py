
from datetime import datetime
import pymssql
import json
from sereTestLib.webservice.parameters_ws import server, user, password, database, port, home
from SereSensysLib.procesamiento.procesamiento_SereSensys import DataSereSensys
from SereSensysLib.procesamiento.parameters_procesamiento import ruta_priv_key
import sereTestLib.parameters 

def pac_sample_id(ci):
    ''''
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
    "talla": "210", "peso": "78", "sexo": "Fem", "sicofarmacos": "No",
    "caida": "No", "inestabilidad": "No", "auxiliar_de_movilidad": "No", 
    "vertigos": "No", "patologias_previas_conocidas": "ninguna", "sample_id": 73,"nombre_farmaco":""}

    Ejemplo2:
    ---------
    Para un paciente que no existe en la tabla PACIENTES, pac_sample_id(49924590) retorna:

    {"IDENTIFICACION": "", "NOMBRE": "", "NOMBRE2": "", 
    "APELLIDO": "", "APELLIDO2": "", "SEXO": "", "FECHANAC": "", 
    "talla": "", "peso": "",  "sicofarmacos": "",
    "caida": "", "inestabilidad": "", "auxiliar_de_movilidad": "", 
    "vertigos": "", "patologias_previas_conocidas": "",'sample_id': 72,"nombre_farmaco":""}

    Ejemplo3:
    ---------

    Para un paciente que existe en la tabla PACIENTES pero no existe una última muestra disponible 
    o no se pudo acceder a ella, pac_sample_id(12345678) retorna:
    
    {"IDENTIFICACION": "12345678", "NOMBRE": "victoria", "NOMBRE2": "vicky123", 
    "APELLIDO": "Vic", "APELLIDO2": "Tour", "SEXO": 1, "FECHANAC": "5/5/2001", 
    "talla": "", "peso": "",  "sicofarmacos": "",
    "caida": "", "inestabilidad": "", "auxiliar_de_movilidad": "", 
    "vertigos": "", "patologias_previas_conocidas": "", "sample_id": 73,"nombre_farmaco":""}

    '''
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    cursor.execute('SELECT IDENTIFICACION, NOMBRE, NOMBRE2, APELLIDO, APELLIDO2, SEXO, FECHANAC FROM PACIENTES WHERE IDENTIFICACION=%s', ci)
    paciente=cursor.fetchall()
    datos = ("IDENTIFICACION", "NOMBRE", "NOMBRE2", 
    "APELLIDO", "APELLIDO2", "SEXO", "FECHANAC", 
    "talla", "peso", "sicofarmacos",
    "caida", "inestabilidad", "auxiliar_de_movilidad", 
    "vertigos", "patologias_previas_conocidas", "sample_id","nombre_farmaco","escala_downton" )
    x = dict.fromkeys(datos)
    if len(paciente)>0:
        x.update(paciente[0])
        path=(ultima_muestra_paciente(ci))
        if path != 0:
            try:
                json_decrypt=DataSereSensys.decrypt(ruta_priv_key , home +path)
                json_dict=json.loads(json_decrypt)
                persona=json_dict[1]
                print(persona)
                keys_to_extract = ["talla", "peso","sicofarmacos","caida","inestabilidad","auxiliar_de_movilidad","vertigos","patologias_previas_conocidas","nombre_farmaco"]
                info = {key: persona[key] for key in keys_to_extract}
                x.update(info)
                if "escala_downton" in persona:
                    y={"escala_downton":persona["escala_downton"]}
                else:
                    y={"escala_downton"}
                x.update(y)
            except:
                print("no")
    cursor2 = conn.cursor(as_dict=False)
    cursor2.execute('SELECT NEXT VALUE FOR SECUENCIA_MUESTRAS')
    sample_id=cursor2.fetchall()
    conn.close()
    y = {"sample_id":sample_id[0][0]}
    x.update(y)
    return(json.dumps(x))


def insertar_muestra(dict, path,cal_id=""):
    '''
    Función que recibe un diccionario con la información de la muestra y el directorio donde está almacenada la muestra
    Inserta el número de muestra, la fecha,la identificación del paciente y del médico 
    y el directorio en donde está guardado el json en la tabla "MUESTRAS" de la base de datos.

    Parameters:
    ----------

    dict (dict): diccionario con la información de la muestra
    path (str): el directorio donde se encuentra la muestra
    ''' 
        
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    sql = "INSERT INTO MUESTRAS (NROMUESTRA, FECHAMUESTRA, PACIENTE,MEDICO,RUTAARCHIVOS,INSTITUCION,LUGAR, NROCALIBRACION, INTERVENCIONTERAPEUTICA,SENSOR) VALUES (%s, %s,%s,%s,%s,%s,%s,%s, %s,%s)"
    cursor.execute(sql, (dict["numero_muestra"],dict["fecha"],dict["cedula"],dict["identificador_medico_tratante"],path.split(str(home))[-1],dict["institucion"],dict["lugar"],cal_id,dict["intervencion_terapeutica"],dict["sensor_utilizado"]))
    conn.commit()
    conn.close

    #cursor = conn.cursor(as_dict=True)
    #cursor.execute('SELECT * FROM MUESTRAS WHERE NROMUESTRA=%s', 3000)
    #print(cursor.fetchall())


def insertar_resultado(sample_id, path_pdf,json_muestra,json_resultado):
    """
    Función que inserta en la tabla RESULTADOMUESTRA 
    el número de muestra, el médico, paciente, resulado y el directorio del reporte de los resultados.

    Args:
        sample_id (str): número de muestra del paciente.
        path_pdf (str): directorio del reporte de resultados.
        json_muestra (json): json con información sobre la muestra.
        json_resultado (json): json con la información de la inferencia.

    """
    json_resultado=json.dumps(json_resultado)
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    sql = "INSERT INTO RESULTADOSMUESTRA (NROMUESTRA, PACIENTE, MEDICO,REPORTE, RESULTADOJSON, MODOSENSOR) VALUES (%s,%s,%s, %s,%s,%s)"
    cursor.execute(sql, (sample_id,json_muestra["cedula"],json_muestra["identificador_medico_tratante"],path_pdf.split(str(home))[-1],(json_resultado),sereTestLib.parameters.accel_type))
    conn.commit()
    conn.close


def existe_agregar_paciente(dict):
    """
    Función que recibe la información de un paciente y consulta a la base de datos si existe en la tabla PACIENTES
    Si no existe, lo agrega.
    Si existe actualiza los datos.

    Args:
        dict (dict): diccionario con la información de un paciente.       
    """
    ci=dict["cedula"]
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    cursor.execute('SELECT IDENTIFICACION FROM PACIENTES WHERE IDENTIFICACION=%s', ci)
    paciente=cursor.fetchall()

    if dict['sexo']=='Masc':
        dict['sexo']=0
    elif dict['sexo']=='Fem':
        dict['sexo']=1
    if len(paciente)==0:
        cursor2 = conn.cursor(as_dict=True)
        sql = "INSERT INTO PACIENTES (IDENTIFICACION, NOMBRE, NOMBRE2, APELLIDO, APELLIDO2, SEXO, FECHANAC) VALUES (%s, %s,%s,%s,%s,%s,%s)"
        cursor2.execute(sql, (dict["cedula"],dict["nombre"],dict["segundo_nombre"],dict["apellido"],dict["segundo_apellido"],dict["sexo"],dict["fecha_nac"]))
        conn.commit()
    else:
        cursor2 = conn.cursor(as_dict=True)
        sql = "UPDATE PACIENTES SET NOMBRE=%s, NOMBRE2=%s, APELLIDO=%s, APELLIDO2=%s, SEXO=%s, FECHANAC=%s WHERE IDENTIFICACION=%s"
        cursor2.execute(sql, (dict["nombre"],dict["segundo_nombre"],dict["apellido"],dict["segundo_apellido"],dict["sexo"],dict["fecha_nac"],dict["cedula"]))
        conn.commit()

    conn.close()


def obtener_mail(dict):
    """
    Función que obtiene el email del médico según su identificador. Si no encuentra retorna False
    Args:
        dict (dict): diccionario donde dict["identificador_medico_tratante"] es el identificador.
    Returns:
        (str): email del médico

    Ejemplo: 
    --------    
    dict =  {"cedula":49924590, "nombre":"Victoria", "apellido":"Tournier","sexo":"1","fecha_nac":"1998-06-08", "identificador_medico_tratante":"JMF"}
    obtener_mail(dict)
    Retorna: juanmateo.ferrari@serelabs.com
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    cursor.execute('SELECT MAIL FROM MEDICOS WHERE IDENTIFICACION=%s', dict["identificador_medico_tratante"])
    mail=cursor.fetchall()
    conn.close()
    if len(mail)==0:
        return(False)
    return(mail[0]['MAIL'])



def pac_results(pac_id):
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
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    cursor.execute('SELECT NROMUESTRA,PACIENTE, MEDICO, FECHAMUESTRA FROM MUESTRAS WHERE PACIENTE=%s', str(pac_id))
    paciente=cursor.fetchall()
    muestras=[]
    for pac in paciente:
        cursor1 = conn.cursor(as_dict=True)
        cursor1.execute('SELECT REPORTE FROM RESULTADOSMUESTRA WHERE NROMUESTRA=%s',str(pac["NROMUESTRA"]))
        path=cursor1.fetchone()
        if path:
            muestras.append(pac)
    #print(muestras)
    conn.close()
    return(muestras)


def reporte(sample):
    """
   Función que consulta a la tabla RESULTADOSMUESTRA por la ubicación del reporte de resultados de una muestra.

    Args:
        sample (str): Número de muestra a consultar

    Reurns:
        (list): lista con la ubicación de los resultados.
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor1 = conn.cursor(as_dict=True)
    cursor1.execute('SELECT REPORTE FROM RESULTADOSMUESTRA WHERE NROMUESTRA=%s',str(sample))
    path=cursor1.fetchone()
    pdf=path["REPORTE"]
    conn.close()
    return(home+pdf)

def doc_id(id):
    """
    Función que chequea que el médico exista en la tabla MEDICOS

    Args:
        id (str): identificador del médico
    Returns:
        (bool): True si existe y False si no existe.
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    cursor.execute('SELECT MAIL FROM MEDICOS WHERE IDENTIFICACION=%s',str(id))
    mail=cursor.fetchall()
    conn.close()
    return(len(mail)>0)

def devolver_feedback(sample_id):
    """
    Función que dado un numero de muestra devuelve el feedback del reporte.

    Args:
        sample_id (str): número de muestra
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    cursor.execute('SELECT FEEDBACK FROM RESULTADOSMUESTRA WHERE NROMUESTRA=%s',str(sample_id))
    feedback=cursor.fetchall()
    conn.close()
    return(feedback[0]["FEEDBACK"])


def insertar_feedback(feedback,sample_id):
    """
    Función que dado un numero de muestra inserta el feedback del reporte.

    Args:
        feedback (str): devolución del médico
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    
    cursor2 = conn.cursor(as_dict=True)
    sql = "UPDATE RESULTADOSMUESTRA SET FEEDBACK=%s WHERE NROMUESTRA=%s"
    cursor2.execute(sql, (feedback,str(sample_id)))
    conn.commit()
    conn.close()

def devolver_instituciones_medico(medico):
    """
    Función que devuelve las instituciones asociadas al médico
    
    Args:
        medico (str): identificador del médico

    Returns:
        (str): lista con las instituciones asociadas
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    cursor.execute('SELECT INSTITUCION FROM INSTITUCIONES WHERE MEDICO=%s',str(medico))
    inst=cursor.fetchall()
    conn.close()
    instituciones=[]
    for i in inst:
        instituciones.append(i["INSTITUCION"])
    return(instituciones)

def devolver_lugares_instituciones(instituciones):
    """
    Función que devuelve los lugares asociados a las instituciones

    Args:
        instituciones (str): institucion 
    Returns: 
        (list): lista con los lugares asociados a las instituciones
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    cursor.execute('SELECT LUGAR FROM LUGARES WHERE INSTITUCION=%s',str(instituciones))
    lugar=cursor.fetchall()
    conn.close()
    lugares=[]
    for l in lugar:
        lugares.append(l["LUGAR"])
    return(lugares)


def ultima_muestra_paciente(id):
    """
    Devuelve el path del json conteniendo la información de la última muestra del paciente.
    Retorna 0 si no hay muestras previas.

    Args:
        id (str): identificador del paciente
    
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)

    cursor.execute('SELECT RUTAARCHIVOS FROM MUESTRAS WHERE PACIENTE=%s',id)
    path=cursor.fetchall()
    conn.close()
    if len(path)>0:
        return(path[-1]['RUTAARCHIVOS'])
    else: 
        return 0


def ultima_calibracion(sensor_id):
    """
    Devuelve el último número de calibracion del sensor.

    Args:
        sensor_id (str): Identificador del sensor
    
    returns:
        str: último número de calibracion del sensor
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)

    cursor.execute('SELECT NROCALIBRACION FROM CALIBRACIONSENSOR WHERE SENSOR=%s',sensor_id)
    nro_cal=cursor.fetchall()
    conn.close()
    return(nro_cal[-1])


def calibracion(cal_id):
    """
   Función que consulta a la tabla CALIBRACIONSENSOR por la ubicación de la calibración.

    Args:
        cal_id (str): Número de calibracion a consultar

    Reurns:
        (str): ubicación de la calibración.
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor1 = conn.cursor(as_dict=True)
    cursor1.execute('SELECT RUTA FROM CALIBRACIONSENSOR WHERE NROCALIBRACION=%s',str(cal_id))
    path=cursor1.fetchone()
    conn.close()
    return(home+path["RUTA"])


def sensor(cal_id):
    """
    Devuelve el sensor correspondiente al identificador de la calibración.

    Args:
        cal_id (str): número de calibración del sensor
    
    Returns:
        str: identificador del sensor
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)

    cursor.execute('SELECT SENSOR FROM CALIBRACIONSENSOR WHERE NROCALIBRACION =%s',cal_id)
    sensor=cursor.fetchone()
    conn.close()
    return(sensor)

def return_cal_id():
    """
    Retorna el próximo valor para el identificador de calibración consultando la secuencia SECUENCIA_CALIBRACIONES.
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor2 = conn.cursor(as_dict=False)
    cursor2.execute('SELECT NEXT VALUE FOR SECUENCIA_CALIBRACIONES')
    cal_id=cursor2.fetchone()
    conn.close()
    return(cal_id[0])


def insert_cal(cal_id, sensor, path, tecnico):
    """
    Inserta a la tabla de CALIBRACIONSENSOR la nueva calibración.
    Args:
        cal_id (str):
        sensor (str):
        path (str):
        tecnico (str):
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor2 = conn.cursor(as_dict=True)
    sql = "INSERT INTO CALIBRACIONSENSOR (NROCALIBRACION, SENSOR, FECHA, TECNICO, RUTA) VALUES (%s, %s,%s,%s,%s)"
    cursor2.execute(sql, (cal_id,sensor,datetime.now().strftime('%Y%m%d %H:%M:%S'),tecnico,path))
    conn.commit()

def datos_muestras_pac(pat_id):
    """
    Función que dado un paciente devuelve los datos de todas sus muestras

    Args: 
        pat_id (str): identificador de paciente
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor2 = conn.cursor(as_dict=False)
    cursor2.execute('SELECT MUESTRAS.RUTAARCHIVOS , RESULTADOSMUESTRA.RESULTADOJSON FROM RESULTADOSMUESTRA INNER JOIN MUESTRAS ON (RESULTADOSMUESTRA.NROMUESTRA=MUESTRAS.NROMUESTRA)   WHERE RESULTADOSMUESTRA.PACIENTE=%s',pat_id)
    cal_id=cursor2.fetchall()
    conn.close()
    return(cal_id)



def cambiar_val(tabla, valor, condicion):
    """
    Dada una tabla, un valor a cambiar y una condición, lo cambia en la base de datos.
    Imprime los cambios luego de realizados

    Args:
        tabla (str): tabla a realizar el cambio
        valor (str): valor a cambiar
        condicion (str): condición que si se cumple se cambia el valor

    Ejemplo de uso:

    cambiar_val("PACIENTES","SEXO=0","IDENTIFICACION="+"'"+49924590+"'") 
    en la tabla PACIENTES, cambia el SEXO a 0 (Masc) a la entrada que tiene la identificación 49924590.

    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    sql = "UPDATE "+str(tabla)+ " SET "+ str(valor)+" WHERE " + str(condicion)
    cursor.execute(sql)
    conn.commit()
    cursor2 = conn.cursor(as_dict=True)
    sql = "SELECT * FROM "+ str(tabla)+" WHERE "+condicion
    cursor2.execute(sql)
    resultado=cursor2.fetchall()
    conn.close()
    print(resultado)

def obtener_json(nromuestra):
    """ 
    Funcion que a partir del numero de muestra devuelve el json asociado.  Devuelve tambien el path del json
   Si no lo encuentra imprime el número de muestra.

    Args:
        nromuestra (str): identificador de la muestra
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor2 = conn.cursor(as_dict=False)
    cursor2.execute('SELECT RUTAARCHIVOS FROM MUESTRAS WHERE NROMUESTRA=%s',nromuestra)
    sample=cursor2.fetchone()
    #print(sample_id)
    try:
        json_decrypt=DataSereSensys.decrypt(ruta_priv_key , home +sample[0])
        json_dict=json.loads(json_decrypt)
        print(json_dict)
        return(json_dict,home+sample[0])

    except:
        print(sample)


def muestras_institucion(institucion, mayor_65=False):
    """ Devuelve una lista de muestras tomadas en determinada institucion. 

    Args:
        institucion (str): institucion de la cual se quiere devolver las muestras
        mayor_65 (bool): Si es True, se devuelven las muestras de los pacientes que son mayores de 65 años. Por defecto es False
    """
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor2 = conn.cursor(as_dict=False)
    if mayor_65:
        cursor2.execute('SELECT * FROM MUESTRAS INNER JOIN PACIENTES ON MUESTRAS.PACIENTE=PACIENTES.IDENTIFICACION WHERE MUESTRAS.INSTITUCION=%s AND DATEDIFF(YY, PACIENTES.FECHANAC, GETDATE())>64',institucion)
    else:
        cursor2.execute('SELECT * FROM MUESTRAS WHERE INSTITUCION=%s',institucion)
    samples=cursor2.fetchall()
    return(samples)

def return_resultado(nromuestra):
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    cursor.execute('SELECT * FROM RESULTADOSMUESTRA WHERE NROMUESTRA=%s',str(nromuestra))
    resultado=cursor.fetchone()
    return resultado
    #print(pac_sample_id("00000000"))

#print(datos_muestras_pac("49924590"))

# SELECT ProductID, Purchasing.Vendor.BusinessEntityID, Name
# FROM Purchasing.ProductVendor INNER JOIN Purchasing.Vendor
#     ON (Purchasing.ProductVendor.BusinessEntityID = Purchasing.Vendor.BusinessEntityID)
# WHERE StandardPrice > $10
#   AND Name LIKE N'F%'
# GO

#print(devolver_lugares_instituciones("Sere"))
#print(ultima_muestra_paciente('49924590'))

#reporte([1003])
#pac_results(49)
#json_dict =  {"cedula":49, "nombre":"Victoria", "apellido":"Tournier","sexo":"1","fecha_nac":"1998-06-08", "identificador_medico_tratante":"JMF","segundo_nombre":"Victoria","segundo_apellido":""}
#existe_agregar_paciente(json_dict)
#print(pac_sample_id(49))
#print(obtener_mail(json_dict))

#json_dict =  {"cedula":4924590, "nombre":"Victoria", "apellido":"Tournier","sexo":"1","fecha_nac":"1998-06-08"}
#existe_agregar_paciente(json_dict)

# conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
# cursor = conn.cursor(as_dict=True)
# sql = "INSERT INTO MEDICOS (IDENTIFICACION, NOMBRE, APELLIDO,MAIL) VALUES (%s, %s,%s,%s)"
# #sql="UPDATE MEDICOS SET INSTITUCION= %s, LUGAR= %s WHERE IDENTIFICACION=%s"
# cursor.execute(sql, ("VMV", "Matias", "Viva","matias.viva@serelabs.com" ))
# conn.commit()


# conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
# cursor = conn.cursor(as_dict=True)

# cursor.execute('SELECT RUTAARCHIVOS FROM MUESTRAS WHERE PACIENTE=49835222')
# mail=cursor.fetchall()
# print(mail[-1]['RUTAARCHIVOS'])
# conn.close()


# cursor2 = conn.cursor(as_dict=False)
# cursor2.execute('SELECT * FROM CALIBRACIONSENSOR')
# sample_id=cursor2.fetchall()
# print(sample_id)

#print(doc_id("vic"))

# conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
# cursor = conn.cursor(as_dict=True)
# cursor.execute(
# "CREATE TABLE LUGARES (LUGAR  VARCHAR(100) NOT NULL, INSTITUCION  VARCHAR(100) NOT NULL,  CONSTRAINT PK_LUGARES PRIMARY KEY (LUGAR , INSTITUCION))"
# )
# conn.commit()
# conn.close()



# conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
# cursor = conn.cursor(as_dict=True)
# sql = "UPDATE RESULTADOSMUESTRA SET REPORTE =%s WHERE NROMUESTRA=1166"
# cursor.execute(sql,("/home/sere/Dropbox/SereNoTocar/sereData/attachments/1166_test.pdf"))
# conn.commit()
# conn.close()
# insertar_feedback("5 bien",1166)



# conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
# cursor = conn.cursor(as_dict=True)
# cursor.execute(
# "CREATE TABLE INSTITUCIONES (MEDICO VARCHAR(100) NOT NULL, INSTITUCION  VARCHAR(100) NOT NULL,  CONSTRAINT PK_INSTITUCIONES PRIMARY KEY (MEDICO , INSTITUCION))"
# )
# conn.commit()
# conn.close()


# conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
# cursor = conn.cursor(as_dict=True)
# sql = "INSERT INTO INSTITUCIONES (MEDICO, INSTITUCION) VALUES (%s, %s)"
# cursor.execute(sql, ("MVT","Replica"))
# conn.commit()
# conn.close()

# conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
# cursor = conn.cursor(as_dict=True)
# cursor.execute('SELECT * FROM MEDICOS')
# mail=cursor.fetchall()
# print(mail)

# conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
# cursor = conn.cursor(as_dict=True)
# cursor.execute('SELECT * FROM RESULTADOSMUESTRA')
# mail=cursor.fetchall()
# print(mail)



# conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
# cursor = conn.cursor(as_dict=True)
# cursor.execute("ALTER TABLE RESULTADOSMUESTRA ADD MODOSENSOR VARCHAR(8000)")
# conn.commit()
# conn.close()



# cursor2 = conn.cursor(as_dict=False)
# cursor2.execute('SELECT * FROM PACIENTES')
# sample_id=cursor2.fetchall()
# print(sample_id)
#print( pac_sample_id(ci))