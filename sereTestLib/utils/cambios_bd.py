from random import sample
import pymssql
import json
import os



from sereTestLib.webservice.parameters_ws import server, user, password, database, port, home,zip_path
from sere_sensys_gui.procesamiento.dataSereSensys import DataSereSensys
from sere_sensys_gui.procesamiento.parameters_procesamiento import ruta_priv_key 
from sereTestLib.webservice.consultas import cambiar_val, obtener_json

from sereTestLib.webservice.functions import reemplazar_info, zipear, encriptar









conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
cursor2 = conn.cursor(as_dict=False)
cursor2.execute('SELECT NROMUESTRA, RUTAARCHIVOS FROM MUESTRAS WHERE MEDICO=%s',("19329704"))
sample_id=cursor2.fetchall()
#print(sample_id)
for sample in sample_id:
    try:
        json_decrypt=DataSereSensys.decrypt(ruta_priv_key , home +sample[1])
        json_dict=json.loads(json_decrypt)
        json_dict=json_dict[1]
        print(json_dict)
    except:
        print(sample)


sexo_masc=['1169258-9','6281862-5','1142801-1','543402-0','1306125-7','5494011-7']
for persona in sexo_masc:
    cambiar_val("PACIENTES","SEXO=0","IDENTIFICACION="+"'"+persona+"'")



cedulas_mal_fecha_nac=['1120124-5','550740-1','572343-3','6281862-5','710604-7','711023-4','781742-4','970228-3']
fecha_nac_bien=['1956-05-20','1939-10-22','1934-12-21','1955-05-07','1936-03-24','1935-10-21','1934-01-05','1943-03-30']

for persona, fecha in zip(cedulas_mal_fecha_nac, fecha_nac_bien):
    cambiar_val("PACIENTES", "FECHANAC="+"'"+fecha+"'", "IDENTIFICACION="+"'"+persona+"'")

cambiar_intervencion_terapeutica=[13,15,17,19,21,23,25]

for sample_id in cambiar_intervencion_terapeutica:
    json_dict, path=obtener_json(str(sample_id))
    json_dict=reemplazar_info(json_dict,'intervencion_terapeutica','Ninguno')
    nombre_archivo= os.path.split(path)
    encriptar(json_dict,nombre_archivo[-1].split(".")[0],nombre_archivo[0])
    zipear(path_entrada=nombre_archivo[0],path_salida=zip_path+"/"+str(sample_id))



cambiar_sexo_masc=[24,26,76,84,97,112]
for sample_id in cambiar_sexo_masc:
    json_dict, path=obtener_json(str(sample_id))
    json_dict=reemplazar_info(json_dict,'sexo','Masc')
    nombre_archivo= os.path.split(path)
    encriptar(json_dict,nombre_archivo[-1].split(".")[0],nombre_archivo[0])
    zipear(path_entrada=nombre_archivo[0],path_salida=zip_path+"/"+str(sample_id))





