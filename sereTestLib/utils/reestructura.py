from sereTestLib.parameters import *
from sereTestLib.webservice.parameters_ws import *
import os
import shutil
import wandb
import pymssql



def crear_carpetas(paths_nuevos):
    for path in paths_nuevos:
        os.makedirs(path, exist_ok = True)

def rename_dirs(paths_viejos, paths_nuevos, files_viejos, files_nuevos):
    for i in range (len(paths_viejos)):

        alldirs = os.listdir(paths_viejos[i])
        for d in alldirs:
            src_path = os.path.join(paths_viejos[i], d)
            shutil.move(src_path, paths_nuevos[i])


            #shutil.move(paths_viejos[i], paths_nuevos[i])
    for i in range(len(files_viejos)):
        allfiles = os.listdir(files_viejos[i])
 
        # iterate on all files to move them to destination folder
        for f in allfiles:
            src_path = os.path.join(files_viejos[i], f)
            dst_path = os.path.join(files_nuevos[i], f)
            shutil.move(src_path, dst_path)

def borrar_dirs(paths_borrar):
    for path in paths_borrar:
        shutil.rmtree(path, ignore_errors=True, onerror=None)
def cargar_modelos():
    run=wandb.init(project="SereTest-autoencoder",reinit=True,job_type="load ae")
    modelo_artifact=run.use_artifact(autoencoder_name+':latest')
    modelo_dir=modelo_artifact.download(model_path_ae)
    run.finish()

    run=wandb.init(project="SereTest-clasificador",reinit=True,job_type="load clf")
    modelo_artifact=run.use_artifact(clasificador_name+':latest')
    modelo_dir=modelo_artifact.download(model_path_clf)
    run.finish()

def cambiar_ruta_bd():
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    sql = "SELECT * FROM MUESTRAS "
    cursor.execute(sql)
    dicts=(cursor.fetchall())
    print(dicts)
    for dict in dicts: 
        path=dict["RUTAARCHIVOS"]
        print(path)
        cursor2 = conn.cursor(as_dict=True)
        sql = "UPDATE MUESTRAS SET RUTAARCHIVOS=%s WHERE NROMUESTRA=%s"
        cursor2.execute(sql, (path.replace("sereDataProd//rawData","sereDataProd1/Muestras"),dict["NROMUESTRA"]))
        conn.commit()
    cursor3 = conn.cursor(as_dict=True)
    sql = "SELECT * FROM MUESTRAS "
    cursor3.execute(sql)
    print(cursor3.fetchall())
    conn.close()
    conn = pymssql.connect(server=server, user=user,port=port, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    sql = "SELECT * FROM RESULTADOSMUESTRA"
    cursor.execute(sql)
    dicts=(cursor.fetchall())
    #print(dicts)
    for dict in dicts: 
        path=dict["REPORTE"]
        print(path)
        cursor2 = conn.cursor(as_dict=True)
        sql = "UPDATE RESULTADOSMUESTRA SET REPORTE=%s WHERE NROMUESTRA=%s"
        cursor2.execute(sql, (path.replace("sereDataProd/","sereDataProd1"),dict["NROMUESTRA"]))
        conn.commit()
    cursor3 = conn.cursor(as_dict=True)
    sql = "SELECT * FROM RESULTADOSMUESTRA "
    cursor3.execute(sql)
    print(cursor3.fetchall())
    conn.close()
def prod_2_prod1():
    conn = pymssql.connect(server=server, user=user,port=1434, password=password, database=database)
    cursor = conn.cursor(as_dict=True)
    sql = "SELECT * FROM PACIENTES "
    cursor.execute(sql)
    pacientes=cursor.fetchall()
    conn2 = pymssql.connect(server=server, user=user,port=1433, password=password, database=database)
    for paciente in pacientes:
        try:
            cursor2 = conn2.cursor(as_dict=True)
            sql = "INSERT INTO PACIENTES (IDENTIFICACION, NOMBRE, NOMBRE2, APELLIDO, APELLIDO2, SEXO, FECHANAC) VALUES (%s,%s,%s,%s,%s,%s,%s) "
            cursor2.execute(sql,(paciente["IDENTIFICACION"],paciente["NOMBRE"],paciente["NOMBRE2"],paciente["APELLIDO"],paciente["APELLIDO2"],paciente["SEXO"],paciente["FECHANAC"]))
            conn2.commit()
        except:
            print(paciente)
    cursor = conn.cursor(as_dict=True)
    sql = "SELECT * FROM MUESTRAS "
    cursor.execute(sql)
    muestras=cursor.fetchall()
    conn2 = pymssql.connect(server=server, user=user,port=1433, password=password, database=database)
    for muestra in muestras:
        cursor2 = conn2.cursor(as_dict=True)
        sql = "INSERT INTO MUESTRAS (NROMUESTRA,FECHAMUESTRA,PACIENTE,MEDICO,SENSOR,INTERVENCIONTERAPEUTICA,RUTAARCHIVOS,INSTITUCION,LUGAR) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
        cursor2.execute(sql,(str(muestra["NROMUESTRA"]),muestra["FECHAMUESTRA"],muestra["PACIENTE"],muestra["MEDICO"],muestra["SENSOR"],muestra["INTERVENCIONTERAPEUTICA"],muestra["RUTAARCHIVOS"],muestra["INSTITUCION"],muestra["LUGAR"]))
        conn2.commit()
    cursor = conn.cursor(as_dict=True)
    sql = "SELECT * FROM RESULTADOSMUESTRA "
    cursor.execute(sql)
    resultados=cursor.fetchall()
    conn2 = pymssql.connect(server=server, user=user,port=1433, password=password, database=database)
    for resultado in resultados:
        cursor2 = conn2.cursor(as_dict=True)
        sql = "INSERT INTO RESULTADOSMUESTRA (NROMUESTRA,MEDICO,PACIENTE, REPORTE, FEEDBACK, RESULTADOJSON) VALUES (%s,%s,%s,%s,%s,%s) "
        cursor2.execute(sql,(resultado["NROMUESTRA"],resultado["MEDICO"],resultado["PACIENTE"],resultado["REPORTE"],resultado["FEEDBACK"],resultado["RESULTADOJSON"]))
        conn2.commit()
    cursor = conn2.cursor(as_dict=True)
    sql = "SELECT * FROM RESULTADOSMUESTRA"
    cursor.execute(sql)
    muestras=cursor.fetchall()

static_path_viejo=home_path + "/Dropbox/PROJECTS/SL2205/sereDataProd/"
files_viejos=[
    static_path_viejo+"stClassifierData/clasificadores",
    static_path_viejo+"rawData/zips",
    static_path_viejo+"attachments/"
]
paths_viejos=[
    static_path_viejo+"ndClassifierData/wavelet_cmor/",
    static_path_viejo+"ndClassifierData/wavelet_Standarizados",
    static_path_viejo+"preprocessData/dataset",
    static_path_viejo+"preprocessData/overlaps",
    static_path_viejo+"preprocessData/rename_overlaps/",
    static_path_viejo+"stClassifierData/features/",
    #static_path_viejo+"stClassifierData/clasificadores",
    static_path_viejo+"rawData/raw_2_process/",
    #static_path_viejo+"rawData/zips",
    static_path_viejo+"rawData/unzip/",
    #static_path_viejo+"attachments/"
    ]
files_nuevos=[
    dir_modelos_act,
    zip_path,
    attachment_path
]

paths_nuevos=[
    directorio_scalogramas_test,
    dir_preprocessed_data_test,
    dir_out_fixed_axes ,
    dir_out_ov_split_test,
    dir_out_ov_split_rename_test,
    dir_out_new_char_test,
    #dir_modelos_act,
    dir_in_original_sample,
    #zip_path,
    to_unzip_path,
    #attachment_path,
    cal_path, 
    model_path_ae,
    model_path_clf,
]

paths_borrar=[
    static_path_viejo+"dense_base_classificator",
    static_path_viejo+"ndClassifierData",
    static_path_viejo+"preprocessData",
    static_path_viejo+"stClassifierData",
    static_path_viejo+"rawData",
]



if __name__== '__main__':

    #crear_carpetas(paths_nuevos)
    rename_dirs(paths_viejos, paths_nuevos, files_viejos, files_nuevos)
    #borrar_dirs(paths_borrar)
    #cargar_modelos()
    #prod_2_prod1()
    #cambiar_ruta_bd()

