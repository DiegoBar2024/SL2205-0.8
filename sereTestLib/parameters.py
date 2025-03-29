from enum import Enum
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split

largo_vector=10
secuencia=(largo_vector)*1400

home_path = str(Path.home())
static_path = home_path + "/Dropbox/PROJECTS/SL2205/sereData/"
# ################################################
# #                   WEBSERVICE
# ################################################
modo_webservice="modo_serv"
if modo_webservice=="modo_testing":
    static_path = home_path + "/Dropbox/PROJECTS/SL2205/sereDataTesting/"
elif modo_webservice=="modo_prod":
    static_path = home_path + "/Dropbox/PROJECTS/SL2205/sereDataProd/"

##########################################################
#  PASO 0: PARAMETROS GENERALES
##########################################################
home_path = str(Path.home())
path_dataset=static_path+ "Dataset/"
#path_dataset_wandb= str(Path.home())+  "/Dropbox/PROJECTS/SL2205/Dataset/"
data_dir="file://"+path_dataset
debug = True #TODO: recibir como parámetro del sistema
name_size = 6
dir_etiquetas=  static_path + "Etiquetas/"
date = datetime.now()
date_day = date.strftime('%Y%m%d')
date = date.strftime('%Y%m%d_%H%M')
extra = "StandarAum" ### Extra en el nombre del archivo: "Augmented", "StandarAum", "AumGiro", "AumGiroStd"
clasificador = 'perceptron'#lda perceptron svm, NN,RF, AE_MLP hierarchical o kmeans o Transfer
loss_name= "mse"             #mse o binary_crosentropy ssim_loss
project_wandb="SereTest-data" #WandB - Proyecto donde se descargan los datos

##########################################################
#  PASO 1: PARAMETROS PREPROCESAMIENTO INICIAL
##########################################################
# dir_in_original_sample = path_dataset + "raw_2_process/"
# dir_out_fixed_axes    = path_dataset + "dataset/"
global accel_type

accel_type = "Accel_" + "LN" # Puede ser LN (low noise) o WR (wide range)

long_sample = False  ## Indica si se esta trabajando con muestras largas,
                    # o con muestras con etiqueta de actividad (1 Sentado, 2 Parado, etc)
preprocesamiento=1 #=1 o =2 si se quieren inferir las etiquetas de actividades
## path_log_preprocesamiento= static_path + "/Logs/preprocesamiento/"+date+ ".log"
check_only_if_preprocessed_data_is_empty = False # A la hora de chequear si alguna carpeta quedó vacía en
                                                # el preprocesado de las muestras esta bandera indica
                                                # si se quiere chequear que estén no vacías solo la carpeta
                                                # final (preprocessed_data) o todas las carpetas intermedias
wavelet_library='pywt'          # Librería a utilizar para realizar las transformadas de Wavelet
                                # Puede ser 'pywt', 'scipy' o 'ssqueezepy'
##########################################################
#  PASO 2: PARAMETROS SEGMENTACION CON OVERLAP
##########################################################
#TODO: Pensar si es viable hacer un slding window sobre la muestra.
time_frame = 3   # segundos por ventana
time_overlap = 2 # tiempo de solapamiento ----- Para eliminar del scalograma!!!! Esto funciona porque son 3s no funciona para 5 por ejemplo.
sampling_period =0.005 # 1/fs, estamos trabajando con fd=200Hz Esto influye en el tamaño de las ventanas (ergo en los scalogramas) y/o en los períodos analizados
dir_out_ov_split = path_dataset + "overlaps/"
# dir_out_ov_split_train=dir_out_ov_split+ "train/"
# dir_out_ov_split_test=dir_out_ov_split+ "test/"



#######################################################
# PASO 3: Generación de características
######################################################
dir_out_new_char = path_dataset + "features/"
# dir_out_new_char_train = dir_out_new_char + "train/"
# dir_out_new_char_test = dir_out_new_char + "test/"


#######################################################
# PASO 4: Clasificar estados de actividad
#######################################################

#TODO: Revisar si no tiene mas sentido reescribir el original
dir_modelos_act=static_path + 'Modelos/' + 'Actividad/'
# filenameAI =  dir_modelos_act+"regLin_Activo_Inactivo.sav"   #ruta a SubClasificador ActivosVSInactivos
# filenameSP = dir_modelos_act+"regLin_sentadoParado.sav"     #ruta a SubClasificador sentadoVSparado
# filenameCE =dir_modelos_act+ "regLin_caminandoEscalera.sav" #ruta a SubClasificador caminataVescalera

dir_out_ov_split_rename = path_dataset + "rename_overlaps/"
# dir_out_ov_split_rename_train = dir_out_ov_split_rename + "train/"
# dir_out_ov_split_rename_test = dir_out_ov_split_rename + "test/"


#-------------DE ACA PARA ABAJO SOLO ACTIVIDADES CAMINATA Y ESCALERA----------------------
#######################################################
# PASO 5: Generar scalogramas (si corresponde)
#######################################################
# USO0
dict_actividades = {'Sentado':'1','Parado':'2','Caminando':'3','Escalera':'4'}
dir_ae=path_dataset + 'ae_intermediate/'
dir_ae_train=dir_ae + "train/"
dir_ae_test=dir_ae + "test/"
escalas_wavelets = [8,136]

clase_waveletes = 'cmorlet'
directorio_scalogramas = path_dataset + 'wavelet_cmor/'
# directorio_scalogramas_train = directorio_scalogramas + 'train/'
# directorio_scalogramas_test = directorio_scalogramas + 'test/'
dir_preprocessed_data=path_dataset+'preprocessed_data/'
# dir_preprocessed_data_train=dir_preprocessed_data + 'train/'
# dir_preprocessed_data_test=dir_preprocessed_data + 'test/'

#dir_preprocessed_data_wandb=path_dataset_wandb+'preprocessed_data/'
#dir_preprocessed_data_train_wandb=dir_preprocessed_data_wandb + 'train/'
#dir_preprocessed_data_test_wandb=dir_preprocessed_data_wandb + 'test/'

fs = 200

# Para el ploteo de los scalogramas
cant_muestras = 5
plot_group = date+"_"+(extra if extra != '' else 'normales')

#######################################################
# PASO 6: Clasificar estabilidad por actividad
#######################################################
layer_name = 'Dense_encoder'
# Parametros del modelo de AutoEncoder
height = 128
width = 600
depth = 6
batch_size = 1
latent_dimension = 256

# Actividades y directorios
modo_ae = "dense_base_classificator" # dense_base_classificator o anom_detector
actividades_dinamicas = ['Caminando','Escalera']#Todas las actividades dinamicas contempladas
act_ae= ['Caminando'] # actividades a considerar en la clasificacion
act_clf= ['Caminando']
act_inact=["Sentado"]
#TODO: hasta que no lo gestionemos con MLOPS
ae_model_version = "0.1.1"
clasificador_model_version = "0.1.1"
git_version_commit = '07eff1c'
autoencoder_dir = modo_ae+"/"



# Cant of secs per segment, in the common cmorlet scalograms
static_window_secs = 3
# Cant of secs in the new segments
window_secs = 3
# Cant of secs of overlaping
overlap_secs= 3
autonormalizar = False
step=3

# inDim = (height, width, depth)
patience=5
num_epochs = 1
base_learning_rate = 0.0001

####################################################
#             Parametros según extra
####################################################

giroz=[0,0]
girox=[0,0]
step_val=3
if extra == "Augmented":
    overlap_secs= 0.5
    escalado="normal"
elif extra == "AumGiro":
    escalado="normal"
    giroz=[1,10]
    girox=[1,10]
    step=0.5
elif extra == "AumGiroStd":
    escalado="standarizado"
    giroz=[1,10]
    girox=[1,10]
    step=0.5

## Esta es la opción que está puesta por defecto
elif extra == 'StandarAum':

    ## Especifico cantidad de segundos de solapamiento
    overlap_secs = 0.5

    ## Especifico tipo de escalado
    escalado = "standarizado"

elif extra=="escalado":
    escalado="normal"
    dir_preprocessed_data_train=dir_preprocessed_data + 'train_escalados/'
elif extra=="escalado_temporal":
    escalado="normal"
    dir_preprocessed_data_train=dir_preprocessed_data + 'train_escalados/'
    time_overlap = 2
    step=7
    step_val=7


if extra == "5segs":
    window_secs = 5
    overlap_secs= 5
    width=1000

################################################
#             Modelos
################################################
autoencoder_name = extra+'_'+act_ae[0]+'_'+str(num_epochs)+'_'+loss_name+'_'+modo_ae
clasificador_name= clasificador+'_'+extra


model_path_ae = static_path + 'Modelos/' + 'autoencoder/'+ autoencoder_name + '/'
model_path_clf =static_path + 'Modelos/' +'clasificador/'+ clasificador_name + '/'

################################################
#                   TRAIN
################################################

# Resultados
results_path_day =  static_path +'results/'+ date_day + '/'
results_path =  results_path_day + date +'_'+ modo_ae + '/'
date_train = '20220824'
results_train_path_day =  static_path +'results/'+ date_train + '/'
results_train_path =  results_path_day + date +'_'+ modo_ae + '/'


class results:
    def __init__(self):
        """
        Class to store results for a sample precessing
        """
        # Sample identyfier
        self.sample_id:str = ''
        self.doctor:str = '' # Es esperado por el backend
        self.patient:str = '' # Es esperado por el backend
        # Sample distribution information acording to SERE Index for arch activity sample_mean = {"CaminandoEscalera":2.3,"Caminando":3.3,"Escalera":2.1}
        self.sample_mean:dict = {}
        self.sample_std :dict = {}
        self.sample_unstable_mean:dict = {}
        self.sample_unstable_std :dict = {}
        self.sample_stable_mean:dict = {}
        self.sample_stable_std :dict = {}
        # Sample stable and unstable percentages acording to SERE Index for arch activity. I.e. stable_percentage = {"CaminandoEscalera":81,"Caminando":70,"Escalera":84}
        self.stable_percentage:dict = {}
        self.unstable_percentage:dict = {}
        self.sere_index:int = 0
        # Time identified that the patient do an activitu during the sample acquisition in format hh:mm:ss. I.e. activities_time = {"Caminando":"00:02:12","Escalera":"01:20:09","Unidentified":"00:05:15"}
        self.activities_time_secs:dict = {}
        self.activities_time_format:dict = {}
        self.images_folder_path:str = ''
        self.long_sample:bool = False
        self.activities:list[str] = []
        """
        Information model dependent, not sample dependent.
        Useful for get model's summary information
        """
        self.models_info:dict= {}
        self.lda_model_version:str  = clasificador_model_version
        self.ae_model_version:str   = ae_model_version
        self.git_version_commit:str = git_version_commit
        self.train_date:str         = date_train
        self.ae_mode:str            = modo_ae
        self.extra:str              = extra
        self.ae_name:str            = ""
        self.clf_name: str          = ""
    def __str__(self,act) -> str:
        stable_percent_str =  "Stable segments: " + str(self.stable_percentage["".join(act)]) + "%\n"
        unstable_percent_str = "Unstable segments: " + str(self.unstable_percentage["".join(act)]) + "%\n"
        act_time_str = "Time that the user perform the activity " + "".join(act) + ": " + self.activities_time_format[''.join(act)] + "%\n"
        return stable_percent_str + unstable_percent_str + act_time_str

url_response = 'http://192.168.89.245:8080/sereServer/sererest/muestras' #TODO: conseguir una ip fija o bien migrarlo al mismo servidor

## ACTUALIZO LAS RUTAS PARA LOS FICHEROS EN MI COMPUTADORA
dir_in_original_sample = 'C:/Yo/Tesis/sereData/sereData/Dataset/raw_2_process'
dir_out_fixed_axes = 'C:/Yo/Tesis/sereData/sereData/Dataset/dataset'
dir_out_ov_split_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/overlaps/train'
dir_out_ov_split_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/overlaps/test'
dir_out_new_char_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/features/train'
dir_out_new_char_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/features/test'
dir_out_ov_split_rename_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/rename_overlaps/train'
dir_out_ov_split_rename_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/rename_overlaps/train'
directorio_scalogramas_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/wavelet_cmor/train'
directorio_scalogramas_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/wavelet_cmor/test'
dir_preprocessed_data_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/preprocessed_data/train'
dir_preprocessed_data_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/preprocessed_data/test'
path_log_preprocesamiento= "C:/Yo/Tesis/sereData/sereData/Logs/preprocesamiento"+ date + ".log"
filenameAI = 'C:/Yo/Tesis/sereData/sereData/Modelos/Actividad/regLin_Activo_Inactivo.sav'
filenameSP = 'C:/Yo/Tesis/sereData/sereData/Modelos/Actividad/regLin_sentadoParado.sav'
filenameCE = 'C:/Yo/Tesis/sereData/sereData/Modelos/Actividad/regLin_caminandoEscalera.sav'
dir_etiquetas=  'C:/Yo/Tesis/sereData/sereData/Etiquetas/'

## Rutas en donde van a estar los escalogramas finales que van a entrar a la red neuronal
dir_escalogramas_nuevo_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo/train'
dir_escalogramas_nuevo_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo/test'

## Rutas en donde van a estar los escalogramas individuales por cada canal
dir_escalogramas_nuevo_ind_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_ind_nuevo/train'
dir_escalogramas_nuevo_ind_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_ind_nuevo/test'

## DIMENSIÓN TEMPORAL A LA QUE SE VA A REMUESTREAR
## Especifico la cantidad de pasos que quiero representar en mis escalogramas
cantidad_pasos = 4

## Especifico el tiempo máximo de paso en segundos que voy a usar para hacer el remuestreo del eje temporal
tiempo_maximo_paso = 1

## Especifico la dimensión temporal objetivo a la que voy a interpolar (remuestreo)
## Ésta cantidad está calculada con parámetros que NO DEPENDEN DEL PACIENTE, lo cual es lo que se pretende
dimension_temporal = cantidad_pasos * int(tiempo_maximo_paso * 200)

## Especifico las dimensiones temporales del tensor tridimensional de entrada al autoencoder
inDim = (6, 128, dimension_temporal)