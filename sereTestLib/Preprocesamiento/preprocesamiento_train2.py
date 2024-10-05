####### IMPORTO LA CARPETA DONDE ESTÁN LOS PARÁMETROS ########
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Preprocesamiento/funcionesPreprocesamiento')
##############################################################

from preprocesamiento_reduccion_correccion_ejes import preprocesamiento_reduccion_correccion_ejes
from split_and_save_samples_overlap import split_and_save_samples_overlap
from create_segments_scalograms import create_segments_scalograms
from create_or_augment_scalograms import create_or_augment_scalograms
from characteristics_file_gen import characteristics_file_gen
from clasificar_estado import clasificar_estado
from rename_segments import rename_segments

from parameters import *

# dir_in_original_sample = 'C:/Yo/Tesis/sereData/sereData/Dataset/raw_2_process'
# dir_out_fixed_axes = 'C:/Yo/Tesis/sereData/sereData/Dataset/dataset'
# dir_out_ov_split_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/overlaps/train'
# dir_out_new_char_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/features/train'
# dir_out_ov_split_rename_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/rename_overlaps/train'
# directorio_scalogramas_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/wavelet_cmor/train'
# dir_preprocessed_data_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/preprocessed_data/train_escalados'

## El parámetro <<sample_id>> va a ser el ID del paciente cuyos datos voy a usar para hacer el preprocesamiento. Para la máquina va a tratarse de un número entero
## Se van a preprocesar los datos dado el ID de muestra
def preprocesamiento_train2(sample_id):
    """
    Preprocess the data given the sample id.

    Parameters
    ----------
    sample_id: int

    """
    ## Imprimo el ID de muestra
    print(sample_id)

    ## Por defecto el parámetro <<long_sample>> está seteado en False debido que en principio se está trabajando con muestras CORTAS
    ## En caso de que la muestra sea larga, hago la concatenación con el caracter 'L'
    if long_sample:

        ## Por ejemplo, si <<sample_id>> = 215 --> <<directorio_muestra>> = '/L215/'
        directorio_muestra = "/L%s/" % (sample_id)

    ## En caso de que la muestra sea corta, hago la concatenación con el carácter 'S'
    else:

        ## Por ejemplo, si <<sample_id>> = 215 --> <<directorio_muestra>> = '/S215/'
        directorio_muestra = "/S%s/" % (sample_id)

    ## FUNCIÓN QUE HACE EL PREPROCESAMIENTO DE REDUCCIÓN Y CORRECCIÓN DE EJES
    ## El parámetro <<dir_in_original_sample>> va a contener la ruta al directorio <<raw_2_process>> con los datos CRUDOS
    ## Para el caso de MI COMPUTADORA: <<dir_in_original_sample>> = 'C:/Yo/Tesis/sereData/sereData/Dataset/raw_2_process'
    ## El parámetro <<dir_out_fixed_axes>> me va a contener la ruta al directorio <<dataset>> en donde se guardarán los datos como resultado del preprocesamiento
    ## Para el caso de MI COMPUTADORA: <<dir_out_fixed_axes>> = 'C:/Yo/Tesis/sereData/sereData/Dataset/dataset'
    ## El parámetro <<directorio_muestra>> me va a contener la carpeta correspondiente al paciente analizado y es el que se arma en las líneas de arriba
    ## Por ejemplo, si se está analizando una muestra corta (<<long_sample>> = False) del paciente de ID 113 (<<sample_ID>> = 113)
    preprocesamiento_reduccion_correccion_ejes(dir_in_original_sample + directorio_muestra, dir_out_fixed_axes + directorio_muestra, long_sample)

    ## El parámetro <<dir_out_fixed_axes>> me va a contener la ruta al directorio <<dataset>> en donde están los datos preprocesados para el paciente correspondiente
    ## Para el caso de MI COMPUTADORA: <<dir_out_fixed_axes>> = 'C:/Yo/Tesis/sereData/sereData/Dataset/dataset'
    ## El parámetro <<directorio_muestra>> me va a contener la carpeta correspondiente al paciente analizado y es el que se arma en las líneas de arriba
    ## Por ejemplo, si se está analizando una muestra corta (<<long_sample>> = False) del paciente de ID 215 (<<sample_ID>> = 215)
    ## El parámetro <<dir_out_ov_split_train>> va a ser la ruta donde voy a guardar la salida de las muestras solapadas correspondientes al paciente que se pasó como entrada.
    ## Para el caso de MI COMPUTADORA <<dir_out_ov_split_train>> = 'C:/Yo/Tesis/sereData/sereData/Dataset/overlaps/train'
    ## El parámetro <<time_frame>> me va a contener la cantidad de segundos de ventana. Usando el valor dado en <<parameters.py>> se tiene <<time_frame>> = 3
    ## El parámetro <<time_overlap>> me va a contener la cantidad de segundos de solapamiento. Usando el valor dado en <<parameters.py>> se tiene <<time_overlap>> = 2
    ## El parámetro <<sampling_period>> me va a contener el período de muestreo de la señal de entrada. Usando el valor dado en <<parameters.py>> se tiene <<sampling_period>> = 0.05
    ## El parámetro <<step>> me va a contener el tamaño de paso. Usando el valor dado en <<parameters.py>> se tiene <<step>> = 3
    split_and_save_samples_overlap(dir_out_fixed_axes + directorio_muestra, dir_out_ov_split_train + directorio_muestra, time_frame, time_overlap, sampling_period, step = step)

    ## El parámetro <<dir_out_ov_split_train>> va a ser la ruta donde tengo los segmentos del paciente preprocesados luego de hacer el solapamiento
    ## Para el caso de MI COMPUTADORA <<dir_out_ov_split_train>> = 'C:/Yo/Tesis/sereData/sereData/Dataset/overlaps/train'
    ## El parámetro <<directorio_muestra>> me va a contener la carpeta correspondiente al paciente analizado y es el que se arma en las líneas de arriba
    ## Por ejemplo, si se está analizando una muestra corta (<<long_sample>> = False) del paciente de ID 215 (<<sample_ID>> = 215)
    ## El parametro <<dir_out_new_char_train>> va a ser la ruta donde voy a guardar las muestras de salida de <<characteristics_file_gen>>
    ## Para el caso de MI COMPUTADORA <<dir_out_new_char_train>> = 'C:/Yo/Tesis/sereData/sereData/Dataset/features/train'
    ## El parámetro <<filter_window>> me va a dar el tamaño de la ventana utilizada para hacer el preprocesado. Por defecto tengo <<filter_window>> = 5
    characteristics_file_gen(dir_out_ov_split_train + directorio_muestra, dir_out_new_char_train + directorio_muestra, filter_window = 5)

    ## El parámetro <<sample_id>> va a tener el ID del paciente correspondiente. Por ejemplo, si estoy trabajando con el paciente de ID 215 tengo <<sampleID>> = 215
    ## El parámetro <<long_sample>> me dice si la muestra es larga o corta. Por defecto mirando el valor en <<parameters.py>> se tiene que <<long_sample>> = False
    ## El parámetro <<dir_out_new_char_train>> tiene la ruta donde tengo las muestras resultantes de hacer el procesamiento en <<characteristics_file_gen>>
    clasificar_estado(sample_id, long_sample, dir_out_new_char_train)

    ## El parámetro <<sample_id>> va a tener el ID del paciente correspondiente. Por ejemplo, si estoy trabajando con el paciente de ID 215 tengo <<sampleID>> = 215
    ## El parámetro <<long_sample>> me dice si la muestra es larga o corta. Por defecto mirando el valor en <<parameters.py>> se tiene que <<long_sample>> = False
    ## El parámetro <<dir_out_new_char_train>> tiene la ruta donde tengo las muestras resultantes de hacer el procesamiento en <<characteristics_file_gen>>
    ## Para el caso de MI COMPUTADORA <<dir_out_new_char_train>> = 'C:/Yo/Tesis/sereData/sereData/Dataset/features/train'
    ## El parámetro <<dir_out_ov_split_rename_train>> tiene la ruta donde quiero que se me guarden las muestras resultantes luego de hacer el renombrado
    ## Para el caso de MI COMPUTADORA <<dir_out_ov_split_rename_train>> = 'C:/Yo/Tesis/sereData/sereData/Dataset/rename_overlaps/train'
    rename_segments(sample_id,dir = dir_out_ov_split_train, dir_rename = dir_out_ov_split_rename_train, dir_out_new_char = dir_out_new_char_train, long_sample = False)

    ## El parametro <<dir_out_ov_split_rename_train>> va a tener la ruta a donde están los datos para todos los pacientes luego de hacer el procesado previo
    ## Para el caso de MI COMPUTADORA <<dir_out_ov_split_rename_train>> = 'C:/Yo/Tesis/sereData/sereData/Dataset/rename_overlaps/train'
    ## El parámetro <<directorio_muestra>> me va a contener la carpeta correspondiente al paciente analizado y es el que se arma en las líneas de arriba
    ## Por ejemplo, si se está analizando una muestra corta (<<long_sample>> = False) del paciente de ID 215 (<<sample_ID>> = 215)
    ## El parámetro <<directorio_scalogramas_train>> va a tener la ruta donde voy a guardar los escalogramas a la salida correspondientes a los segmentos procesados.
    ## Para el caso de MI COMPUTADORA <<directorio_scalogramas_train>> = 'C:\Yo\Tesis\sereData\sereData\Dataset\wavelet_cmor\train'
    ## El parámetro <<escalas>> me va a contener el rango de escalas que voy a usar para hacer el cálculo de las transformadas de wavelet. Se importa del fichero <<parametros.py>> y tiene como valor <<escalas_wavelets>> = [8, 136]
    ## El parámetro <<dt>> es el período de muestreo de mis segmentos. Se importa del fichero <<parameters.py>> y tiene como valor <<dt>> = 0.05
    ## Los parámetros <<girox>> y <<giroz>> me dan el valor de los giros. Se importan del fichero <<parameters.py>> y tienen como valor <<girox>> = [0,0] y <<giroz>> = [0,0] por defecto
    ## El parámetro <<act_ae>> es una lista que contiene las actividades que yo quiero procesar. Se importa del fichero <<parameters.py>> y tiene como valor <<act_ae>> = ['Caminando']
    create_segments_scalograms(dir_out_ov_split_rename_train + directorio_muestra, directorio_scalogramas_train + directorio_muestra,  escalas = escalas_wavelets, dt = sampling_period, girox = girox, giroz = giroz, actividades = act_ae)

    ## El parámetro <<directorio_scalogramas_train>> va a tener la ruta donde están los escalogramas que voy a usar para procesar
    ## Para el caso de MI COMPUTADORA <<directorio_scalogramas_train>> = 'C:\Yo\Tesis\sereData\sereData\Dataset\wavelet_cmor\train'
    ## El parámetro <<dir_preprocessed_data_train>> me va a almacenar los escalogramas de salida luego del preprocesado
    ## Para el caso de MI COMPUTADORA <<dir_preprocessed_data_train>> = 'C:\Yo\Tesis\sereData\sereData\Dataset\preprocessed_data\train'
    ## El parámetro <<directorio_muestra>> me va a contener la carpeta correspondiente al paciente analizado y es el que se arma en las líneas de arriba
    ## Por ejemplo, si se está analizando una muestra corta (<<long_sample>> = False) del paciente de ID 215 (<<sample_ID>> = 215)
    ## El parámetro <<act_ae>> es una lista que contiene las actividades que yo quiero procesar. Se importa del fichero <<parameters.py>> y tiene como valor <<act_ae>> = ['Caminando']
    create_or_augment_scalograms(directorio_scalogramas_train + directorio_muestra, dir_preprocessed_data_train + directorio_muestra, actividades = act_ae, static_window_secs = static_window_secs, movil_window_secs = window_secs, overlap_secs = overlap_secs, fs = fs, escalado = escalado)

preprocesamiento_train2(106)