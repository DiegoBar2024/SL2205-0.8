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

from parameters  import *

def preprocesamiento_val_test2(sample_id):
    """
    Preprocess the data given the sample id.

    Parameters
    ----------
    sample_id: int

    """
    print(sample_id)

    if long_sample:
        directorio_muestra = "/L%s/" % (sample_id)
    else:
        directorio_muestra = "/S%s/" % (sample_id)

    preprocesamiento_reduccion_correccion_ejes(dir_in_original_sample + directorio_muestra, dir_out_fixed_axes+directorio_muestra, long_sample)
    split_and_save_samples_overlap(dir_out_fixed_axes+directorio_muestra, dir_out_ov_split_test+directorio_muestra, time_frame, time_overlap, sampling_period, step=3)
    characteristics_file_gen(dir_out_ov_split_test+directorio_muestra,dir_out_new_char_test+directorio_muestra, filter_window=5)
    clasificar_estado(sample_id, long_sample,dir_out_new_char_test)
    rename_segments(sample_id, long_sample= long_sample,dir=dir_out_ov_split_test,dir_rename=dir_out_ov_split_rename_test,dir_out_new_char=dir_out_new_char_test)
    create_segments_scalograms(dir_out_ov_split_rename_test+directorio_muestra, directorio_scalogramas_test + directorio_muestra, escalas=escalas_wavelets, dt=sampling_period, actividades=act_ae)
    create_or_augment_scalograms(directorio_scalogramas_test+directorio_muestra,  dir_preprocessed_data_test+directorio_muestra, actividades=act_ae, static_window_secs=static_window_secs, movil_window_secs=window_secs, overlap_secs=3, fs=fs, escalado=escalado)


