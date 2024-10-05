from sereTestLib.parameters import *
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.preprocesamiento_reduccion_correccion_ejes import preprocesamiento_reduccion_correccion_ejes
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.split_and_save_samples_overlap import split_and_save_samples_overlap
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.create_segments_scalograms import create_segments_scalograms
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.create_or_augment_scalograms import create_or_augment_scalograms
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.characteristics_file_gen import characteristics_file_gen
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.clasificar_estado import clasificar_estado
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.rename_segments import rename_segments
import wandb

def Preprocesamiento_para_inferir(val_test, upload_wandb=False):
    """
    Preprocessing only to infer the samples.
    
    Unlike other preprocessing functions, this one avoids compressing files to optimize execution time. 
    Instead, the scalograms matrix is passed as an argument from one function to another

    Args:
        val_test (list): 
            list of validation and test samples
        upload_wandb (bool, optional):
            Defaults to False.
    """
    if upload_wandb:
        run = wandb.init(project=project_wandb,job_type="load dataset")
        print("PASO 1 -  preprocesar datos")
        
    for sample_id in val_test:
        
        if preprocesamiento==1:
            if long_sample:
                directorio_muestra = "/L%s/" % (sample_id)
            else:
                directorio_muestra = "/S%s/" % (sample_id)
            
            preprocesamiento_reduccion_correccion_ejes(dir_in_original_sample + directorio_muestra, dir_out_fixed_axes+directorio_muestra, long_sample, actividades=act_ae)
            split_and_save_samples_overlap(dir_out_fixed_axes+directorio_muestra, dir_out_ov_split_test+directorio_muestra,time_frame, time_overlap, sampling_period, step=3)
            escalogramas_por_segmento = create_segments_scalograms(dir_out_ov_split_test+directorio_muestra, directorio_scalogramas='', escalas=escalas_wavelets, dt=sampling_period, save_files=False)
            create_or_augment_scalograms(scalogram_path='', output_path=dir_preprocessed_data_test+directorio_muestra, static_window_secs=static_window_secs, movil_window_secs=window_secs, overlap_secs=3, fs=fs, escalado=escalado, escalogramas=escalogramas_por_segmento)


        elif preprocesamiento==2:
            if long_sample:
                directorio_muestra = "/L%s/" % (sample_id)
            else:
                directorio_muestra = "/S%s/" % (sample_id)

            preprocesamiento_reduccion_correccion_ejes(dir_in_original_sample + directorio_muestra, dir_out_fixed_axes+directorio_muestra, long_sample)
            split_and_save_samples_overlap(dir_out_fixed_axes+directorio_muestra, dir_out_ov_split_test+directorio_muestra, time_frame, time_overlap, sampling_period, step=3)
            characteristics_file_gen(dir_out_ov_split_test+directorio_muestra,dir_out_new_char_test+directorio_muestra, filter_window=5)
            clasificar_estado(sample_id, long_sample,dir_out_new_char_test)
            rename_segments(sample_id, long_sample= long_sample,dir=dir_out_ov_split_test,dir_rename=dir_out_ov_split_rename_test,dir_out_new_char=dir_out_new_char_test)
            escalogramas_por_segmento = create_segments_scalograms(dir_out_ov_split_rename_test+directorio_muestra, directorio_scalogramas='', escalas=escalas_wavelets, dt=sampling_period, actividades=act_ae, save_files=False)
            create_or_augment_scalograms(scalogram_path='',  output_path=dir_preprocessed_data_test+directorio_muestra, static_window_secs=static_window_secs, movil_window_secs=window_secs, overlap_secs=3, fs=fs, escalado=escalado, escalogramas=escalogramas_por_segmento)

    if upload_wandb:
        run = wandb.init(project=project_wandb)
        my_data = wandb.Artifact(extra+str(preprocesamiento), type="raw_data")
        my_data.add_reference(data_dir, max_objects=90000)
        run.log_artifact(my_data)
        run.finish()
        
if __name__== '__main__':
    print(static_path)
    sample_id = 3004
    val_test=[sample_id]
    Preprocesamiento_para_inferir(val_test, upload_wandb=False)

    
