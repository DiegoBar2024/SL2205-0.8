import sereTestLib.parameters as p
import os

def tiene_carpeta_vacia(sample_id, train_list, val_test_list, only_preprocessed_data_folder=False):
    """ Detect if there are empty folders in the preprocessing of a sample

    Args:
        sample_id (int): sample_id
        train_list (list): training samples list
        val_test_list (list): test samples list
        only_preprocessed_data_folder (bool): true if you want to check only the 'preprocessed_data'' folder 
                                                and False if you want to check all the preprocessing folders 
        
    Returns:
        bool: true if folders are missing and false if not
    """
    muestra = {'carpetas_vacias': [],
            'carpetas_vacias_train': [],
            'carpetas_vacias_test': []}
        
    if p.long_sample:
        directorio_muestra = "/L%s/" % (sample_id)
    else:
        directorio_muestra = "/S%s/" % (sample_id)
    
    
    # check 'preprocessed_data' folder
    if sample_id in train_list:
        path_preprocessed_data = p.dir_preprocessed_data_train + directorio_muestra
        if not os.path.exists(path_preprocessed_data) or (os.path.exists(path_preprocessed_data) and len(os.listdir(path_preprocessed_data)) == 0):
            muestra['carpetas_vacias_train'].append('preprocessed_data')       
    if sample_id in val_test_list:
        path_preprocessed_data = p.dir_preprocessed_data_test + directorio_muestra
        if not os.path.exists(path_preprocessed_data) or (os.path.exists(path_preprocessed_data) and len(os.listdir(path_preprocessed_data)) == 0):
            muestra['carpetas_vacias_test'].append('preprocessed_data')
    
    
    if not only_preprocessed_data_folder:
        # check folders from 'raw_2_process' to 'wavelet_cmor'
        
        path_raw_2_process = p.dir_in_original_sample + directorio_muestra
        if not os.path.exists(path_raw_2_process) or (os.path.exists(path_raw_2_process) and len(os.listdir(path_raw_2_process)) == 0):
            muestra['carpetas_vacias'].append('raw_2_process')
            
        path_dataset = p.dir_out_fixed_axes + directorio_muestra
        if not os.path.exists(path_dataset) or (os.path.exists(path_dataset) and len(os.listdir(path_dataset)) == 0):
            muestra['carpetas_vacias'].append('dataset')

        if sample_id in train_list:
            path_overlaps = p.dir_out_ov_split_train + directorio_muestra
            if not os.path.exists(path_overlaps) or (os.path.exists(path_overlaps) and len(os.listdir(path_overlaps)) == 0):
                muestra['carpetas_vacias_train'].append('overlaps')
            
            path_wavelet_cmor = p.directorio_scalogramas_train + directorio_muestra
            if not os.path.exists(path_wavelet_cmor) or (os.path.exists(path_wavelet_cmor) and len(os.listdir(path_wavelet_cmor)) == 0):
                muestra['carpetas_vacias_train'].append('wavelet_cmor')
            
            if p.preprocesamiento == 2: # Con detector de actividades
                path_features = p.dir_out_new_char_train + directorio_muestra
                if not os.path.exists(path_features) or (os.path.exists(path_features) and len(os.listdir(path_features)) == 0):
                    muestra['carpetas_vacias_train'].append('features')
                    
                path_rename_overlaps =  p.dir_out_ov_split_rename_train + directorio_muestra
                if not os.path.exists(path_rename_overlaps) or (os.path.exists(path_rename_overlaps) and len(os.listdir(path_rename_overlaps)) == 0):
                    muestra['carpetas_vacias_train'].append('rename_overlaps')
                
        if sample_id in val_test_list:
            path_overlaps = p.dir_out_ov_split_test + directorio_muestra
            if not os.path.exists(path_overlaps) or (os.path.exists(path_overlaps) and len(os.listdir(path_overlaps)) == 0):
                muestra['carpetas_vacias_test'].append('overlaps')
                
            path_wavelet_cmor = p.directorio_scalogramas_test + directorio_muestra      
            if not os.path.exists(path_wavelet_cmor) or (os.path.exists(path_wavelet_cmor) and len(os.listdir(path_wavelet_cmor)) == 0):
                muestra['carpetas_vacias_test'].append('wavelet_cmor')
            
            if p.preprocesamiento == 2: # Con detector de actividades
                path_features = p.dir_out_new_char_test + directorio_muestra
                if not os.path.exists(path_features) or (os.path.exists(path_features) and len(os.listdir(path_features)) == 0):
                    muestra['carpetas_vacias_test'].append('features')
                    
                path_rename_overlaps = p.dir_out_ov_split_rename_test + directorio_muestra
                if not os.path.exists(path_rename_overlaps) or (os.path.exists(path_rename_overlaps) and len(os.listdir(path_rename_overlaps)) == 0):
                    muestra['carpetas_vacias_test'].append('rename_overlaps')
                
            
    # sort folder lists
    order = ['raw_2_process', 'dataset', 'overlaps', 'features', 'rename_overlaps', 'wavelet_cmor', 'preprocessed_data']
    keydict = dict(zip(order, list(range(7))))
    muestra['carpetas_vacias'].sort(key=keydict.get)
    muestra['carpetas_vacias_train'].sort(key=keydict.get)
    muestra['carpetas_vacias_test'].sort(key=keydict.get)
    
    if ((len(muestra['carpetas_vacias']) != 0) or 
        (len(muestra['carpetas_vacias_train']) != 0) or 
        (len(muestra['carpetas_vacias_test']) != 0)):
        
        if len(muestra['carpetas_vacias']) != 0:       
            print('Muestra', str(sample_id), 'tiene carpeta vacía en', muestra['carpetas_vacias'])            
        if len(muestra['carpetas_vacias_train']) != 0:       
            print('Muestra', str(sample_id), 'de train tiene carpeta vacía en', muestra['carpetas_vacias_train'])
        if len(muestra['carpetas_vacias_test']) != 0:       
            print('Muestra', str(sample_id), 'de test tiene carpeta vacía en', muestra['carpetas_vacias_test'])
        return True
    
    else:
        return False
           
