#%%
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.preprocesamiento_reduccion_correccion_ejes import preprocesamiento_reduccion_correccion_ejes
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.split_and_save_samples_overlap import split_and_save_samples_overlap
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.create_segments_scalograms import create_segments_scalograms
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.create_or_augment_scalograms import create_or_augment_scalograms
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.characteristics_file_gen import characteristics_file_gen
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.clasificar_estado import clasificar_estado
from sereTestLib.Preprocesamiento.funcionesPreprocesamiento.rename_segments import rename_segments
from sereTestLib.utils.ingesta_etiquetas import ingesta_etiquetas
import pandas as pd
import joblib
from natsort.natsort import natsorted
import os,re
from sereTestLib.parameters  import *
from sklearn.metrics import accuracy_score

def estado_act(estado):
    act=0
    if int(estado)>2:
        act=1
    return act

def devolver_estados(sample_id):
    if long_sample:
        directorio_muestra = '/L' + str(sample_id) + "/"
    else:
        directorio_muestra = '/S' + str(sample_id) + "/"
    preprocesamiento_reduccion_correccion_ejes(dir_in_original_sample + directorio_muestra, dir_out_fixed_axes+directorio_muestra, long_sample)
    split_and_save_samples_overlap(dir_out_fixed_axes+directorio_muestra, dir_out_ov_split_inf+directorio_muestra, time_frame, time_overlap, sampling_period, step=3)
    characteristics_file_gen(dir_out_ov_split_inf+directorio_muestra,dir_out_new_char+directorio_muestra, filter_window=5)
    clasificar_estado(sample_id, long_sample)
    if long_sample:
        character = 'L'
    else:
        character = 'S'
    paciente = dir_out_new_char + character +str(sample_id) + '/'
    archivos_caracteristicas = natsorted(os.listdir(paciente))
    y=[]
    y_pred=[]
    fp=[]    
    fn=[]
    for file in archivos_caracteristicas:
        df = pd.read_csv(paciente + file)
        for k in range(df.shape[0]):
            oldname =  df.file_name[k]
            newname= df.new_name[k]
            estado, n_paciente, n_segmento,_,_ ,_,_= re.split(r'\D', oldname)
            estado_new, _, _,_,_ ,_,_= re.split(r'\D', newname)
            act=estado_act(estado)
            act_new=estado_act(estado_new)
            y.append(act)
            y_pred.append(act_new)
            if int(act)==0 and int(act_new)==1:
                fp.append(oldname)

            if int(act)==1 and int(act_new)==0: 
    
                fn.append(oldname)
    return(y,y_pred, fp,fn)

#x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf,  x_inestables_test_clf,x_estables_test_clf, x_ae_train, x_ae_val,x_estables_ae_train,x_inestables_ae_train,x_estables_ae_val,x_inestables_ae_val=ingesta_etiquetas()
    

if __name__== '__main__':

#train=np.concatenate((x_inestables_train_clf, x_estables_train_clf, x_ae_train), axis=None)
#val_test=np.concatenate((x_inestables_test_clf, x_estables_test_clf, x_ae_val, x_inestables_val_clf, x_estables_val_clf), axis=None)
    y=[]
    y_pred=[]
    fp=[]
    fn=[]
    #samples=np.concatenate((train,val_test))
    samples=list(range(250,251))+list(range(252,297))+list(range(298,314))
    for sample_id in  samples:  
        y_,y_pred_, fp_,fn_=devolver_estados(sample_id)
        fp=np.concatenate((fp,fp_))
        fn=np.concatenate((fn,fn_))    
        y=np.concatenate((y,y_))
        y_pred=np.concatenate((y_pred,y_pred_))
#%%
    assert len(y)==len(y_pred),"largos no coinciden"

    print("accuracy:", accuracy_score(y, y_pred))
    print("Tasa de inactivos clasificados como activos (False Positive): ",len(fp)/len(y))
    print("Tasa de activos clasificados como inactivos (False Negative): ",len(fn)/len(y))
    print("Lista de Falsos Negativos: ",fn)
    print("Lista de Falsos Positivos: ",fp)

    file1 = open(dir_etiquetas+'estado'+".txt","w")
    file1.write("accuracy:"+ str(accuracy_score(y, y_pred)) +"\n")
    file1.write("Tasa de inactivos clasificados como activos (False Positive): "+str(len(fp)/len(y))+"\n")
    file1.write("Tasa de activos clasificados como inactivos (False Negative): "+str(len(fn)/len(y))+"\n")
    file1.write("Lista de Falsos Negativos: "+str(fn)+"\n")
    file1.write("Lista de Falsos Positivos: "+str(fp)+"\n")
    file1.close()


# %%
