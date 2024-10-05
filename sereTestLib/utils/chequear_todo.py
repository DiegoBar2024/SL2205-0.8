import os
from sereTestLib.parameters import *
from natsort.natsort import natsorted
from sereTestLib.utils.ingesta_etiquetas import ingesta_etiquetas


def chequear_todo(sample_ids, path1, path2):
    for id in sample_ids:
        fullD1=path1+'S'+str(id)
        files1 = os.listdir(fullD1)
        actividades = [dict_actividades.get(activity) for activity in act_clf]
        cant1=(len(natsorted([file for file in files1 if file.startswith(tuple(actividades))])))
        if cant1==0:
            print("Carpeta Vacia: ", fullD1)
        fullD2=path2+'S'+str(id)
        files2 = os.listdir(fullD2)
        actividades = [dict_actividades.get(activity) for activity in act_clf]
        cant2=(len(natsorted([file for file in files2 if file.startswith(tuple(actividades))])))
        if cant2==0:
            print("Carpeta Vacia: ", fullD2)        
        if cant1!=cant2:
            print("Distinta cantidad de archivos de id: ", id)
            print("Carpeta ",fullD1 )
            print("Tamaño ", cant1)
            print("Carpeta ",fullD2 )
            print("Tamaño ", cant2)

if __name__== '__main__':

    x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf,  x_inestables_test_clf,x_estables_test_clf, x_ae_train, x_ae_val=ingesta_etiquetas()
    train=np.concatenate((x_inestables_train_clf, x_estables_train_clf, x_ae_train), axis=None)
    print("Train ", train)
    chequear_todo(train, dir_segundo_clas + "wavelet_aumentados3/", dir_segundo_clas + "wavelet_StandarAum/")
    chequear_todo(train, dir_segundo_clas + "wavelet_AumGiro/", dir_segundo_clas + "wavelet_AumGiroStd/")

    val_test=np.concatenate(( x_inestables_val_clf, x_estables_val_clf,  x_inestables_test_clf,x_estables_test_clf,x_ae_val), axis=None)
    print("Val ", val_test)
    chequear_todo(val_test, dir_segundo_clas + "wavelet_normales/", dir_segundo_clas + "wavelet_Standarizados/")


