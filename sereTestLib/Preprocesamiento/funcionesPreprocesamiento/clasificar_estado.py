####### IMPORTO LA CARPETA DONDE ESTÁN LOS PARÁMETROS ########
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
##############################################################

import pandas as pd
import joblib
from natsort.natsort import natsorted
import os,re
from parameters import *

def clasificar_estado(pat, long_sample = True, path = dir_out_new_char_test):
    """
    Classifies the segments according to the following activities:

    AI: inactivo: 0   | activo   1
    SP: sentado:  0   | parado   1
    CE: caminando 0   | escalera 1

    Parameters
    ----------
    pat: int
        Sample id 
    
    long_sample: bool
        
    """

    ## En caso de que la muestra tomada sea larga, asigno a <<character>> el valor 'L'    
    if long_sample:
        character = 'L'

    ## En caso de que la muestra tomada sea corta, asigno a <<character>> el valor 'S'
    else:
        character = 'S'

    ## Se carga el modelo correspondiente a la clasificacion AI (Activo/Inactivo)
    AI_model = joblib.load(filenameAI)

    ## Se carga el modelo correspondiente a la clasificacion SP (Sentado/Parado)
    SP_model = joblib.load(filenameSP)

    ## Se carga el modelo correspondiente a la clasificación CE (Caminando/Escalera)
    CE_model = joblib.load(filenameCE)

    paciente = path + '/' + character + str(pat) + '/'

    archivos_caracteristicas = natsorted(os.listdir(paciente))

    for file in archivos_caracteristicas:
        df = pd.read_csv(paciente + file)
        df['state']=AI_model.predict(df[['stdAC_y','stdAC_x']]) # AI
        #print(df['state'])
        sub1=df[['meanAC_y','meanAC_z','stdAC_y', 'stdAC_z']]   # CE
        if not sub1.loc[df.state == 1].empty :
            df.loc[df.state == 1, 'state'] =  3
        sub2=df[['meanAC_z','stdAC_x', 'stdAC_z']]              #SP
        if not sub2.loc[df.state == 0].empty:
            df.loc[df.state == 0, 'state'] =  1
        df['new_name'] = ""
        for k in range(df.shape[0]):
            name=df['file_name'].loc[k]
            name=name.replace('.csv','')
            estado, n_paciente, n_segmento = re.split(r'\D', name)
            df['new_name'].iloc[k] = str(int(df['state'].loc[k])) + character + n_paciente + 's' + str(k) + '.csv'

        df.to_csv(paciente + file , index = False)