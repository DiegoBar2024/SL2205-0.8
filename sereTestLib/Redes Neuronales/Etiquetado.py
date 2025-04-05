####### IMPORTO LA CARPETA DONDE ESTÁN LOS PARÁMETROS ########
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Preprocesamiento/funcionesPreprocesamiento')
##############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parameters import *

## El csv path va a tener la dirección del archivo de las clasificaciones antropometricos.
## <<dir_etiquetas>> va a tener la ruta a la carpeta en cuyo interior se encuentra el excel de <<clasificaciones antropométricos>>
def ingesta_etiquetas(csv_path = dir_etiquetas + 'clasificaciones_antropometricos.csv', plot_path = dir_etiquetas, seed = 42):
    """
    Function that receives the labels of the samples and returns train, validation and test groups for the autoencoder and the classifier.
    It also plots data statistics.

    Parameters
    ----------
    csv_path: str, optional
        Path to the csv containing the labelsfrom scipy.cluster.hierarchy import dendrogram

    plot_path: str, optional
        Output path to the plots
    seed: int, optional
        Defaults to 42

    Returns
    -------
    groups: arrays
        Arrays containing the groups sample ids.

    """

    ## Leo el archivo csv de <<clasificaciones_antropométricos>> y guardo en una estructura DataFrame Pandas
    data = pd.read_csv(csv_path)

    ## En conclusión del análisis tengo 71 personas estables mayores a 70 años
    ## Luego tengo 72 personas estables mayores a 70 años
    ## Tengo 254 individuos en total

    ## División de grupos
    ## Hago una copia usando el método <<.copy()>> del DataFrame en donde están almacenados todos los datos del .csv
    df = data.copy()

    ## Una observación es que df[["Edad"]] y df["Edad"] NO SON IGUALES. Tienen diferentes dimensiones y son objetos de diferentes tipos
    ## Entiendo que acá hago el pasaje de las edades de tipo <<string>> a tipo <<numeric>> es decir el pasaje de cadena a número
    df[["Edad"]] = df[["Edad"]].apply(pd.to_numeric)

    ## La función <<len(df)>> me devuelve el total de filas de DATOS que tengo en mi fichero
    ## De éste modo yo lo que estoy haciendo acá es imprimiendo la cantidad de muestras que se sacaron. En total me darán 254 muestras
    print("Total de Muestras: ", len(df))

    ## SEPARACIÓN Y ETIQUETADO DE MUESTRAS ESTABLES E INESTABLES
    ## IMPORTANTE: ACÁ SE DEFINE EL CRITERIO DE CÓMO SE ETIQUETAN A LOS PACIENTES COMO ESTABLES O INESTABLES
    ## EL ETIQUETADO QUE SE USA SIGUE LOS CRITERIOS INDICADOS POR HAMLET

    ## Construyo una lista con todos aquellos pacientes denominados estables no añosos
    id_estables_no_añosos = np.array([114, 127, 128, 129, 130, 133, 213, 224, 226, 44, 294])

    ## Construyo una lista con todos aquellos pacientes denominados estables añosos
    ## En principio estos pacientes se consideran como estables pero se van a mantener por separado del análisis
    id_estables_añosos = np.array([67, 77, 111, 112, 115, 216, 229, 271, 273])

    ## Obtengo una lista con los identificadores de todos los pacientes estables
    id_estables = np.concatenate([id_estables_añosos, id_estables_no_añosos])

    ## Construyo una lista con aquellos pacientes denominados inestables
    id_inestables = np.array([69, 72, 90, 122, 137, 139, 142, 144, 148, 149, 158, 167, 178, 221, 223, 232, 256])
    
    ## Selecciono aquellos pacientes que son estables del dataframe de entrada en base a sus IDs
    estables = df[df["sampleid"].isin(id_estables)]

    ## Selecciono aquellos pacientes que son inestables del dataframe de entrada en base a sus IDs
    inestables = df[df["sampleid"].isin(id_inestables)]

    ## SEPARACIÓN DE MUESTRAS PARA EL ENTRENAMIENTO DEL CLASIFICADOR

    ## i) CONJUNTO DE ENTRENAMIENTO CON MUESTRAS ESTABLES

    ## Selecciono el 70% de las muestras estables restantes que me quedan en el DataFrame <<estables>> para el ENTRENAMIENTO del clasificador. El 30% de las muestras que quedan irán para la validación del clasificador
    ## Usando el comando <<.sample(frac=0.7)>> yo lo que hago es seleccionar ALEATORIAMENTE un 70% de las muestras que estan en el DataFrame <<estables>>
    estables_train_clf = estables.sample(frac = 0.7, random_state = 40)

    ## Usando el método <<drop>> elimino del DataFrame <<estables>> todas aquellas muestras de edad mayor o igual a 70 que no usen bastón que separé antes para entrenar al clasificador
    estables_clf = estables.drop(estables_train_clf.index)

    ## ii) CONJUNTO DE ENTRENAMIENTO CON MUESTRAS INESTABLES
    
    ## Selecciono el 70% de las muestras inestables restantes que me quedan en el DataFrame <<inestables>> para el ENTRENAMIENTO del clasificador. El 30% de las muestras que quedan irán para la validación del clasificador
    ## Usando el comando <<.sample(frac=0.7)>> yo lo que hago es seleccionar ALEATORIAMENTE un 70% de las muestras que estan en el DataFrame <<inestables>>
    inestables_train_clf = inestables.sample(frac = 0.7, random_state = 40)

    ## Usando el método <<drop>> elimino del DataFrame <<inestables>> todas aquellas muestras de edad mayor o igual a 70 que no usen bastón que separé antes para entrenar al clasificador
    inestables_clf = inestables.drop(inestables_train_clf.index)

    ## SEPARACIÓN DE MUESTRAS PARA LA VALIDACIÓN DEL CLASIFICADOR

    ## i) CONJUNTO DE VALIDACIÓN CON MUESTRAS ESTABLES

    ## Asigno los estables restantes a la validación del clasificador los cuales NO hayan sido elegidos para el entrenamiento
    estables_val_clf = estables_clf

    ## i) CONJUNTO DE VALIDACIÓN CON MUESTRAS INESTABLES

    ## Asigno los inestables restantes a la validación del clasificador los cuales NO hayan sido elegidos para el entrenamiento
    inestables_val_clf = inestables_clf

    ## SEPARACIÓN DE MUESTRAS PARA EL ENTRENAMIENTO DEL AUTOENCODER

    ## i) CONJUNTO DE ENTRENAMIENTO CON MUESTRAS ESTABLES

    ## Selecciono el 70% de las muestras estables restantes que me quedan en el DataFrame <<estables>> para el ENTRENAMIENTO del autoencoder. El 30% de las muestras que quedan irán para la validación del autoencoder
    ## Usando el comando <<.sample(frac=0.7)>> yo lo que hago es seleccionar ALEATORIAMENTE un 70% de las muestras que estan en el DataFrame <<estables>>
    estables_ae_train = estables.sample(frac = 0.7, random_state = 42)

    ## Elimino del DataFrame <<estables>> aquellas muestras que separé anteriormente para el entrenamiento del autoencoder
    estables_ae = estables.drop(estables_ae_train.index)

    ## ii) CONJUNTO DE ENTRENAMIENTO CON MUESTRAS INESTABLES

    ## Selecciono el 70% de las muestras inestables restantes que me quedan en el DataFrame <<inestables>> para el ENTRENAMIENTO del autoencoder. El 30% de las muestras que quedan irán para la validación del autoencoder
    ## Usando el comando <<.sample(frac=0.7)>> yo lo que hago es seleccionar ALEATORIAMENTE un 70% de las muestras que estan en el DataFrame <<inestables>>
    inestables_ae_train = inestables.sample(frac = 0.7, random_state = 42)

    ## Elimino del DataFrame <<inestables>> aquellas muestras que separé anteriormente para el entrenamiento del autoencoder
    inestables_ae = inestables.drop(inestables_ae_train.index)

    ## SEPARACIÓN DE MUESTRAS PARA LA VALIDACIÓN DEL AUTOENCODER

    ## i) CONJUNTO DE VALIDACIÓN CON MUESTRAS ESTABLES

    ## Como dije antes, el resto de las muestras que quedan en <<estables>> van a ir para la validación del autoencoder
    ## De éste modo, no tengo que hacer nada más que asignar las muestras que quedan en <<estables>> para el conjunto de validación del autoencoder
    estables_ae_val = estables_ae

    ## ii) CONJUNTO DE VALIDACIÓN CON MUESTRAS INESTABLES

    ## Como dije antes, el resto de las muestras que quedan en <<inestables>> van a ir para la validación del autoencoder
    ## De éste modo, no tengo que hacer nada más que asignar las muestras que quedan en <<inestables>> para el conjunto de validación del autoencoder
    inestables_ae_val = inestables_ae

    ## Termino de construír las muestras para el entrenamiento y validación del autoencoder
    ## <<ae_train>> va a contener aquellos pacientes (estables e inestables) que van a usarse para ENTRENAR AL AUTOENCODER
    ## <<ae_val>> va a contener aquellos pacientes (estables e inestables) que van a usarse para VALIDAR EL AUTOENCODER
    ae_train = pd.concat([estables_ae_train, inestables_ae_train])
    ae_val = pd.concat([estables_ae_val, inestables_ae_val])

    ## Obtengo las muestras para el entrenamiento y validación del clasificador
    clf_train = pd.concat([estables_train_clf, inestables_train_clf])
    clf_val = pd.concat([inestables_val_clf, inestables_val_clf])

    ## <<x_estables_ae_train>> va a ser una lista de los <<sample_id>> de todas las MUESTRAS ESTABLES que fueron seleccionadas para el ENTRENAMIENTO del AUOTENCODER
    ## Se hace una traducción a un vector numpy
    x_estables_ae_train = estables_ae_train["sampleid"].to_numpy()

    ## <<x_inestables_ae_train>> va a ser una lista de los <<sample_id>> de todas las MUESTRAS INESTABLES que fueron seleccionadas para el ENTRENAMIENTO del AUTOENCODER
    ## Se hace una traducción a un vector numpy
    x_inestables_ae_train = inestables_ae_train["sampleid"].to_numpy()

    ## <<x_estables_ae_val>> va a ser una lista de los <<sample_id>> de todas las MUESTRAS ESTABLES que fueron seleccionadas para la VALIDACIÓN del AUTOENCODER
    ## Se hace una traducción a un vector numpy
    x_estables_ae_val = estables_ae_val["sampleid"].to_numpy()

    ## REVISAR: ALGO ESTÁ RARO ACÁ Y ES QUE EN EL SEGUNDO MIEMBRO ESTÁN TOMANDO <<inestables_ae_train>> CUANDO EN REALIDAD DEBERÍA SER <<inestables_ae_val>>
    ##          ÉSTO HACE QUE LO QUE ESTÉ GUARDADO EN LA VARIABLE <<x_inestables_ae_val>> VA A SER LO MISMO QUE <<x_inestables_ae_train>> Y NO ESTOY SEGURO QUE DEBA SER ASÍ
    ## <<x_inestables_ae_val>> va a ser una lista de los <<sample_id>> de todas las MUESTRAS INESTABLES que fueron seleccionadas para la validación del AUTOENCODER
    ## Lo dejo comentado por las dudas pero lo corrijo en la línea siguiente
    ## x_inestables_ae_val = inestables_ae_train["sampleid"].to_numpy()
    x_inestables_ae_val = inestables_ae_val["sampleid"].to_numpy()

    ## <<x_ae_train>> va a ser un numpy array que contenga la lista de IDs de TODOS los pacientes que van a ser usados para ENTRENAR el AUTOENCODER
    x_ae_train = ae_train["sampleid"].to_numpy()

    ## <<x_ae_valor>> va a ser un numpy array que contenga la lista de IDs de TODOS los pacientes que van a ser usados para VALIDAR en AUTOENCODER
    x_ae_val = ae_val["sampleid"].to_numpy()

    ## Obtengo las ID de todos aquellos pacientes inestables que hayan sido separados para el entrenamiento del clasificador
    x_inestables_train_clf = inestables_train_clf["sampleid"].to_numpy()

    ## Obtengo las ID de todos aquellos pacientes estables que hayan sido separados para el entrenamiento del clasificador
    x_estables_train_clf = estables_train_clf["sampleid"].to_numpy()

    ## Obtengo las ID de todos aquellos pacientes inestables que hayan sido separados para la validación del clasificador
    x_inestables_val_clf = inestables_val_clf["sampleid"].to_numpy()

    ## Obtengo las ID de todos aquellos pacientes estables que hayan sido separados para la validación del clasificador
    x_estables_val_clf = estables_val_clf["sampleid"].to_numpy()

    ## Obtengo las ID de todos los pacientes que se van a usar para el entrenamiento del clasificador
    x_clf_train = clf_train["sampleid"].to_numpy()

    ## Obtengo las ID de todos los pacientes que se van a usar para la validación del clasificador
    x_clf_val = clf_val["sampleid"].to_numpy()

    ## Impresión en pantalla indicando la cantidad de muestras
    print("Cantidad de muestras para el entrenamiento del autoencoder: ", len(x_ae_train))
    print("Cantidad de muestras para la validación del autoencoder: ", len(x_ae_val))
    print("Cantidad de muestras para el entrenamiento del clasificador: ", len(x_clf_train))
    print("Cantidad de muestras para la validación del clasificador: ", len(x_clf_val))

    return (x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, x_ae_train, x_ae_val, x_estables_ae_train, x_inestables_ae_train, x_estables_ae_val, x_inestables_ae_val)