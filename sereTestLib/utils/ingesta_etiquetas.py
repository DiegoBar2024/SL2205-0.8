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

    # No coinciden estables: 262 276 280 291 305
    # No coinciden inestables:  300 301

    ## División de grupos
    ## Hago una copia usando el método <<.copy()>> del DataFrame en donde están almacenados todos los datos del .csv
    df = data.copy()

    ## Una observación es que df[["Edad"]] y df["Edad"] NO SON IGUALES. Tienen diferentes dimensiones y son objetos de diferentes tipos
    ## Entiendo que acá hago el pasaje de las edades de tipo <<string>> a tipo <<numeric>> es decir el pasaje de cadena a número
    df[["Edad"]] = df[["Edad"]].apply(pd.to_numeric)

    ## La función <<len(df)>> me devuelve el total de filas de DATOS que tengo en mi fichero
    ## De éste modo yo lo que estoy haciendo acá es imprimiendo la cantidad de muestras que se sacaron. En total me darán 254 muestras
    print("Total de Muestras: ", len(df))

    ## SEPARACIÓN DE MUESTRAS ESTABLES E INESTABLES

    ## IMPORTANTE: ACÁ SE DEFINE EL CRITERIO DE CÓMO SE ETIQUETAN A LOS PACIENTES COMO ESTABLES O INESTABLES
    ## Mi criterio va a ser que las muestras "estables" van a ser aquellas muestras que en el campo <<Crit1>> del excel tengan valor 0
    ## Separo estables de inestables armando dos DataFrames más pequeños en base al DataFrame que contiene las 254 muestras originales

    ## Algo interesante acá es que yo puedo armar SubFrames en base a un frame original df como df' = df[<<condicion>>]
    ## Ahí <<condicion>> puede ser una condición de filtrado de datos. Por ejemplo si quiero aquellas personas mayores a 18 años: df' = df[df["Edad"] > 18]
    ## El DataFrame <<estables>> va a ser el SubDataFrame del original el cual contenga únicamente aquellas filas de las muestras ESTABLES
    estables = df[df["Crit1"] == 0]

    ## El DataFrame <<inestables>> va a ser el SubDataFrame del original el cual contenga únicamente aquellas filas de las muestras INESTABLES (el hecho que la expresión tenga paréntesis no me produce ningún cambio)
    inestables = df[(df["Crit1"] == 1)]

    ## Acá de nuevo hago un filtrado para obtener un SubDataFrame en base a un DataFrame aplicando condiciones de selección dentro de los corchetes rectos, como explicaba arriba
    ## Separo los pacientes que se infirieron mal para clasificador train. Separo primero los pacientes mal catalogados como estables
    no_coinciden_estables = estables[(estables["sampleid"] == 262) | (estables["sampleid"] == 276) | (estables["sampleid"] == 280) | (estables["sampleid"] == 291) | (estables["sampleid"] == 305)]

    ## Ahora separo aquellos pacientes que estan mal catalogados como inestables
    no_coinciden_inestables = inestables[(inestables["sampleid"] == 300) | (inestables["sampleid"] == 301)]

    ## Usando el comando <<drop>> voy a eliminar del DataFrame <<estables>> aquellas muestras que estaban mal catalogadas como estables
    ## Al aplicar el método <<.index>> se me devuelve la lista de índices asociados a todos los elementos del DataFrame <<no_coinciden_estables>>
    estables = estables.drop(no_coinciden_estables.index)

    ## Usando el comando <<drop>> voy a eliminar del DataFrame <<inestables>> aquellas muestras que estaban mal catalogadas como inestables
    ## Al aplicar el método <<.index>> se me devuelve la lista de índices asociados a todos los elementos del DataFrame <<no_coinciden_inestables>>
    inestables = inestables.drop(no_coinciden_inestables.index)

    ## SEPARACIÓN DE MUESTRAS PARA EL ENTRENAMIENTO DEL CLASIFICADOR

    ## i) CONJUNTO DE ENTRENAMIENTO CON MUESTRAS ESTABLES

    ## Separo primero 20 pacientes estables de edad mayor o igual a 70 que no usen bastón (de modo que el campo Uso Baston debe estar en 0). Como resultado me va a quedar un SubDataFrame filtrado con dichas condiciones
    ## Aplicando el método <<DataFrame.sample(n=20)>> lo que se hace es hacer un submuestreo del DataFrame seleccionando una muestra ALEATORIA de 20 pacientes del SubDataFrame
    ## Éste es parte del conjunto de muestras estables que voy a usar para entrenar al clasificador train
    estables_train_clf = estables[(estables["Edad"] > 69) & (estables["Uso Baston"] == 0)].sample(n = 20, random_state = seed)

    ## Usando el método <<drop>> elimino del DataFrame <<estables>> todas aquellas muestras de edad mayor o igual a 70 que no usen bastón que separé antes para entrenar al clasificador
    estables = estables.drop(estables_train_clf.index)

    ## Usando el método de concatenación <<pd.concat>> incluyo al DataFrame donde guardo el conjunto de entrenamiento del clasificador los que separé antes con los estables no coincidentes
    ## De éste modo el DataFrame <<estables_train_clf>> tiene un CONJUNTO DE MUESTRAS ESTABLES QUE SON LAS QUE SE VAN A UTILIZAR PARA ENTRENAR AL CLASIFICADOR
    estables_train_clf = pd.concat([estables_train_clf, no_coinciden_estables])

    ## ii) CONJUNTO DE ENTRENAMIENTO CON MUESTRAS INESTABLES
    
    ## Luego separo a 23 pacientes de edad mayor o igual a 70 los cuales NO usen baston (es decir el campo Uso Baston debe estar en 0). Como resultado me va a quedar un SubDataFrame con dichas condiciones de filtrado
    ## Aplicando el método <<DataFrame.sample(n=23)>> lo que se hace es hacer un submuestreo del DataFrame seleccionando una muestra ALEATORIA de 23 pacientes del SubDataFrame
    inestables_train_clf = inestables[(inestables["Edad"] > 69) & (inestables["Uso Baston"] == 0)].sample(n = 23,random_state=seed)

    ## Usando el método <<drop>> elimino del DataFrame <<inestables>> todas aquellas muestras de edad mayor o igual a 70 que no usen bastón que separé antes para entrenar al clasificador
    inestables = inestables.drop(inestables_train_clf.index)

    ## Usando el método de concatenación <<pd.concat>> incluyo al DataFrame donde guardo el conjunto de entrenamiento del clasificador los que separé antes con los inestables no coincidentes
    ## De éste modo el DataFrame <<estables_train_clf>> tiene un CONJUNTO DE MUESTRAS INESTABLES QUE SON LAS QUE SE VAN A UTILIZAR PARA ENTRENAR AL CLASIFICADOR
    inestables_train_clf = pd.concat([inestables_train_clf, no_coinciden_inestables])

    ## SEPARACIÓN DE MUESTRAS PARA LA VALIDACIÓN DEL CLASIFICADOR

    ## i) CONJUNTO DE VALIDACIÓN CON MUESTRAS ESTABLES

    ## Separo 10 pacientes estables con una edad mayor o igual a 70 años que no usen bastón (o sea el campo Uso Baston debe estar en 0)
    ## Al aplicar el método <<DataFrame.sample(n=10)>> yo lo que estoy haciendo es seleccionar ALEATORIAMENTE 10 muestras del SubDataFrame que sale del filtrado que se hace con las condiciones de la edad y del bastón
    ## Una cosa importante es que NO ESTOY REPITIENDO MUESTRAS con respecto al conjunto de entrenamiento que separé anteriormente
    ## Ésto es porque antes, luego de que separé mis muestras para hacer el entrenamiento, las eliminé del DataFrame de estables (idem con las inestables) usando el comando <<drop>>
    ## Haciendo ésto yo me aseguro que ninguna de las muestras que estoy seleccionando para VALIDAR va a coincidir con ninguna de las muestras que separé para ENTRENAR el clasificador
    estables_val_clf = estables[(estables["Edad"] > 69) & (estables["Uso Baston"] == 0)].sample(n = 10, random_state = seed)

    ## Elimino del DataFrame <<estables>> aquellas muestras que separé arriba para hacer la validación del clasificador
    estables = estables.drop(estables_val_clf.index)

    ## i) CONJUNTO DE VALIDACIÓN CON MUESTRAS INESTABLES

    ## Separo ahora 10 pacientes inestables los cuales tengan una edad mayor o igual a 70 años y que no usen bastón (o sea que el campo Uso Baston debe estar en 0)
    ## Al aplicar el método <<DataFrame.sample(n=10)>> yo lo que estoy haciendo es seleccionar ALEATORIAMENTE 10 muestras del SubDataFrame que sale del filtrado que hice antes con las condiciones de la edad y del bastón
    ## Por lo mismo que expliqué arriba del <<drop>>, yo me aseguro que NO ESTOY repitiendo muestras inestables que había usado para el entrenamiento, para hacer la validación
    inestables_val_clf = inestables[(inestables["Edad"] > 69) & (inestables["Uso Baston"] == 0)].sample(n = 10, random_state = seed)
    
    ## Elimino del DataFrame <<inestables>> aquellas muestras que separé arriba para hacer la validación del clasificador
    inestables = inestables.drop(inestables_val_clf.index)

    ## SEPARACIÓN DE MUESTRAS PARA EL ENTRENAMIENTO DEL AUTOENCODER

    ## i) CONJUNTO DE ENTRENAMIENTO CON MUESTRAS ESTABLES

    ## Selecciono el 70% de las muestras estables restantes que me quedan en el DataFrame <<estables>> para el ENTRENAMIENTO del autoencoder. El 30% de las muestras que quedan irán para la validación del autoencoder
    ## Usando el comando <<.sample(frac=0.7)>> yo lo que hago es seleccionar ALEATORIAMENTE un 70% de las muestras que estan en el DataFrame <<estables>>
    estables_ae_train = estables.sample(frac = 0.7, random_state = seed)

    ## ii) CONJUNTO DE ENTRENAMIENTO CON MUESTRAS INESTABLES

    ## Selecciono el 70% de las muestras inestables restantes que me quedan en el DataFrame <<inestables>> para el ENTRENAMIENTO del autoencoder. El 30% de las muestras que quedan irán para la validación del autoencoder
    ## Usando el comando <<.sample(frac=0.7)>> yo lo que hago es seleccionar ALEATORIAMENTE un 70% de las muestras que estan en el DataFrame <<inestables>>
    inestables_ae_train = inestables.sample(frac = 0.7, random_state = seed)

    ## SEPARACIÓN DE MUESTRAS PARA LA VALIDACIÓN DEL AUTOENCODER

    ## i) CONJUNTO DE VALIDACIÓN CON MUESTRAS ESTABLES

    ## Elimino del DataFrame <<estables>> aquellas muestras que separé anteriormente para el entrenamiento del autoencoder
    estables = estables.drop(estables_ae_train.index)

    ## Como dije antes, el resto de las muestras que quedan en <<estables>> van a ir para la validación del autoencoder
    ## De éste modo, no tengo que hacer nada más que asignar las muestras que quedan en <<estables>> para el conjunto de validación del autoencoder
    estables_ae_val = estables

    ## ii) CONJUNTO DE VALIDACIÓN CON MUESTRAS INESTABLES

    ## Elimino del DataFrame <<inestables>> aquellas muestras que separé anteriormente para el entrenamiento del autoencoder
    inestables = inestables.drop(inestables_ae_train.index)

    ## Como dije antes, el resto de las muestras que quedan en <<inestables>> van a ir para la validación del autoencoder
    ## De éste modo, no tengo que hacer nada más que asignar las muestras que quedan en <<inestables>> para el conjunto de validación del autoencoder
    inestables_ae_val = inestables

    ## Concateno las muestras de validacion del autoencoder que correspondan al entrenamiento del clasificador 
    ## <<estables_train_clf>> va a contener aquellos pacientes ESTABLES que van a utilizarse para entrenar el clasificador
    ## <<inestables_train_clf>> va a contener aquellos pacientes INESTABLES que van a utilizarse para entrenar al clasificador
    estables_train_clf = pd.concat([estables_train_clf, estables_ae_val[(estables_ae_val["Edad"] > 69) & (estables_ae_val["Uso Baston"] == 0)]])
    inestables_train_clf = pd.concat([inestables_train_clf, inestables_ae_val[(inestables_ae_val["Edad"] > 69) & (inestables_ae_val["Uso Baston"] == 0)]])

    ## Concateno las muestras de validacion del clasificador al entrenamiento del autoencoder
    ## <<estables_ae_train>> va a contener aquellos pacientes ESTABLES que van a utilizarse para ENTRENAR AL AUTOENCODER
    ## <<inestables_ae_train>> va a contener aquellos pacientes INESTABLES que van a utilizarse para ENTRENAR AL AUTOENCODER
    estables_ae_train = pd.concat([estables_ae_train, estables_val_clf])
    inestables_ae_train = pd.concat([inestables_ae_train, inestables_val_clf])

    ## Termino de construír las muestras para el entrenamiento y validación del autoencoder
    ## <<ae_train>> va a contener aquellos pacientes (estables e inestables) que van a usarse para ENTRENAR AL AUTOENCODER
    ## <<ae_val>> va a contener aquellos pacientes (estables e inestables) que van a usarse para VALIDAR EL AUTOENCODER
    ae_train = pd.concat([estables_ae_train, inestables_ae_train])
    ae_val = pd.concat([estables_ae_val, inestables_ae_val])

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

    x_inestables_train_clf = inestables_train_clf["sampleid"].to_numpy()
    x_estables_train_clf = estables_train_clf["sampleid"].to_numpy()
    x_inestables_val_clf = inestables_val_clf["sampleid"].to_numpy()
    x_estables_val_clf = estables_val_clf["sampleid"].to_numpy()

    print("Cantidad de muestras para el entrenamiento del autoencoder: ", len(x_ae_train))
    print("Cantidad de muestras para la validación del autoencoder: ", len(x_ae_val))
    print("Cantidad de muestras para el entrenamiento del clasificador: ", len(x_inestables_train_clf)+len(x_estables_train_clf))
    print("Cantidad de muestras para la validación del clasificador: ", len(x_inestables_val_clf)+len(x_estables_val_clf))

    #en el csv especifico los grupos
    df = df.set_index("sampleid")
    df.loc[x_ae_train, "grupo ae"] = "ae train"
    df.loc[x_ae_val, "grupo ae"] = "ae val"
    df.loc[x_estables_train_clf, "grupo clf"] = "train clf"
    df.loc[x_inestables_train_clf, "grupo clf"] = "train clf"
    df.loc[x_estables_val_clf, "grupo clf"] = "val clf"
    df.loc[x_inestables_val_clf, "grupo clf"] = "val clf"

    df.to_csv(csv_path)

    return (x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, x_ae_train, x_ae_val, x_estables_ae_train, x_inestables_ae_train, x_estables_ae_val, x_inestables_ae_val)

def ingesta_etiquetas_concat(csv_path = dir_etiquetas + 'clasificaciones_antropometricos_concat.csv', plot_path = dir_etiquetas, seed = 42):
    
    ## Se hace la lectura del .csv de 'clasificaciones_antropometricos_concat.csv'
    data = pd.read_csv (csv_path)

    #71 estables >70
    #72 inestables >70
    #254 en total
    #no coinciden estables: 262 276 280 291 305
    #no coinciden inestables:  300 301
    #División de grupos

    ## Hago una copia del DataFrame Pandas
    df = data.copy()

    ## Una observación es que df[["Edad"]] y df["Edad"] NO SON IGUALES. Tienen diferentes dimensiones y son objetos de diferentes tipos
    ## Entiendo que acá hago el pasaje de las edades de tipo <<string>> a tipo <<numeric>> es decir el pasaje de cadena a número
    df[["Edad"]] = df[["Edad"]].apply(pd.to_numeric)

    ## La función <<len(df)>> me devuelve el total de filas de DATOS que tengo en mi fichero
    ## De éste modo yo lo que estoy haciendo acá es imprimiendo la cantidad de muestras que se sacaron. En total me darán 254 muestras
    print("Total de Muestras: ", len(df))

    ## Ahí <<condicion>> puede ser una condición de filtrado de datos. Por ejemplo si quiero aquellas personas mayores a 18 años: df' = df[df["Edad"] > 18]
    ## El DataFrame <<estables>> va a ser el SubDataFrame del original el cual contenga únicamente aquellas filas de las muestras ESTABLES
    estables = df[df["Crit1"] == 0]

    ## El DataFrame <<inestables>> va a ser el SubDataFrame del original el cual contenga únicamente aquellas filas de las muestras INESTABLES (el hecho que la expresión tenga paréntesis no me produce ningún cambio)
    inestables = df[(df["Crit1"] == 1)]

    ## Luego separo a 23 pacientes de edad mayor o igual a 70 los cuales NO usen baston (es decir el campo Uso Baston debe estar en 0). Como resultado me va a quedar un SubDataFrame con dichas condiciones de filtrado
    ## Aplicando el método <<DataFrame.sample(n=23)>> lo que se hace es hacer un submuestreo del DataFrame seleccionando una muestra ALEATORIA de 23 pacientes del SubDataFrame
    no_coinciden_estables = estables[(estables["sampleid"] == 262) | (estables["sampleid"] == 276) | (estables["sampleid"] == 280) | (estables["sampleid"] == 291) | (estables["sampleid"] == 305)]
    
    ## Ahora separo aquellos pacientes que estan mal catalogados como inestables
    no_coinciden_inestables = inestables[(inestables["sampleid"] == 300) | (inestables["sampleid"] == 301)]

    ## Usando el comando <<drop>> voy a eliminar del DataFrame <<estables>> aquellas muestras que estaban mal catalogadas como estables
    ## Al aplicar el método <<.index>> se me devuelve la lista de índices asociados a todos los elementos del DataFrame <<no_coinciden_estables>>
    estables = estables.drop(no_coinciden_estables.index)

    ## Usando el método <<.sample>> se selecciona un 80% de los pacientes estables para el entrenamiento
    estables_train = estables.sample(frac = 0.8, random_state = seed)

    ## Usando el metodo <<.drop>> se eliminan de los estables aquellos que se seleccionaron para entrenar
    estables = estables.drop(estables_train.index)

    ## El 20% de los estables que quedan se van a usar para la validación
    estables_val = estables

    ## Usando el método <<.drop>> se eliminan de los inestables aquellos inestables no coincidentes
    inestables = inestables.drop(no_coinciden_inestables.index)

    ## Usando el método <<.sample>> se selecciona un 80% de los pacientes inestables para el entrenamiento
    inestables_train = inestables.sample(frac = 0.8, random_state = seed)
    
    ## Usando el método <<.drop>> se eliminan de los inestables aquellos que se seleccionaron para entrenar
    inestables = inestables.drop(inestables_train.index)

    ## El 20% de los pacientes inestables se usan para la validación
    inestables_val = inestables

    ## <<x_estables_train>> es un vector de IDs de aquellos pacientes ESTABLES que fueron separados para ENTRENAR
    x_estables_train = estables_train["sampleid"].to_numpy()

    ## <<x_inestables_train>> es un vector de IDs de aquellos pacientes INESTABLES que fueron separados para ENTRENAR
    x_inestables_train = inestables_train["sampleid"].to_numpy()

    ## <<x_estables_val>> es un vector de IDs de aquellos pacientes ESTABLES que fueron separados para VALIDAR
    x_estables_val = estables_val["sampleid"].to_numpy()

    ## <<x_inestables_val>> es un vector de IDs de aquellos pacientes INESTABLES que fueron separados para VALIDAR
    x_inestables_val = inestables_val["sampleid"].to_numpy()

    ## Imprimo la cantidad total de muestras (ESTABLES + INESTABLES) que se usan para el entrenamiento
    print("Cantidad de muestras para el entrenamiento: ", len(x_estables_train) + len(x_inestables_train))

    ## Imprimo la cantidad total de muestras (ESTABLES + INESTABLES) que se usan para la validación
    print("Cantidad de muestras para la validación: ", len(x_inestables_val) + len(x_estables_val))

    # En el csv especifico los grupos a los que pertenecen aquellos pacientes
    df = df.set_index("sampleid")
    df.loc[x_estables_train, "grupo"]= "train"
    df.loc[x_inestables_train, "grupo"]= "train"
    df.loc[x_estables_val, "grupo"]= "val"
    df.loc[x_inestables_val, "grupo"]= "val"

    ## Transformo el DataFrame Pandas en un archivo .csv
    df.to_csv(csv_path)

    return(x_estables_train, x_inestables_train,x_estables_val,x_inestables_val)

if __name__== '__main__':

    x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, x_ae_train, x_ae_val,x_estables_ae_train,x_inestables_ae_train,x_estables_ae_val,x_inestables_ae_val = ingesta_etiquetas()