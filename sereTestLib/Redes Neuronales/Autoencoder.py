## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/autoencoder')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/utils')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Wavelets')
from parameters import *
from Modelo_AE import *
from Etiquetado import *

## ------------------------------- SEPARACIÓN Y SELECCIÓN DE MUESTRAS ----------------------------------

## Especifico la ruta donde se encuentran los archivos
ruta_escalogramas =  'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo/train/'

## Obtengo una lista de todos los archivos presentes en la ruta donde se encuentran los escalogramas nuevos
archivos_escalogramas = [archivo for archivo in os.listdir(ruta_escalogramas)]

## Importación de etiquetas (provenientes del fichero de ingesta_etiquetas())
(x_inestables_train_clf, x_estables_train_clf, x_inestables_val_clf, x_estables_val_clf, train, val_test) = ingesta_etiquetas() 

## ESTE PROCESO SE HACE PARA NO TENER QUE CALCULAR LOS ESCALOGRAMAS DE TODOS LOS PACIENTES PARA ENTRENAR LA RED
## Creo una lista con los identificadores de aquellos pacientes cuyos escalogramas están en la carpeta
lista_pacientes_existentes = np.zeros(len(archivos_escalogramas))

## Itero para cada uno de los ficheros de escalogramas existentes
for i in range (0, len(archivos_escalogramas)):

    ## Selecciono el identificador del paciente asociado al archivo
    id_paciente_archivo = int(archivos_escalogramas[i][1:])

    ## Agrego el paciente detectado a la lista de pacientes existentes
    lista_pacientes_existentes[i] = id_paciente_archivo

## Hago que los pacientes de entrenamiento sea la intersección entre los pacientes cuyo escalograma existe y los pacientes de entrenamiento originales
train = np.intersect1d(lista_pacientes_existentes, train)

## Hago que los pacientes de validación sea la intersección entre los pacientes cuyo escalograma existe y los pacientes de validación originales
val_test = np.intersect1d(lista_pacientes_existentes, val_test)

## ----------------------------------- ENTRENAMIENTO Y VALIDACIÓN --------------------------------------

## Especifico la ruta en la cual yo voy a guardar el modelo de autoencoder
ruta_ae = 'C:\\Users\\diego/Dropbox/PROJECTS/SL2205/sereData/Modelos/autoencoder/ModeloAutoencoder/'

## Especifico el nombre que yo le voy a dar al autoencoder
nombre_autoencoder = 'AutoencoderUCU_v2'

## Especifico la cantidad de épocas del entrenamiento del autoencoder
num_epochs = 1

## Especifico el tamaño del batch que se va a utilizar para el entrenamiento del autoencoder
batch_size = 1

## Hago el llamado a la función del entrenamiento del autoencoder
ae_train_save_model(nombre_autoencoder, dir_escalogramas_nuevo_train, dir_escalogramas_nuevo_test, ruta_ae, inDim, train, val_test, num_epochs, act_ae, batch_size, debug = True)