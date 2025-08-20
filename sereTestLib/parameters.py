## ACTUALIZO LAS RUTAS PARA LOS FICHEROS EN MI COMPUTADORA
## Modificar en otras máquinas
dir_in_original_sample = 'C:/Yo/Tesis/sereData/sereData/Dataset/raw_2_process'
dir_out_fixed_axes = 'C:/Yo/Tesis/sereData/sereData/Dataset/dataset'
dir_out_ov_split_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/overlaps/train'
dir_out_ov_split_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/overlaps/test'
dir_out_new_char_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/features/train'
dir_out_new_char_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/features/test'
dir_out_ov_split_rename_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/rename_overlaps/train'
dir_out_ov_split_rename_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/rename_overlaps/train'
directorio_scalogramas_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/wavelet_cmor/train'
directorio_scalogramas_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/wavelet_cmor/test'
dir_preprocessed_data_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/preprocessed_data/train'
dir_preprocessed_data_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/preprocessed_data/test'
path_log_preprocesamiento= "C:/Yo/Tesis/sereData/sereData/Logs/preprocesamiento"+ date + ".log"
filenameAI = 'C:/Yo/Tesis/sereData/sereData/Modelos/Actividad/regLin_Activo_Inactivo.sav'
filenameSP = 'C:/Yo/Tesis/sereData/sereData/Modelos/Actividad/regLin_sentadoParado.sav'
filenameCE = 'C:/Yo/Tesis/sereData/sereData/Modelos/Actividad/regLin_caminandoEscalera.sav'
dir_etiquetas=  'C:/Yo/Tesis/sereData/sereData/Etiquetas/'

## Rutas en donde van a estar los escalogramas finales que van a entrar a la red neuronal
dir_escalogramas_nuevo_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo/train'
dir_escalogramas_nuevo_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo/test'

## Rutas en donde van a estar los escalogramas individuales por cada canal
dir_escalogramas_nuevo_ind_train = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_ind_nuevo/train'
dir_escalogramas_nuevo_ind_test = 'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_ind_nuevo/test'

## Rutas en donde se van a guardar los resultados de las inferencias
results_path = "C:/Yo/Tesis/sereData/sereData/Inferencias"

## DIMENSIÓN TEMPORAL A LA QUE SE VA A REMUESTREAR
## Especifico la cantidad de pasos que quiero representar en mis escalogramas
cantidad_pasos = 4

## Especifico el tiempo máximo de paso en segundos que voy a usar para hacer el remuestreo del eje temporal
tiempo_maximo_paso = 1

## Especifico la dimensión temporal objetivo a la que voy a interpolar (remuestreo)
## Ésta cantidad está calculada con parámetros que NO DEPENDEN DEL PACIENTE, lo cual es lo que se pretende
dimension_temporal = cantidad_pasos * int(tiempo_maximo_paso * 200)

## Especifico las dimensiones temporales del tensor tridimensional de entrada al autoencoder
inDim = (6, 128, dimension_temporal)

## Especifico el nombre de la métrica de error a utilizar
loss_name = "mse"