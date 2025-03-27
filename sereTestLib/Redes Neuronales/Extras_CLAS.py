import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
from parameters import *
from tensorflow import keras
from GeneradorDatos import *
# from sereTestLib.autoencoder.DataClassAuto import DataGeneratorAuto

def clasificador_name_creation(activities, clasificador):
    """
    Function that craates the classifier model name.
    Parameters:
    -----------
    activities: list
        List of activities to train and validate the model.
    clasificador: str
        classification method
    
    Returns:
    -------
    classifier model name
    """
    basic_path = modo_ae+str(latent_dimension)+"".join(activities)+str(num_epochs)
    clasificador_complete_name = clasificador_model_version +'_' + ae_model_version + '_' + git_version_commit +'_' + date_train +'_'+clasificador+'_'+ extra
    return clasificador_complete_name+basic_path

## El parámetro <<patient_list>> va a ser una lista con los IDs de los pacientes cuyas muestras voy a usar para entrenar
## El parámetro <<modelo>> va a ser el modelo del autoencoder ya entrenado
## El parámetro <<layer_name>> me dice cuál es el layer del autoencoder de donde yo voy a buscar la salida
## Esencialmente ésta función me va a devolver, para todas las muestras de entrada, cuál sería el vector de 256 características COMPRIMIDO arrojado por el autoencoder
def patient_group_aelda(patient_list, modelo, layer_name = 'Dense_encoder', **params):
    """
    Function that takes as input a list of patients and the parameters for the data generator
    and returns the intermediate layer values for each data sample.
    If the patient doesnt have samples for the activity, returns an empty array
    """
    ## Inicializo en <<intermediate>> un vector numpy el cual esté vacío
    intermediate = np.array([])

    ## Generador de datos
    ## Construyo un nuevo objeto DataGeneratorAuto al cual le paso como parámetro los IDs de los pacientes cuyas muestras voy a usar para entrenar
    ## Al pasarle <<**params>> también le paso el resto de los parámetros correspondientes (están en el fichero <<evaluar_aelda.py>>)
    generator = DataGeneratorAuto(list_IDs = patient_list, **params)

    ## Verificar que se encontraron muestras
    ## Recuerdo que el atributo <<n_samples>> me daba la CANTIDAD TOTAL DE SEGMENTOS, restringido a los pacientes cuyas IDs le pasé como parámetro y la lista de actividades que especifiqué que había encontrado el DataGenerator
    ## En caso de que exista algun segmento asociado a alguno de los pacientes que especifiqué, asociado a al menos una de las actividades que pasé, entro al if
    ## Si se da el caso de que NINGUNO de los pacientes cuyas IDs pasé como entrada tiene NINGÚN segmento asociado a la(s) actividad(es) especificada(s) retorno una lista vacía
    if generator.n_samples > 0:

        ## Se crea un modelo en base al autoencoder que toma como entrada la entrada al autoencoder y que toma como la salida las 256 características del autoencoder con los parámetros que ya tiene
        intermediate_layer_model = keras.Model(inputs = modelo.input,
                                    outputs = modelo.get_layer(layer_name).output)

        ## Se realiza entonces la predicción del autoencoder a los datos del generador dando como resultado para cada muestra las 256 características a que correspondan a la salida
        intermediate = intermediate_layer_model.predict(generator)

    return intermediate