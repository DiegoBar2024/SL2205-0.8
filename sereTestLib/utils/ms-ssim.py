from sereTestLib.parameters import *
from sereTestLib.autoencoder.ae_train_save_model import autoencoder_model_name_creation
from sereTestLib.autoencoder.DataClassAuto import  DataGeneratorAuto
from sereTestLib.autoencoder.ae_train_save_model import ae_train_save_model, autoencoder_model_name_creation, ssim_loss
from tensorflow import keras
import os


#levanto el modelo
basic_path = modo_ae+str(latent_dimension)+"".join(act_ae)+str(num_epochs)
autoencoder_string = ae_model_version + '_' +git_version_commit+ '_' + date_train +  '_autoencoder_' + extra
autoencoder_name =autoencoder_string +  basic_path+'ssim_loss'+'.h5'
modelo_autoencoder=keras.models.load_model(model_path + autoencoder_name, custom_objects={"ssim_loss":ssim_loss})

#entreno nuevamente
if not os.path.exists(model_path):   #Crear directorio de salida de datos
        os.makedirs(model_path)

paramsT= {'data_dir' : path_to_scalograms_train,
                            'dim': input_dimension,
                            'batch_size': batch_size,
                            'shuffle': True,
                            'activities':activities}

paramsV= {'data_dir' : path_to_scalograms_val,
                            'dim': input_dimension,
                            'batch_size': batch_size,
                            'shuffle': False,
                            'activities':activities}


training_generator = DataGeneratorAuto(list_IDs=list_of_samples_number_train, **paramsT)
validation_generator = DataGeneratorAuto(list_IDs=list_of_samples_number_validation, **paramsV)
    #print(len(training_generator[0][0][ 0,0, 0, :]))
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=2)

    # Train the model
history = autoencoder.fit(x=training_generator,
                    validation_data=validation_generator,
                    epochs=number_epochs,
                    verbose=1, callbacks=[callback])


    # Generates autoencoder file name
autoencoder_name = autoencoder_model_name_creation(activities)

    # Saves autoencoder model
autoencoder.save(model_path+autoencoder_name+model_extension)

    #plot_testgroup_scalograms([7], autoencoder,  1, results())


    # Clear keras session. Free memory and reduce loop problems
    del autoencoder
    keras.backend.clear_session()