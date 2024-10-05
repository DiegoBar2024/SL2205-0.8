from parameters import static_path, model_path, act_ae
import wandb
from pathlib import Path
from autoencoder.ae_train_save_model import autoencoder_model_name_creation




home_path = str(Path.home())
static_path3 = home_path + "/Dropbox/PROJECTS/SL2205/sereData3/"
static_path4 = home_path + "/Dropbox/PROJECTS/SL2205/sereData4/"
data_dir="file://"+static_path3
run = wandb.init(project="my_project", job_type='training')

# my_data = wandb.Artifact("third_dataset", type="raw_data")
# my_data.add_reference(data_dir)


raw_data_artifact = run.use_artifact('third_dataset:latest')
raw_dataset = raw_data_artifact.checkout(static_path3)

# print(raw_dataset)




# autoencoder_name = autoencoder_model_name_creation(act_ae)+'.h5'
# dir_model= model_path
# trained_model_artifact=wandb.Artifact("modelo_ae", type="model")
# trained_model_artifact.add_dir(dir_model)
# run.log_artifact(trained_model_artifact)
# run.link_artifact(trained_model_artifact, "serelabs/my_project/prueba")

