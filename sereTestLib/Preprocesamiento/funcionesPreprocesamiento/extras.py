import pandas as pd
import os


def get_age_pathology(extra_data_file):
    #TODO, cambiarla cuando se utilice una DB
    """
    Extract patient age and pathology
    Parameters
    ----------
    extra_data_file

    Returns
    -------
    age : int
        Patient age
    pathology : int
        Patient pathology
    """
    if extra_data_file and os.path.exists(extra_data_file):
        data = pd.read_csv(extra_data_file, low_memory=False)
        age = data.Edad.iloc[0]
        patology= data.Patologia.iloc[0]
    else:
        #TODO: Loggear si es un S para indicar que no estaba.
        age = patology = 0
    return age, patology