## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
import parameters
from parameters import long_sample, dict_actividades
import os
from natsort.natsort import natsorted
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as it

## ----------------------------------- OBTENCIÓN VALORES MAGNETOMETRO ----------------------------------