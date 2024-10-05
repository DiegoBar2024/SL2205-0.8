import pandas as pd
from sereTestLib.parameters import *



#adquirir señal
def adquirir_señal(sample_id):
    ruta=dir_out_fixed_axes+"S"+str(sample_id)+"/3S"+str(sample_id)+".csv"
    data = pd.read_csv(ruta, low_memory=False)
    return(data)
#calcular v por canal
def calcular_v(data):
    v=np.zeros((len(data),3))
    v[:,0]=data["AC_x"]*sampling_period
    v[:,1]=data["AC_y"]*sampling_period
    v[:,2]=data["AC_z"]*sampling_period
    return(v)

#calcular v media por canal
def calcular_vm(v):
    vm=np.zeros((3))
    vm[0]=np.mean(v[:,0])
    vm[1]=np.mean(v[:,1])
    vm[2]=np.mean(v[:,2])
    return(vm)
#hacer gráficas

sample_id=68
data=adquirir_señal(sample_id)
v=calcular_v(data)
vm=calcular_vm(v)
print(vm)
