## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from sereTestLib.Cinematica.SegmentacionM1 import *

## --------------------------------------- FILTRADO ACELERACIÓN ----------------------------------------

## Defino la señal de aceleración mediolateral como la aceleración medida por el sensor
acel_ML = acel[:,0]

## Hago un filtrado pasabanda para quedarme con la componente fundamental
sos = signal.butter(N = 4, Wn = [0.5, 1.1], btype = 'bandpass', fs = 1 / periodoMuestreo, output = 'sos')

## Aplico la etapa de filtrado a la señal
acel_ML_filtrada = signal.sosfiltfilt(sos, acel_ML)

plt.plot(acc_analytic[:,1] - np.mean(acc_analytic[:,1]), label = 'Corregida')
plt.plot(acel[:,0], label = 'Original')
plt.plot(acel_ML_filtrada,label='Pasos')
plt.legend()
plt.show()

## -------------------------------------- DISTINCIÓN DE PASOS ------------------------------------------

## Creo una lista en donde voy a guardar la pierna con la cual se dio el paso
## Asigno 1 para una pierna y 0 para otra pierna
## Una vez que sepamos la dirección de los ejes puedo asignar 1 o 0 como izquierda o derecha
piernas_pasos = []

## Itero para cada uno de los pasos que tengo detectados
for i in range (len(pasos) - 1):

    ## Hago la segmentación de la señal
    segmento = acel_ML_filtrada[pasos[i]['IC'][0] : pasos[i]['IC'][1]]

    ## Calculo la diferencia del segmento para ver una aproximación de la derivada en cada paso
    ## Sean x = a el extremo izquierdo y x = b el extremo derecho
    ## Si f'(a) - f'(b) > 0 --> Concluyo que tengo un máximo local en el segmento (concavidad positiva)
    ## Si f'(a) - f'(b) < 0 --> Concluyo que tengo un mínimo local en el segmento (concavidad negativa)
    diff_segmento = np.diff(segmento)

    ## En caso de tener concavidad positiva en el segmento (maximo local)
    if (diff_segmento[0] - diff_segmento[-1] > 0):

        ## Asigno la pierna 0 como la que dio el paso
        piernas_pasos.append(0)
    
    ## En caso de tener concavidad negativa en el segmento (minimo local)
    else:

        ## Asigno la pierna 0 como la que dio el paso
        piernas_pasos.append(1)

## -------------------------------------- ASOCIACIÓN DE PASOS ------------------------------------------

## Creo una lista donde guardo los pasos que tengo agregando la orientación de la pierna
pasos_orientacion = []

## Itero para cada uno de los pasos detectados
for i in range (len(pasos) - 1):

    ## Asocio cada paso con su orientación según lo que calculé antes
    pasos_orientacion.append((piernas_pasos[i], pasos[i][0], pasos[i][1]))

print(pasos_orientacion)