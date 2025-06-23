## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

# Generating random data
X = np.random.random((100, 108, 256)) 
y = np.random.randint(2, size=(100, 1))

model = Sequential([
    LSTM(50, activation='tanh', return_sequences=True, input_shape=(108, 256)),  # First LSTM layer
    LSTM(30, activation='tanh'),  # Second LSTM layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X, y, epochs=10, batch_size=16)

## Construyo un modelo secuencial de Keras
model1 = Sequential()

## Agrego una capa de LSTM al modelo especificando la cantidad de hidden units
model1.add(LSTM(100, activation = 'tanh', return_sequences = True, input_shape = (108, 256)))

## Agrego una segunda capa de LSTM
model1.add(LSTM(30, activation = 'tanh'))

## Agrego una neuronal de salida para hacer la clasificacion
model1.add(Dense(1, activation = 'sigmoid'))

## Compilo el modelo LSTM
model1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model1.fit(X, y, epochs=10, batch_size=16)