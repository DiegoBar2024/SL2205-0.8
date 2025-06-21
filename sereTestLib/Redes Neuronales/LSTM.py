## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

# Generating random data
X = np.random.random((100, 10, 5)) 
y = np.random.randint(2, size=(100, 1))

model = Sequential([
    LSTM(50, activation='tanh', return_sequences=True, input_shape=(10, 5)),  # First LSTM layer
    LSTM(30, activation='tanh'),  # Second LSTM layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X, y, epochs=10, batch_size=16)