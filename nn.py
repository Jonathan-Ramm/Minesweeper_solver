import keras
from keras import layers, Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import tensorflow as tf
import tensorflowjs as tfjs

# Modell definieren
model = keras.Sequential(
    [
        keras.Input(shape=(81,)),
        layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(), 
        layers.Dropout(0.3), 
        layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)), 
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(81, activation='linear') 
    ]
)

# Dummy-Kompilierung (nur nötig, wenn du auch trainieren willst)
model.compile(optimizer='adam', loss='mse')
