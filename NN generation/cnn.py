from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import numpy as np

def create_cnn_model(input_shape=(5,5,1), num_actions=25):
    model = Sequential()
    model.add(input_shape)
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))  # nur ein Q-Wert für das Zentrum

    
    model.compile(optimizer='adam', loss='mse')
    return model

# Beispiel Input: 9x9 Board als 3D Tensor (Batchgröße 1, Höhe 9, Breite 9, Kanäle 1)
dummy_input = np.zeros((1,5,5,1), dtype=np.float32)

model = create_cnn_model()
q_values = model.predict(dummy_input)   
print(q_values.shape)  # (1, 81)

model.save("cnn_model_5x5.keras")  # Speichern des Modells