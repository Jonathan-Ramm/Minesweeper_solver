from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

def build_dqn():
    model = Sequential()
    model.add(Input(shape=(81,)))  # Eingabe = 9x9 Spielfeld
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(81, activation='linear'))  # Q-Werte für jede Zelle
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

model = build_dqn()
model.save("my_model2.keras")  # Nur beim ersten Mal nötig
