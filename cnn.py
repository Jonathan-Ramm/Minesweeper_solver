import numpy as np
import time
import os
import logging
import random
from multiprocessing import Process, Lock
from keras.models import load_model, Sequential
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.optimizers import Adam

MODEL_PATH = 'keras_data/cnn_model.keras'
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
BATCH_SIZE = 16
LOG_FILE = 'training.log'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def create_cnn_model():
    model = Sequential()
    model.add(Input(shape=(9, 9, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(81, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def load_model_safe(path, retries=10, delay=1):
    for i in range(retries):
        try:
            return load_model(path)
        except Exception as e:
            logging.warning(f"Fehler beim Laden des Modells: {e} (Versuch {i+1}/{retries})")
            time.sleep(delay)
    raise Exception("Modell konnte nicht geladen werden.")

def average_weights(weights1, weights2):
    return [(w1 + w2) / 2 for w1, w2 in zip(weights1, weights2)]

def sync_model_update(local_model, lock):
    with lock:
        if os.path.exists(MODEL_PATH):
            shared_model = load_model(MODEL_PATH)
            new_weights = average_weights(shared_model.get_weights(), local_model.get_weights())
            shared_model.set_weights(new_weights)
            shared_model.save(MODEL_PATH)
            logging.info("Modell synchronisiert.")
        else:
            local_model.save(MODEL_PATH)
            logging.info("Modell initial gespeichert.")

def count_revealed_cells(field):
    return np.sum(field != -1)

def play_single_game(lock, process_id):
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options
    from selenium.common.exceptions import NoSuchElementException

    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options)

    if not os.path.exists(MODEL_PATH):
        model = create_cnn_model()
        model.save(MODEL_PATH)

    model = load_model_safe(MODEL_PATH)
    epsilon = EPSILON_START

    def is_game_over():
        cells = driver.find_elements(By.CLASS_NAME, "cell")
        return any("hdd_type10" in c.get_attribute("class") for c in cells)

    def new_game():
        try:
            driver.get("https://minesweeper.online/game/4875759543")
            time.sleep(1)
            face = driver.find_element(By.CSS_SELECTOR, 'div[id="top_area_face"]')
            face.click()
            time.sleep(0.5)
            logging.info(f"[P{process_id}] Neues Spiel gestartet.")
        except Exception as e:
            logging.warning(f"[P{process_id}] Fehler beim Neustart: {e}")

    def read_field():
        feld = []
        for cell in driver.find_elements(By.CLASS_NAME, "cell"):
            cls = cell.get_attribute("class")
            if "hdd_type" in cls:
                for i in range(9):
                    if f"hdd_type{i}" in cls:
                        feld.append(i)
                        break
                else:
                    feld.append(-2)
            elif "hdd_closed" in cls:
                feld.append(-1)
            else:
                feld.append(-2)
        return np.array(feld, dtype=np.float32).reshape((9, 9))

    new_game()
    memory = []

    while True:
        prev_field = read_field()
        input_state = prev_field.reshape(1, 9, 9, 1)

        if random.random() < epsilon:
            valid_indices = np.argwhere(prev_field.flatten() == -1).flatten()
            best_index = random.choice(valid_indices)
        else:
            q_values = model.predict(input_state, verbose=0)[0]
            mask = (prev_field.flatten() == -1)
            masked_q = np.where(mask, q_values, -np.inf)
            best_index = np.argmax(masked_q)

        row, col = divmod(best_index, 9)
        try:
            selector = f'div.cell[data-x="{col}"][data-y="{row}"]'
            ziel = driver.find_element(By.CSS_SELECTOR, selector)
            ziel.click()
        except Exception as e:
            logging.warning(f"[P{process_id}] Klickfehler: {e}")
            continue

        done = is_game_over()
        new_field = read_field()
        delta = max(0, count_revealed_cells(new_field) - count_revealed_cells(prev_field))

        reward = -100 if done else delta * 2
        memory.append((prev_field.copy(), best_index, reward, new_field.copy(), done))

        if done:
            logging.info(f"[P{process_id}] Spiel vorbei. Belohnung: {reward}. Training...")
            if len(memory) >= BATCH_SIZE:
                for _ in range(3):  # mehrfach trainieren
                    minibatch = random.sample(memory, min(BATCH_SIZE, len(memory)))
                    X, y = [], []
                    for state, action, reward, next_state, done in minibatch:
                        q_vals = model.predict(state.reshape(1, 9, 9, 1), verbose=0)[0]
                        if done:
                            q_vals[action] = reward
                        else:
                            future_q = model.predict(next_state.reshape(1, 9, 9, 1), verbose=0)[0]
                            q_vals[action] = reward + GAMMA * np.max(future_q)
                        X.append(state.reshape(9, 9, 1))
                        y.append(q_vals)
                    model.fit(np.array(X), np.array(y), verbose=0)
            memory.clear()
            sync_model_update(model, lock)
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            new_game()
            model = load_model_safe(MODEL_PATH)
            time.sleep(0.2)

def main():
    lock = Lock()
    num_processes = 4
    processes = [Process(target=play_single_game, args=(lock, i)) for i in range(num_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
