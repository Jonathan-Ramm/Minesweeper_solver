import numpy as np
import time
import random
from collections import deque
from multiprocessing import Process, Lock
from keras.models import load_model
import os
import cv2

MODEL_PATH = 'cnn_model_5x5.keras'
GAMMA = 0.95
REPLAY_CAPACITY = 10000
BATCH_SIZE = 32
EXPLORATION_RATE = 0.1

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)

def extract_patch(field, y, x):
    padded = np.pad(field, 2, mode='constant', constant_values=-2)
    return padded[y:y+5, x:x+5]

def is_win(field, num_mines=10):
    """
    Gibt True zurück, wenn nur noch `num_mines` ungeöffnete Zellen vorhanden sind.
    """
    num_closed = np.sum(field == -1)
    return num_closed == num_mines

def normalize_field(field):
    norm = np.copy(field).astype(np.float32)
    norm[norm == -1] = 0.0   # ungeöffnet = 0.0
    norm[norm == 0] = 0.1    # leeres Feld = 0.1
    norm[norm == 1] = 0.2
    norm[norm == 2] = 0.3
    norm[norm == 3] = 0.4
    norm[norm == 4] = 0.5
    norm[norm == 5] = 0.6
    norm[norm == 6] = 0.7
    norm[norm == 7] = 0.8
    norm[norm == 8] = 0.9
    norm[norm < -1] = -1.0   # z. B. explodiert = -1.0 (optional)
    return norm

def train_from_replay(model, buffer):
    if len(buffer) < BATCH_SIZE:
        return

    batch = buffer.sample(BATCH_SIZE)
    x_batch = []
    y_batch = []

    for state_patch, action, reward, next_patch, done in batch:
        input_patch = normalize_field(state_patch).reshape(1, 5, 5, 1)
        q_values = model.predict(input_patch, verbose=0)[0]
        if done:
            q_values[action] = reward
        else:
            future_q = model.predict(next_patch.reshape(1, 5, 5, 1), verbose=0)[0]
            q_values[action] = reward + GAMMA * np.max(future_q)
        x_batch.append(state_patch)
        y_batch.append(q_values)

    x_batch = np.array(x_batch).reshape(-1, 5, 5, 1)
    y_batch = np.array(y_batch)
    model.fit(x_batch, y_batch, verbose=0)

def test_model(driver, model):
    import matplotlib.pyplot as plt

    def read_field():
        feld = []
        cells = driver.find_elements(By.CLASS_NAME, "cell")
        for cell in cells:
            classes = cell.get_attribute("class").split()
            if "hdd_type0" in classes: feld.append(0)
            elif "hdd_type1" in classes: feld.append(1)
            elif "hdd_type2" in classes: feld.append(2)
            elif "hdd_type3" in classes: feld.append(3)
            elif "hdd_type4" in classes: feld.append(4)
            elif "hdd_type5" in classes: feld.append(5)
            elif "hdd_type6" in classes: feld.append(6)
            elif "hdd_type7" in classes: feld.append(7)
            elif "hdd_type8" in classes: feld.append(8)
            elif "hdd_closed" in classes: feld.append(-1)
            else: feld.append(-2)
        return np.array(feld).reshape(9, 9)

    from selenium.webdriver.common.by import By

    field = normalize_field(read_field())
    vis = (np.clip(field, 0, 8) * 255).astype(np.uint8)
    vis = cv2.resize(vis, (270, 270), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Aktueller Zustand", vis)
    cv2.waitKey(1)

def play_single_game(lock, process_id):
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options
    from selenium.common.exceptions import NoSuchElementException

    options = Options()
    options.headless = False
    driver = webdriver.Firefox(options=options)

    print(f"[{process_id}] Modell laden...")
    model = load_model(MODEL_PATH)
    buffer = ReplayBuffer(REPLAY_CAPACITY)

    driver.get("https://minesweeper.online/game/4875759543")
    time.sleep(1)
    # Fenster-Titel setzen für bessere Übersicht
    driver.execute_script(f"document.title = 'Process {process_id}'")


    def is_game_over():
        cells = driver.find_elements(By.CLASS_NAME, "cell")
        return any("hdd_type10" in cell.get_attribute("class") for cell in cells)

    def restart():
        try:
            driver.find_element(By.CSS_SELECTOR, 'div[id="top_area_face"]').click()
            time.sleep(0.5)
        except:
            pass

    def read_field():
        feld = []
        cells = driver.find_elements(By.CLASS_NAME, "cell")
        for cell in cells:
            classes = cell.get_attribute("class").split()
            if "hdd_type0" in classes: feld.append(0)
            elif "hdd_type1" in classes: feld.append(1)
            elif "hdd_type2" in classes: feld.append(2)
            elif "hdd_type3" in classes: feld.append(3)
            elif "hdd_type4" in classes: feld.append(4)
            elif "hdd_type5" in classes: feld.append(5)
            elif "hdd_type6" in classes: feld.append(6)
            elif "hdd_type7" in classes: feld.append(7)
            elif "hdd_type8" in classes: feld.append(8)
            elif "hdd_closed" in classes: feld.append(-1)
            else: feld.append(-2)
        return np.array(feld).reshape((9, 9))

    while True:
        restart()
        done = False
        won = False

        while not done and not won:
            field = read_field()
            #test_model(driver, model)

            q_table = []
            actions = []
            for y in range(9):
                for x in range(9):
                    if field[y, x] == -1:
                        patch = extract_patch(field, y, x)
                        input_patch = patch.reshape(1, 5, 5, 1)
                        q_val = model.predict(input_patch, verbose=0)[0][0]
                        q_table.append((q_val, y, x, patch))

            if not q_table:
                break

            if np.random.rand() < EXPLORATION_RATE:
                _, y, x, patch = random.choice(q_table)
            else:
                _, y, x, patch = max(q_table, key=lambda tup: tup[0])

            prev_field = field.copy()
            selector = f'div.cell[data-x="{x}"][data-y="{y}"]'
            try:
                driver.find_element(By.CSS_SELECTOR, selector).click()
            except:
                break


            field = read_field()
            delta = np.sum(field != -1) - np.sum(prev_field != -1)
            done = is_game_over()
            won = is_win(field)

            if done:
                reward = -100
            elif won:
                reward = 200  # Großer Reward bei Sieg
                print(f"Process {process_id}: Spiel gewonnen!")
                game_url = driver.current_url
                with open("game_log.txt", "a") as f:    
                    f.write(f"Process {process_id} gewonnen: {game_url}\n")
                print(f"Process {process_id}: Spiel gewonnen! Link gespeichert.")
            else:
                reward = delta * 2
                


            next_patch = extract_patch(field, y, x)
            buffer.add(patch, 0, reward, next_patch, done)
            train_from_replay(model, buffer)

        # Modell speichern
        lock.acquire()
        try:
            model.save(MODEL_PATH)
        finally:
            lock.release()

def main():
    lock = Lock()
    num_processes = 1
    processes = []

    for i in range(num_processes):
        p = Process(target=play_single_game, args=(lock, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
