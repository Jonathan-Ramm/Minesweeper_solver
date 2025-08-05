import numpy as np
import time
from multiprocessing import Process, Manager, Lock
from keras.models import load_model
from keras.models import clone_model
import os

MODEL_PATH = 'my_model.keras'
GAMMA = 0.95

def average_weights(weights1, weights2):
    # Mittelung von 2 Gewichtelisten
    return [(w1 + w2) / 2 for w1, w2 in zip(weights1, weights2)]

def sync_model_update(local_model, lock):
    """
    Mit Lock sichern, gemeinsames Modell laden, gewichte mitteln und speichern.
    """
    lock.acquire()
    try:
        if os.path.exists(MODEL_PATH):
            shared_model = load_model(MODEL_PATH)
            # Gewichte laden
            shared_weights = shared_model.get_weights()
            local_weights = local_model.get_weights()

            # Mittelung
            new_weights = average_weights(shared_weights, local_weights)

            # Setze gemittelte Gewichte zurück
            shared_model.set_weights(new_weights)

            # Speichern
            shared_model.save(MODEL_PATH)
            print("Modell synchronisiert und gespeichert.")
        else:
            # Wenn kein Modell, einfach speichern
            local_model.save(MODEL_PATH)
            print("Modell gespeichert (erstes Modell).")
    finally:
        lock.release()

def play_single_game(lock, process_id):
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options
    from selenium.common.exceptions import NoSuchElementException

    options = Options()
    options.headless = False
    driver = webdriver.Firefox(options=options)

    if not os.path.exists(MODEL_PATH):
        raise Exception(f"Process {process_id}: Modell '{MODEL_PATH}' nicht gefunden!")

    print(f"Process {process_id}: Modell laden...")
    local_model = load_model(MODEL_PATH)
    print(f"Process {process_id}: Modell geladen.")

    driver.get("https://minesweeper.online/game/4875759543")
    time.sleep(2)

    def is_game_over():
        cells = driver.find_elements(By.CLASS_NAME, "cell")
        for cell in cells:
            if "hdd_type10" in cell.get_attribute("class"):
                return True
        return False

    def new_game():
        try:
            driver.find_element(By.CSS_SELECTOR, 'div[id="top_area_face"]').click()
            time.sleep(0.2)
            print(f"Process {process_id}: Neues Spiel gestartet.")
        except NoSuchElementException:
            print(f"Process {process_id}: Spiel-Neustart-Button nicht gefunden!")

    def read_field():
        feld = []
        cells = driver.find_elements(By.CLASS_NAME, "cell")
        for cell in cells:
            classes = cell.get_attribute("class").split()
            if "hdd_type0" in classes:
                feld.append(0)
            elif "hdd_type1" in classes:
                feld.append(1)
            elif "hdd_type2" in classes:
                feld.append(2)
            elif "hdd_type3" in classes:
                feld.append(3)
            elif "hdd_type4" in classes:
                feld.append(4)
            elif "hdd_type5" in classes:
                feld.append(5)
            elif "hdd_type6" in classes:
                feld.append(6)
            elif "hdd_type7" in classes:
                feld.append(7)
            elif "hdd_type8" in classes:
                feld.append(8)
            elif "hdd_closed" in classes:
                feld.append(-1)
            else:
                feld.append(-2)
        return np.array(feld, dtype=np.float32).reshape((9, 9))

    new_game()

    while True:
        while not is_game_over():
            state = read_field()
            input_state = state.reshape(1, 81)

            q_values = local_model.predict(input_state, verbose=0)[0]

            mask = (state == -1).reshape(81,)
            masked_q = np.where(mask, q_values, -np.inf)

            best_index = np.argmax(masked_q)
            row = best_index // 9
            col = best_index % 9

            try:
                selector = f'div.cell[data-x="{col}"][data-y="{row}"]'
                zielzelle = driver.find_element(By.CSS_SELECTOR, selector)
                zielzelle.click()
            except Exception as e:
                print(f"Process {process_id}: Fehler beim Klicken: {e}")
                break

            done = is_game_over()
            reward = -50 if done else 1
            new_state = read_field().reshape(1, 81)

            target = q_values.copy()
            if done:
                target[best_index] = reward
            else:
                future_q = local_model.predict(new_state, verbose=0)[0]
                target[best_index] = reward + GAMMA * np.max(future_q)

            local_model.fit(input_state, target.reshape(1, 81), verbose=0)

            if done:
                print(f"Process {process_id}: Spiel vorbei, synchronisiere Modell...")
                sync_model_update(local_model, lock)
                new_game()
                time.sleep(0.1)
                # Lade das aktuellste Modell nach der Synchronisation
                local_model = load_model(MODEL_PATH)

def main():
    from multiprocessing import Process, Lock

    lock = Lock()
    num_processes = 4
    processes = []

    for i in range(num_processes):
        p = Process(target=play_single_game, args=(lock, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
