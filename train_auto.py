import numpy as np
import time
from multiprocessing import Process, Lock
from keras.models import load_model
import os

MODEL_PATH = 'my_model.keras'
GAMMA = 0.95

def average_weights(weights1, weights2):
    return [(w1 + w2) / 2 for w1, w2 in zip(weights1, weights2)]

def sync_model_update(local_model, lock):
    lock.acquire()
    try:
        if os.path.exists(MODEL_PATH):
            shared_model = load_model(MODEL_PATH)
            shared_weights = shared_model.get_weights()
            local_weights = local_model.get_weights()
            new_weights = average_weights(shared_weights, local_weights)
            shared_model.set_weights(new_weights)
            shared_model.save(MODEL_PATH)
            print("Modell synchronisiert und gespeichert.")
        else:
            local_model.save(MODEL_PATH)
            print("Modell gespeichert (erstes Modell).")
    finally:
        lock.release()

def get_neighbors(x, y):
    return [(nx, ny) for nx in range(x-1, x+2)
                     for ny in range(y-1, y+2)
                     if 0 <= nx < 9 and 0 <= ny < 9 and (nx != x or ny != y)]

def find_safe_moves(state, flags):
    safe_to_open = []
    sure_mines = []

    for y in range(9):
        for x in range(9):
            value = state[y][x]
            if 1 <= value <= 8:
                neighbors = get_neighbors(x, y)
                closed = [(nx, ny) for (nx, ny) in neighbors if state[ny][nx] == -1 and (nx, ny) not in flags]
                marked = [(nx, ny) for (nx, ny) in neighbors if (nx, ny) in flags]
                if value == len(marked) and closed:
                    safe_to_open.extend(closed)
                elif len(closed) + len(marked) == value and closed:
                    sure_mines.extend(closed)
    return list(set(safe_to_open)), list(set(sure_mines))


def play_single_game(lock, process_id):
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options
    from selenium.common.exceptions import NoSuchElementException
    from selenium.webdriver import ActionChains


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
        return np.array(feld, dtype=np.float32).reshape((9, 9))

    new_game()

    while True:
        while not is_game_over():
            state = read_field()
            input_state = state.reshape(1, 81)

            # 🔍 Solver: sichere Minen oder sichere Züge erkennen
            for y in range(9):
                for x in range(9):
                    val = state[y, x]
                    if val in range(1, 9):  # Nur offene Zahlenfelder prüfen
                        neighbors = [(ny, nx) for ny in range(y-1, y+2)
                                                for nx in range(x-1, x+2)
                                                if 0 <= ny < 9 and 0 <= nx < 9 and not (ny == y and nx == x)]
                        closed = [(ny, nx) for ny, nx in neighbors if state[ny, nx] == -1]
                        flagged = [(ny, nx) for ny, nx in neighbors
                                if "hdd_flag" in driver.find_element(By.CSS_SELECTOR,
                                    f'div.cell[data-x="{nx}"][data-y="{ny}"]').get_attribute("class")]
                        
                        # 🚩 Sicher: alle geschlossenen sind Minen
                        if len(closed) > 0 and val - len(flagged) == len(closed):
                            for ny, nx in closed:
                                selector = f'div.cell[data-x="{nx}"][data-y="{ny}"]'
                                cell_class = driver.find_element(By.CSS_SELECTOR, selector).get_attribute("class")
                                if "hdd_flag" not in cell_class:
                                    # 🖱️ Rechtsklick
                                    driver.execute_script(
                                        "arguments[0].dispatchEvent(new MouseEvent('contextmenu', {bubbles: true}));",
                                        driver.find_element(By.CSS_SELECTOR, selector)
                                    )
                                    print(f"🧠 Solver: Mine markiert bei ({nx},{ny})")

                                    # 🎓 Learning Reward
                                    reward = +5
                                    target = local_model.predict(input_state, verbose=0)[0]
                                    target[ny * 9 + nx] = reward
                                    local_model.fit(input_state, target.reshape(1, 81), verbose=0)

                                    # 🌀 Spielfeld neu einlesen & nächste Schleife
                                    state = read_field()
                                    input_state = state.reshape(1, 81)
                            # Nach Markierung nochmal alles prüfen
                            continue

                        # ✅ Sicher: Zahl erfüllt => restliche geschlossene Felder aufdecken
                        if len(closed) > 0 and val == len(flagged):
                            for ny, nx in closed:
                                selector = f'div.cell[data-x="{nx}"][data-y="{ny}"]'
                                try:
                                    driver.find_element(By.CSS_SELECTOR, selector).click()
                                    print(f"🧠 Solver: Sicheres Feld geöffnet ({nx},{ny})")

                                    # 🎓 Learning Reward
                                    reward = +3
                                    target = local_model.predict(input_state, verbose=0)[0]
                                    target[ny * 9 + nx] = reward
                                    local_model.fit(input_state, target.reshape(1, 81), verbose=0)

                                    state = read_field()
                                    input_state = state.reshape(1, 81)
                                except Exception as e:
                                    print(f"❌ Fehler beim Solver-Klick: {e}")
                            continue  # neue Bewertung, falls neue Hinweise




           # --- DQN nur wenn keine sichere Option ---
            input_state = state.reshape(1, 81)
            q_values = local_model.predict(input_state, verbose=0)[0]
            mask = (state == -1).reshape(81,)
            masked_q = np.where(mask, q_values, -np.inf)
            best_index = np.argmax(masked_q)
            row = best_index // 9
            col = best_index % 9

            selector = f'div.cell[data-x="{col}"][data-y="{row}"]'
            try:
                zielzelle = driver.find_element(By.CSS_SELECTOR, selector)
                zielzelle.click()
            except Exception as e:
                print(f"Process {process_id}: Fehler beim DQN-Klick: {e}")
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
                local_model = load_model(MODEL_PATH)

def main():
    lock = Lock()
    processes = []
    for i in range(1):
        p = Process(target=play_single_game, args=(lock, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
