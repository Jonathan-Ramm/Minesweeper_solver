from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
import numpy as np
import time
from keras.models import load_model
from selenium.common.exceptions import NoSuchElementException

MODEL_PATH = "my_model.keras"

# === Modell laden oder neu bauen ===
try:
    model = load_model(MODEL_PATH)
    print("✅ Modell geladen.")
except Exception as e:
    print("⚠️ Kein gespeichertes Modell gefunden, baue neues Modell...")
    from build_dqn_model import build_model
    model = build_model(81, 81)
    model.save(MODEL_PATH)
    print("✅ Neues Modell erstellt und gespeichert.")

# === Selenium Setup ===
options = Options()
options.headless = False
driver = webdriver.Firefox(options=options)
driver.get("https://minesweeper.online/game/4875759543")
time.sleep(3)

def new_game():
    try:
        driver.find_element(By.CSS_SELECTOR, 'div[id="top_area_face"]').click()
        time.sleep(0.5)
        print("🔄 Neues Spiel gestartet.")
    except NoSuchElementException:
        print("❌ Spiel-Neustart-Button nicht gefunden!")

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

def is_game_over():
    cells = driver.find_elements(By.CLASS_NAME, "cell")
    for cell in cells:
        if "hdd_type10" in cell.get_attribute("class"):
            return True
    return False

def find_clicked_cell(old_field, new_field):
    diff = (old_field == -1) & (new_field != -1)
    idx = np.where(diff)
    if len(idx[0]) == 0:
        return None, None
    return idx[1][0], idx[0][0]  # x=col, y=row

# Starte ein neues Spiel
new_game()

print("Starte manuelles Training. Klicke im Browser auf eine Zelle...")

while True:
    old_field = read_field()
    print("Feld (geschlossene Zellen = -1):")
    print(old_field)

    print("Bitte klicke im Browser auf eine Zelle...")

    changed = False
    timeout = 10
    start_time = time.time()
    while not changed and (time.time() - start_time) < timeout:
        new_field = read_field()
        if not np.array_equal(old_field, new_field):
            changed = True
        else:
            time.sleep(0.1)

    if not changed:
        print("⏰ Timeout: Kein Klick erkannt, speichere Modell und starte neues Spiel.")
        model.save(MODEL_PATH)
        new_game()
        continue

    x, y = find_clicked_cell(old_field, new_field)
    if x is None:
        print("⚠️ Keine neue geöffnete Zelle erkannt. Versuche erneut.")
        continue

    print(f"Erkannter Zug: Spalte {x}, Zeile {y}")

    new_open = np.sum((new_field != -1) & (old_field == -1))
    reward = new_open * 2
    if is_game_over():
        reward = -20
    print(f"🎯 Belohnung: {reward}")

    input_state = old_field.reshape(1, 81)
    label = model.predict(input_state, verbose=0)
    label[0, y * 9 + x] = reward

    # Training - mehrfache Fits für stärkere Gewichtung (optional)
    for _ in range(5):
        model.fit(input_state, label, verbose=0)

    if is_game_over():
        print("💥 Spiel vorbei. Speichere Modell und starte neues Spiel.")
        model.save(MODEL_PATH)
        new_game()
