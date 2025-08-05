import numpy as np
import time
import random
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains
import re

options = Options()
options.headless = False
driver = webdriver.Firefox(options=options)


def get_surrounding_cells(cell):
    x = int(cell.get_attribute("data-x"))
    y = int(cell.get_attribute("data-y"))
    surrounding = []

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            try:
                selector = f".cell[data-x='{x + dx}'][data-y='{y + dy}']"
                neighbor = driver.find_element(By.CSS_SELECTOR, selector)
                surrounding.append(neighbor)
            except:
                continue  # z. B. außerhalb des Spielfelds
    return surrounding


def main_algo():
    cells = driver.find_elements(By.CSS_SELECTOR, ".cell.hdd_opened:not(.hdd_type0)")
    if not cells:
        return
    random.shuffle(cells)

    for cell in cells:
        classes = cell.get_attribute("class").split()
        if "finished" in classes:
            continue

        # Minenzahl extrahieren, z. B. hdd_type3 → 3
        mine_count = None
        for cls in classes:
            if cls.startswith("hdd_type"):
                try:
                    mine_count = int(cls.replace("hdd_type", ""))
                    break
                except ValueError:
                    continue

        if mine_count is None:
            continue

        surrounding_cells = get_surrounding_cells(cell)

        closed_neighbors = []
        flagged_neighbors = 0

        # Umgebung klassifizieren
        for neighbor in surrounding_cells:
            neighbor_class = neighbor.get_attribute("class")
            neighbor_classes = neighbor_class.split()

            if "hdd_flag" in neighbor_classes:
                flagged_neighbors += 1
            elif "hdd_closed" in neighbor_classes:
                closed_neighbors.append(neighbor)

        actions_done = False

        # Fall 1: alle restlichen geschlossenen = restliche Minen
        if closed_neighbors and (flagged_neighbors + len(closed_neighbors)) == mine_count:
            for c in closed_neighbors:
                ActionChains(driver).context_click(c).perform()
            actions_done = True

        # Fall 2: alle Minen markiert → restliche Felder sind sicher
        elif flagged_neighbors == mine_count and closed_neighbors:
            for c in closed_neighbors:
                c.click()
            actions_done = True

        if actions_done:
            driver.execute_script("arguments[0].classList.add('finished')", cell)
            return  # Optional: nur eine Aktion pro Durchlauf
                    
        

def random_move():
    cells = driver.find_elements(By.CSS_SELECTOR, ".cell.hdd_closed")
    random.shuffle(cells)
    for cell in cells:
        cell.click()
        return

def is_win(field, num_mines=10):
    """
    Gibt True zurück, wenn nur noch `num_mines` ungeöffnete Zellen vorhanden sind.
    """
    num_closed = np.sum(field == -1)
    return num_closed == num_mines

def restart():
        try:
            driver.find_element(By.CSS_SELECTOR, 'div[id="top_area_face"]').click()
            time.sleep(0.2)
        except:
            pass

def is_game_over():
        cells = driver.find_elements(By.CLASS_NAME, "cell")
        return any("hdd_type10" in cell.get_attribute("class") for cell in cells)

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

def play_single_game(process_id):
    driver.get("https://minesweeper.online/game/4875759543")
    time.sleep(5)
    # Fenster-Titel setzen für bessere Übersicht
    driver.execute_script(f"document.title = 'Process {process_id}'")

    while True:
        restart()
        done = False
        won = False
        start = driver.find_elements(By.CSS_SELECTOR, ".start")
        for s in start:
            s.click()

        while not done and not won:
            field = read_field()
            main_algo()
            done = is_game_over()
            won = is_win(field)



def main():
    num_processes = 1
    processes = []

    for i in range(num_processes):
        play_single_game(i)

if __name__ == "__main__":
    main()
