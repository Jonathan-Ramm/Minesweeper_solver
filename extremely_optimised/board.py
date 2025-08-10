import numpy as np
import random
from collections import deque

class Minesweeper:
    def __init__(self, width, height, num_mines):
        self.W = width
        self.H = height
        self.N = num_mines
        self.reset()

    def reset(self):
        self.board = np.zeros((self.H, self.W), dtype=np.int8)       # -1 = Mine, sonst 0–8
        self.visible = np.zeros((self.H, self.W), dtype=bool)        # Sichtbar oder nicht
        self.flagged = np.zeros((self.H, self.W), dtype=bool)        # Geflaggt oder nicht
        self.mines_placed = False
        self.game_over = False

    def place_mines(self, first_click):
        cells = [(y, x) for y in range(self.H) for x in range(self.W)]
        fx, fy = first_click
        safe_zone = {(fy + dy, fx + dx)
                    for dy in range(-1, 2)
                    for dx in range(-1, 2)
                    if 0 <= fy + dy < self.H and 0 <= fx + dx < self.W}
        valid_cells = [cell for cell in cells if cell not in safe_zone]

        mines = random.sample(valid_cells, self.N)

        for y, x in mines:
            self.board[y, x] = -1

        # Zahlen setzen
        for y in range(self.H):
            for x in range(self.W):
                if self.board[y, x] == -1:
                    continue
                self.board[y, x] = self.count_adjacent_mines(x, y)

        self.mines_placed = True
        mine_count = np.sum(self.board == -1)
        if mine_count != self.N:
            print(f"Warnung: {mine_count} statt {self.N} Minen platziert!")


    def count_adjacent_mines(self, x, y):
        count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.W and 0 <= ny < self.H and self.board[ny, nx] == -1:
                    count += 1
        return count
    
    def click(self, x, y):
        if self.game_over or self.flagged[y, x]:
            return False, 0

        if not self.mines_placed:
            self.place_mines((x, y))

        if self.board[y, x] == -1:
            self.visible[y, x] = True
            self.game_over = True
            return True, 0  # Treffer auf Mine

        revealed = self.flood_fill(x, y)
        return False, revealed
    
    def flood_fill(self, x, y):
        queue = deque()
        queue.append((x, y))
        revealed = 0

        while queue:
            cx, cy = queue.popleft()
            if self.visible[cy, cx]:
                continue

            self.visible[cy, cx] = True
            revealed += 1

            if self.board[cy, cx] == 0:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < self.W and 0 <= ny < self.H:
                            if not self.visible[ny, nx] and not self.flagged[ny, nx]:
                                queue.append((nx, ny))
        return revealed

    def flag(self, x, y):
        if not self.visible[y, x] and not self.flagged[y, x]:
            self.flagged[y, x] = True

    def get_visible_state(self):
        """Gibt sichtbaren Spielzustand zurück (nur Zahlen oder -2 für verdeckt)."""
        state = np.full((self.H, self.W), -2, dtype=np.int8)
        for y in range(self.H):
            for x in range(self.W):
                if self.visible[y, x]:
                    state[y, x] = self.board[y, x]
                elif self.flagged[y, x]:
                    state[y, x] = -3  # Sondercode für Flagge
        return state

    def is_finished(self):
        return np.sum(self.visible) + self.N == self.W * self.H


    def print_board(self):
        state = self.get_visible_state()
        mapping = { -2: "■", -3: "⚑", -1: "*", 0: " ", 1: "1", 2: "2", 3: "3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8" }
        for row in state:
            print(" ".join(mapping.get(int(cell), "?") for cell in row))
        print("")

class RuleBasedSolver:
    def __init__(self, game: Minesweeper):
        self.game = game
        self.H = game.H
        self.W = game.W

    def get_neighbors(self, x, y):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.W and 0 <= ny < self.H:
                    yield nx, ny

    def step(self):
        """Führt alle sicheren Züge aus. Gibt True zurück, wenn etwas getan wurde."""
        if self.apply_simple_patterns():
            print("Simple pattern logic applied")
            return True
        if self.apply_basic_logic():
            return True
        """if self.apply_hole_patterns():
            print("Hole logic applied")
            return True"""
        if self.remaining_mines_logic():
            print("Remaining mines logic applied")
            return True 
        if self.apply_reduction_logic():
            return True
        if self.random_move():
            print("Random move applied")
            return True
        return False

    def apply_basic_logic(self):
        visible = self.game.visible
        board = self.game.board
        flagged = self.game.flagged

        did_something = False

        for y in range(self.H):
            for x in range(self.W):
                if not visible[y, x]:
                    continue
                value = board[y, x]
                if value <= 0:
                    continue

                covered = []
                flag_count = 0

                for nx, ny in self.get_neighbors(x, y):
                    if 0 <= nx < self.W and 0 <= ny < self.H:
                        if flagged[ny, nx]:
                            flag_count += 1
                        elif not visible[ny, nx]:
                            covered.append((nx, ny))

                # Bedingung 1: Alle restlichen verdeckten Nachbarn = Minen
                if value - flag_count == len(covered) and len(covered) > 0:
                    for nx, ny in covered:
                        if not flagged[ny, nx] and not visible[ny, nx]:
                            self.game.flag(nx, ny)
                            did_something = True

                # Bedingung 2: Alle Minen bereits markiert => Rest sicher
                if flag_count == value and len(covered) > 0:
                    for nx, ny in covered:
                        if not visible[ny, nx]:
                            self.game.click(nx, ny)
                            did_something = True

        return did_something


    def apply_simple_patterns(self):
        state = self.game.get_visible_state()

        def reduced_value(x, y):
            """Gibt die Zahl am Feld minus der gesetzten Minen drumherum zurück."""
            if not (0 <= x < self.W and 0 <= y < self.H):
                return None
            val = state[y, x]
            if val < 0:  # verdeckt oder Flagge
                return val
            flags = 0
            for ny in range(y-1, y+2):
                for nx in range(x-1, x+2):
                    if 0 <= nx < self.W and 0 <= ny < self.H:
                        if self.game.flagged[ny, nx]:
                            flags += 1
            return val - flags

        # -------- Horizontal 1-2-1 (reduziert) --------
        for y in range(self.H):
            for x in range(self.W - 2):
                if reduced_value(x, y) == 1 and reduced_value(x+1, y) == 2 and reduced_value(x+2, y) == 1:
                    cands = [(x - 1, y), (x + 3, y)]
                    for cx, cy in cands:
                        if 0 <= cx < self.W and 0 <= cy < self.H:
                            if not self.game.visible[cy, cx] and not self.game.flagged[cy, cx]:
                                self.game.click(cx, cy)
                                return True

        # -------- Horizontal 1-2-2-1 (reduziert + Flaggen setzen) --------
        for y in range(self.H):
            for x in range(self.W - 3):
                if reduced_value(x, y) == 1 and reduced_value(x+1, y) == 2 and reduced_value(x+2, y) == 2 and reduced_value(x+3, y) == 1:
                    for fx in (x+1, x+2):
                        for fy in (y-1, y+1):
                            if 0 <= fx < self.W and 0 <= fy < self.H and not self.game.flagged[fy, fx]:
                                self.game.flag(fx, fy)
                    return True

        # -------- Vertikal 1-2-1 (reduziert) --------
        for x in range(self.W):
            for y in range(self.H - 2):
                if reduced_value(x, y) == 1 and reduced_value(x, y+1) == 2 and reduced_value(x, y+2) == 1:
                    cands = [(x, y - 1), (x, y + 3)]
                    for cx, cy in cands:
                        if 0 <= cx < self.W and 0 <= cy < self.H:
                            if not self.game.visible[cy, cx] and not self.game.flagged[cy, cx]:
                                self.game.click(cx, cy)
                                return True

        # -------- Vertikal 1-2-2-1 (reduziert + Flaggen setzen) --------
        for x in range(self.W):
            for y in range(self.H - 3):
                if reduced_value(x, y) == 1 and reduced_value(x, y+1) == 2 and reduced_value(x, y+2) == 2 and reduced_value(x, y+3) == 1:
                    for fy in (y+1, y+2):
                        for fx in (x-1, x+1):
                            if 0 <= fx < self.W and 0 <= fy < self.H and not self.game.flagged[fy, fx]:
                                self.game.flag(fx, fy)
                    return True

        return False



    """def apply_hole_patterns(self):
        state = self.game.get_visible_state()
        for y in range(1, self.H - 1):
            for x in range(1, self.W - 1):
                if state[y, x] == 1:
                    around = [(x, y-1), (x-1, y), (x+1, y), (x, y+1)]
                    if all(0 <= ax < self.W and 0 <= ay < self.H and state[ay, ax] == -2 for ax, ay in around):
                        self.game.flag(x, y-1)
                        self.game.flag(x-1, y)
                        self.game.flag(x+1, y)
                        self.game.flag(x, y+1)
                        return True
        return False"""

    def remaining_mines_logic(self):
        remaining = self.game.N - np.sum(self.game.flagged)
        if remaining >= 7:
            return False

        covered = [(x, y) for y in range(self.H) for x in range(self.W)
                   if not self.game.visible[y, x] and not self.game.flagged[y, x]]

        from itertools import combinations
        for combo in combinations(covered, remaining):
            test_flags = set(combo)
            if self.validate_flag_combination(test_flags):
                safe = set(covered) - test_flags
                for x, y in safe:
                    self.game.click(x, y)
                return True
        return False

    def validate_flag_combination(self, test_flags):
        for y in range(self.H):
            for x in range(self.W):
                if not self.game.visible[y, x]:
                    continue
                value = self.game.board[y, x]
                if value <= 0:
                    continue
                neighbors = self.get_neighbors(x, y)
                count = sum((nx, ny) in test_flags for nx, ny in neighbors if 0 <= nx < self.W and 0 <= ny < self.H)
                covered = [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.W and 0 <= ny < self.H
                           and not self.game.visible[ny, nx] and (nx, ny) not in test_flags]
                if count > value or count + len(covered) < value:
                    return False
        return True

    def apply_reduction_logic(self):
        # Problem für Später :sob:
        return False


    def random_move(self):
        candidates = [
            (x, y)
            for y in range(self.H)
            for x in range(self.W)
            if not self.game.visible[y, x] and not self.game.flagged[y, x]
        ]
        if candidates:
            x, y = random.choice(candidates)
            self.game.click(x, y)

import pygame
import sys

class GUI:
    CELL_SIZE = 20
    GRID_W, GRID_H = 16, 16
    WIDTH = CELL_SIZE * GRID_W
    HEIGHT = CELL_SIZE * GRID_H + 50  # Platz für Buttons unten

    COLORS = {
        "covered": (92, 92, 92),
        "uncovered": (224, 224, 224),
        "flag": (255, 0, 0),
        "mine": (0, 0, 0),
        "text": (0, 0, 255),
        "bg": (128, 128, 128),
        "button": (100, 100, 100),
        "button_hover": (150, 150, 150),
        "button_text": (255, 255, 255),
    }

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Minesweeper")
        self.font = pygame.font.SysFont(None, 24)
        self.clock = pygame.time.Clock()

        self.auto_solver = False  # ⇨ Schalter
        self.create_new_game()

    def create_new_game(self):
        self.game = Minesweeper(self.GRID_W, self.GRID_H, 40)
        self.solver = RuleBasedSolver(self.game)

    def draw_board(self):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if self.game.visible[y, x]:
                    pygame.draw.rect(self.screen, self.COLORS["uncovered"], rect)
                    val = self.game.board[y, x]
                    if val > 0:
                        txt = self.font.render(str(val), True, self.COLORS["text"])
                        self.screen.blit(txt, (x * self.CELL_SIZE + 5, y * self.CELL_SIZE + 2))
                    elif val == -1:
                        pygame.draw.circle(self.screen, self.COLORS["mine"], rect.center, self.CELL_SIZE // 3)
                else:
                    pygame.draw.rect(self.screen, self.COLORS["covered"], rect)
                    if self.game.flagged[y, x]:
                        pygame.draw.circle(self.screen, self.COLORS["flag"], rect.center, self.CELL_SIZE // 3)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

    def draw_ui(self):
        # Neustart-Knopf
        self.restart_btn = pygame.Rect(10, self.HEIGHT - 40, 100, 30)
        pygame.draw.rect(self.screen, self.COLORS["button"], self.restart_btn)
        txt = self.font.render("Neustart", True, self.COLORS["button_text"])
        self.screen.blit(txt, (self.restart_btn.x + 10, self.restart_btn.y + 5))

        # Auto-Solver-Schalter
        self.auto_solver_btn = pygame.Rect(120, self.HEIGHT - 40, 180, 30)
        color = self.COLORS["button_hover"] if self.auto_solver else self.COLORS["button"]
        pygame.draw.rect(self.screen, color, self.auto_solver_btn)
        status = "AN" if self.auto_solver else "AUS"
        txt2 = self.font.render(f"Auto Solver: {status}", True, self.COLORS["button_text"])
        self.screen.blit(txt2, (self.auto_solver_btn.x + 10, self.auto_solver_btn.y + 5))

    def handle_click(self, pos, button):
        x = pos[0] // self.CELL_SIZE
        y = pos[1] // self.CELL_SIZE

        if y >= self.GRID_H:
            return  # Klick im Buttonbereich → ignorieren

        if x < 0 or x >= self.GRID_W or y < 0 or y >= self.GRID_H:
            return

        if button == 1:
            hit_mine, _ = self.game.click(x, y)
            if hit_mine:
                print("Game Over")
        elif button == 3:
            self.game.flag(x, y)

    def check_ui_click(self, pos):
        if self.restart_btn.collidepoint(pos):
            self.create_new_game()
            print("Neustart!")
        elif self.auto_solver_btn.collidepoint(pos):
            self.auto_solver = not self.auto_solver
            print(f"Auto Solver: {'AN' if self.auto_solver else 'AUS'}")

    def main(self):
        self.screen.fill(self.COLORS["bg"])
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.pos[1] >= self.HEIGHT - 50:
                        self.check_ui_click(event.pos)
                    else:
                        self.handle_click(event.pos, event.button)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        changed = self.solver.step()
                        

            if self.auto_solver:
                changed = self.solver.step()
                

            
            self.draw_board()
            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(30)
            


if __name__ == "__main__":
    GUI().main()








"""def benchmark(n=10000):
    wins = 0
    for i in range(n):
        game = Minesweeper(10, 10, 10)
        solver = RuleBasedSolver(game)
        game.click(0, 0)

        while not game.game_over and not game.is_finished():
            if not solver.solve():
                break  # Keine sicheren Züge mehr möglich

        if game.is_finished():
            wins += 1

        #print(f"Spiel {i+1}/{n} beendet – {'Gewonnen' if game.is_finished() else 'Verloren'}")

    print(f"\nBenchmark: {wins}/{n} Spiele gewonnen ({100*wins/n:.1f}%)")"""
