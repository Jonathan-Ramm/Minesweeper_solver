import numpy as np
import random
from collections import deque, defaultdict
from itertools import combinations, chain

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
        # first_click erwartet (x,y)
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
        mine_count = int(np.sum(self.board == -1))
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
        # Toggle-Flag: Flaggen/Entfernen möglich, aber nicht auf sichtbaren Feldern
        if 0 <= x < self.W and 0 <= y < self.H and not self.visible[y, x]:
            self.flagged[y, x] = not self.flagged[y, x]

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
        return int(np.sum(self.visible)) + int(self.N) == self.W * self.H


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

        # Precompute neighbor lists for speed
        self.neighbors_map = {}
        for y in range(self.H):
            for x in range(self.W):
                nb = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.W and 0 <= ny < self.H:
                            nb.append((nx, ny))
                self.neighbors_map[(x, y)] = nb

        # Component size limit for combinatorial searches
        self.MAX_COMPONENT_SIZE = 30  # konservativ; bei Bedarf erhöhen

    def get_neighbors(self, x, y):
        return list(self.neighbors_map.get((x, y), []))

    def step(self):
        """Führt alle sicheren Züge aus. Gibt True zurück, wenn etwas getan wurde."""
        # 1) klassische sichere Logik
        if self.apply_basic_logic():
            #print("basic applied")
            return True

        # 2) einfache Muster, vorsichtig (nur sichere Aktionen)
        if self.apply_simple_patterns():
            #print("simple patterns applied")
            return True

        # 3) Lochmuster (nur wenn validierbar)
        if self.apply_hole_patterns():
            #print("hole patterns applied")
            return True

        # 4) Paarweise Subset-Logik (A \ B => mines/safe)
        if self.apply_pairwise_subset_rule():
            #print("pairwise subset applied")
            return True

        # 5) Reduktionslogik (komponentenspezifisch, kombinatorisch, aber begrenzt)
        if self.apply_reduction_logic():
            #print("reduction applied")
            return True

        # 6) verbleibende Minen - Prüfe kleine Komponenten
        if self.remaining_mines_logic():
            #print("remaining mines applied")
            return True

        # 7) Fallback: zufälliger Zug
        if self.random_move():
            #print("random move")
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
                value = int(board[y, x])
                if value <= 0:
                    continue

                covered = []
                flag_count = 0

                for nx, ny in self.get_neighbors(x, y):
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
                        if not visible[ny, nx] and not flagged[ny, nx]:
                            # safe click
                            hit, _ = self.game.click(nx, ny)
                            if hit:
                                # Sollte nicht passieren, aber handle gracefully
                                self.game.game_over = True
                            did_something = True

        return did_something


    def apply_simple_patterns(self):
        """Wenige sichere, schnelle Muster. Flag-Kandidaten werden vorher global validiert."""
        state = self.game.get_visible_state()

        def reduced_value(x, y):
            """Gibt die Zahl am Feld minus der gesetzten Minen drumherum zurück."""
            if not (0 <= x < self.W and 0 <= y < self.H):
                return None
            val = state[y, x]
            if val < 0:  # verdeckt oder Flagge
                return val
            flags = 0
            for nx, ny in self.get_neighbors(x, y):
                if self.game.flagged[ny, nx]:
                    flags += 1
            return int(val) - flags

        # -------- Horizontal 1-2-1 (reduziert) --------
        for y in range(self.H):
            for x in range(self.W - 2):
                if reduced_value(x, y) == 1 and reduced_value(x+1, y) == 2 and reduced_value(x+2, y) == 1:
                    # Kandidaten sind (x-1,y) und (x+3,y) als sichere Aufdeckungen
                    cands = [(x - 1, y), (x + 3, y)]
                    for cx, cy in cands:
                        if 0 <= cx < self.W and 0 <= cy < self.H:
                            if not self.game.visible[cy, cx] and not self.game.flagged[cy, cx]:
                                # nur klicken wenn der Klick nach klassischen Regeln sicher erscheint
                                hit, _ = self.game.click(cx, cy)
                                if hit:
                                    self.game.game_over = True
                                return True

        # -------- Horizontal 1-2-2-1 (reduziert + Flaggen setzen wenn global valid) --------
        for y in range(self.H):
            for x in range(self.W - 3):
                if reduced_value(x, y) == 1 and reduced_value(x+1, y) == 2 and reduced_value(x+2, y) == 2 and reduced_value(x+3, y) == 1:
                    # mögliche Flags (oben oder unten der inneren 2er)
                    candidate_flags = set()
                    for fx in (x+1, x+2):
                        for fy in (y-1, y+1):
                            if 0 <= fx < self.W and 0 <= fy < self.H and not self.game.flagged[fy, fx] and not self.game.visible[fy, fx]:
                                candidate_flags.add((fx, fy))
                    # Validate candidate flags before setting
                    if candidate_flags:
                        current_flags = set((cx, cy) for cy in range(self.H) for cx in range(self.W) if self.game.flagged[cy, cx])
                        test_flags = current_flags.union(candidate_flags)
                        if self.validate_flag_combination(test_flags):
                            for fx, fy in candidate_flags:
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
                                hit, _ = self.game.click(cx, cy)
                                if hit:
                                    self.game.game_over = True
                                return True

        # -------- Vertikal 1-2-2-1 (reduziert + Flaggen setzen wenn valid) --------
        for x in range(self.W):
            for y in range(self.H - 3):
                if reduced_value(x, y) == 1 and reduced_value(x, y+1) == 2 and reduced_value(x, y+2) == 2 and reduced_value(x, y+3) == 1:
                    candidate_flags = set()
                    for fy in (y+1, y+2):
                        for fx in (x-1, x+1):
                            if 0 <= fx < self.W and 0 <= fy < self.H and not self.game.flagged[fy, fx] and not self.game.visible[fy, fx]:
                                candidate_flags.add((fx, fy))
                    if candidate_flags:
                        current_flags = set((cx, cy) for cy in range(self.H) for cx in range(self.W) if self.game.flagged[cy, cx])
                        test_flags = current_flags.union(candidate_flags)
                        if self.validate_flag_combination(test_flags):
                            for fx, fy in candidate_flags:
                                self.game.flag(fx, fy)
                            return True

        return False


    def apply_hole_patterns(self):
        """Erkennen von 'Kreuzloch'-Mustern: eine sichtbare 1 mit vier orthogonal
        verdeckten Nachbarn (oben/unten/links/rechts). Nur Flaggen, wenn global konsistent."""
        state = self.game.get_visible_state()
        for y in range(1, self.H - 1):
            for x in range(1, self.W - 1):
                if state[y, x] != 1:
                    continue
                orth = [(x, y-1), (x-1, y), (x+1, y), (x, y+1)]
                # alle orthogonal verdeckt (nicht sichtbar, nicht geflaggt aktuell)
                if all(0 <= ax < self.W and 0 <= ay < self.H and (not self.game.visible[ay, ax]) for ax, ay in orth):
                    candidate_flags = set((ax, ay) for ax, ay in orth if not self.game.flagged[ay, ax])
                    if not candidate_flags:
                        continue
                    current_flags = set((cx, cy) for cy in range(self.H) for cx in range(self.W) if self.game.flagged[cy, cx])
                    test_flags = current_flags.union(candidate_flags)
                    if self.validate_flag_combination(test_flags):
                        for fx, fy in candidate_flags:
                            self.game.flag(fx, fy)
                        return True
        return False

    def remaining_mines_logic(self):
        """Versucht durch kombinatorische Überlegung auf kleinen Komponenten sichere Felder zu finden.
        Wenn noch >= 10 Minen verbleiben, wird abgebrochen (zu teuer)."""
        remaining = int(self.game.N - np.sum(self.game.flagged))
        if remaining >= 10:
            return False

        # Sammle alle verdeckten, ungeflagten Felder
        covered = [(x, y) for y in range(self.H) for x in range(self.W)
                   if not self.game.visible[y, x] and not self.game.flagged[y, x]]

        if not covered:
            return False

        # Baue Komponenten basierend auf Nachbarschaft zu sichtbaren Zahlen
        comps = self._connected_components_of_covered(covered)
        for comp in comps:
            if len(comp) > self.MAX_COMPONENT_SIZE:
                continue
            # Wir versuchen alle Kombinationen in dieser Komponente mit genau 'k' Flags,
            # wobei k von 0..remaining durchprobiert wird, aber k begrenzen wir auf <= len(comp)
            comp_list = list(comp)
            for k in range(0, min(remaining, len(comp_list)) + 1):
                for combo in combinations(comp_list, k):
                    test_flags = set((cx, cy) for cy in range(self.H) for cx in range(self.W) if self.game.flagged[cy, cx])
                    test_flags = test_flags.union(set(combo))
                    if self.validate_flag_combination(test_flags):
                        # sichere Felder = covered - test_flags für diese gültige Kombination?
                        # Um wirklich sichere Felder zu identifizieren, bräuchten wir alle gültigen
                        # Kombinationen; hier wenden wir vereinfachend an: wenn es eine gültige
                        # Kombination gibt, können wir daraus *keine* sicheren Felder ableiten.
                        # Stattdessen sammeln wir alle gültigen Kombinationen zuerst:
                        pass
            # Um wirklich sichere Felder aus allen gültigen Kombinationen zu finden,
            # rufen wir die reduzierte Komponentenkombinatorik (apply_reduction_logic) auf.
        return False

    def validate_flag_combination(self, test_flags):
        """Testet, ob die Menge test_flags (Set von (x,y)) zu keinem sichtbaren Feld im Widerspruch steht."""
        # test_flags: set of tuples (x,y)
        for y in range(self.H):
            for x in range(self.W):
                if not self.game.visible[y, x]:
                    continue
                value = int(self.game.board[y, x])
                if value <= 0:
                    continue
                neighbors = self.get_neighbors(x, y)
                # neighbors ist Liste; nun zählen
                count = sum((nx, ny) in test_flags for nx, ny in neighbors)
                covered = [(nx, ny) for nx, ny in neighbors if not self.game.visible[ny, nx] and (nx, ny) not in test_flags]
                # Bedingung: count darf value nicht überschreiten, und selbst mit allen verbliebenen verdeckten
                # Feldern kann man value nicht mehr erreichen -> invalid
                if count > value:
                    return False
                if count + len(covered) < value:
                    return False
        return True

    def _connected_components_of_covered(self, covered):
        """Verbinde verdeckte Felder, wenn sie gemeinsame angrenzende Zahlen haben (Constraint-Komponenten)."""
        covered_set = set(covered)
        # Baue Karte: covered cell -> welche nearby clues (sichtbare Zellen) es beeinflusst
        cell_to_clues = defaultdict(set)
        clues = []
        for y in range(self.H):
            for x in range(self.W):
                if not self.game.visible[y, x]:
                    continue
                val = int(self.game.board[y, x])
                if val <= 0:
                    continue
                clue = (x, y)
                clues.append(clue)
                for nx, ny in self.get_neighbors(x, y):
                    if (nx, ny) in covered_set:
                        cell_to_clues[(nx, ny)].add(clue)

        # Build adjacency: two covered cells are adjacent if they share at least one clue
        adj = defaultdict(set)
        for cell, clueset in cell_to_clues.items():
            for other_cell, other_clueset in cell_to_clues.items():
                if cell == other_cell:
                    continue
                if clueset & other_clueset:
                    adj[cell].add(other_cell)

        # Find components
        visited = set()
        components = []
        for c in covered_set:
            if c in visited:
                continue
            # BFS
            stack = [c]
            comp = set()
            visited.add(c)
            while stack:
                u = stack.pop()
                comp.add(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            components.append(comp)
        # cells that had no clue adjacency (isolated) are their own components
        for c in covered_set:
            if c not in visited:
                components.append({c})
        return components

    def apply_reduction_logic(self):
        """Kombinatorische Logik auf Komponentenebene:
        - Für kleine Komponenten (<= MAX_COMPONENT_SIZE) enumerieren wir alle gültigen Flag-Kombinationen
        - Wenn in allen gültigen Lösungen ein Feld immer Mine (oder immer sicher) ist, wenden wir das an."""
        # Sammle alle verdeckten Felder, gruppiere in Komponenten
        covered = [(x, y) for y in range(self.H) for x in range(self.W)
                   if not self.game.visible[y, x] and not self.game.flagged[y, x]]
        if not covered:
            return False

        comps = self._connected_components_of_covered(covered)
        any_action = False
        for comp in comps:
            if len(comp) == 0 or len(comp) > self.MAX_COMPONENT_SIZE:
                continue
            comp_list = list(comp)
            valid_assignments = []
            current_flags = set((cx, cy) for cy in range(self.H) for cx in range(self.W) if self.game.flagged[cy, cx])

            # We need to find all assignments of flags inside comp that don't contradict clues.
            # Upper bound: at most len(comp) mines there, but we can prune by using adjacent clue counts.
            # Compute upper bound k by checking sum of remaining values among adjacent clues — but keep simple:
            max_k = len(comp_list)
            for k in range(0, max_k + 1):
                # Iterate combinations of positions in comp of size k
                for combo in combinations(comp_list, k):
                    test_flags = current_flags.union(set(combo))
                    if self.validate_flag_combination(test_flags):
                        valid_assignments.append(set(combo))
                # small optimization: if we have many valid assignments, we can stop early if not useful
                if len(valid_assignments) > 2000:
                    break
            if not valid_assignments:
                continue
            # Intersection/Union
            always_mine = set(valid_assignments[0])
            always_safe = set(comp_list) - set(valid_assignments[0])
            for a in valid_assignments[1:]:
                always_mine &= a
                always_safe &= (set(comp_list) - a)
            # Apply deduced flags
            for fx, fy in list(always_mine):
                if not self.game.flagged[fy, fx]:
                    self.game.flag(fx, fy)
                    any_action = True
            for sx, sy in list(always_safe):
                if not self.game.visible[sy, sx] and not self.game.flagged[sy, sx]:
                    hit, _ = self.game.click(sx, sy)
                    if hit:
                        self.game.game_over = True
                    any_action = True
            if any_action:
                return True
        return False


    def apply_pairwise_subset_rule(self):
        """Wenn die verdeckten Felder einer Zahl eine Obermenge der einer anderen Zahl sind,
        kann die Differenz an Feldern als Minen markiert oder als sicher aufgeklärt werden."""
        visible = self.game.visible
        board = self.game.board
        flagged = self.game.flagged

        clues = []
        for y in range(self.H):
            for x in range(self.W):
                if not visible[y, x]:
                    continue
                value = int(board[y, x])
                if value <= 0:
                    continue
                # compute covered neighbors excluding already flagged
                covered = set()
                flag_count = 0
                for nx, ny in self.get_neighbors(x, y):
                    if flagged[ny, nx]:
                        flag_count += 1
                    elif not visible[ny, nx]:
                        covered.add((nx, ny))
                rem = value - flag_count
                clues.append(((x, y), covered, rem))

        # pairwise compare
        for i in range(len(clues)):
            for j in range(len(clues)):
                if i == j:
                    continue
                (ax, ay), A_cov, A_rem = clues[i]
                (bx, by), B_cov, B_rem = clues[j]
                if not A_cov or not B_cov:
                    continue
                # If A_cov is superset of B_cov:
                if A_cov.issuperset(B_cov):
                    diff = A_cov - B_cov
                    if len(diff) == A_rem - B_rem and (A_rem - B_rem) > 0:
                        # diff are mines — validate before setting
                        current_flags = set((cx, cy) for cy in range(self.H) for cx in range(self.W) if self.game.flagged[cy, cx])
                        test_flags = current_flags.union(diff)
                        if self.validate_flag_combination(test_flags):
                            for fx, fy in diff:
                                if not self.game.flagged[fy, fx]:
                                    self.game.flag(fx, fy)
                            return True
                    if B_rem == A_rem and len(A_cov - B_cov) > 0:
                        # A_cov - B_cov are safe
                        safe_cells = A_cov - B_cov
                        for sx, sy in safe_cells:
                            if not self.game.visible[sy, sx] and not self.game.flagged[sy, sx]:
                                hit, _ = self.game.click(sx, sy)
                                if hit:
                                    self.game.game_over = True
                                return True
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
            return True
        return False


# ----------------- GUI -----------------
import pygame
import sys  

class TextBox:
    """Ein einfacher numerischer Text-Input für Pygame."""
    def __init__(self, rect, placeholder=""):
        self.rect = pygame.Rect(rect)
        self.text = ""
        self.placeholder = placeholder
        self.active = False
        self.max_length = 4  # genug für Dimensionen/Minen

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isdigit() and len(self.text) < self.max_length:
                self.text += event.unicode

    def draw(self, surf, font, colors):
        color = (200, 200, 200) if self.active else (160, 160, 160)
        pygame.draw.rect(surf, color, self.rect)
        pygame.draw.rect(surf, (0,0,0), self.rect, 2)
        display = self.text if self.text != "" else self.placeholder
        txt = font.render(display, True, (0,0,0))
        surf.blit(txt, (self.rect.x + 6, self.rect.y + 4))

    def get_int(self, default):
        try:
            val = int(self.text)
            return val
        except:
            return default
class GUI:
    CELL_SIZE = 20

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
        "menu_bg": (40, 40, 40),
    }

    # Standard difficulty presets
    DIFFICULTIES = {
        "Einfach": (9, 9, 10),
        "Mittel": (16, 16, 40),
        "Schwer": (30, 16, 99),
    }

    def __init__(self):
        pygame.init()
        # Startfenstergröße (Menü)
        self.MENU_W, self.MENU_H = 800, 480
        self.screen = pygame.display.set_mode((self.MENU_W, self.MENU_H))
        pygame.display.set_caption("Minesweeper")
        self.font = pygame.font.SysFont(None, 24)
        self.clock = pygame.time.Clock()

        self.state = "menu"  # "menu" oder "playing"
        self.selected_difficulty = "Mittel"
        self.custom_mines_box = TextBox((320, 230, 80, 34), "Minen")
        self.custom_width_box = TextBox((420, 230, 80, 34), "Breite")
        self.custom_height_box = TextBox((520, 230, 80, 34), "Höhe")

        # Default game placeholders
        self.game = None
        self.solver = None
        self.auto_solver = False

        # Buttons definieren (Menu)
        self.diff_buttons = []
        start_x = 150
        start_y = 140
        gap = 160
        for i, name in enumerate(["Einfach", "Mittel", "Schwer", "Custom"]):
            rect = pygame.Rect(start_x + i*gap, start_y, 140, 48)
            self.diff_buttons.append((name, rect))

        self.start_btn = pygame.Rect(330, 300, 140, 44)
        self.quit_btn = pygame.Rect(330, 360, 140, 36)

        # Inits für Spiel-Fenster-Größe
        self.GRID_W = 16
        self.GRID_H = 16
        self.WIDTH = self.GRID_W * self.CELL_SIZE
        self.HEIGHT = self.GRID_H * self.CELL_SIZE + 50

    def create_new_game(self, w=None, h=None, mines=None, preserve_auto=False):
        """
        Erstellt ein neues Spiel.
        Wenn preserve_auto=True, bleibt der aktuelle auto_solver-Status erhalten (nützlich für Neustart).
        Beim Start aus dem Menü wird preserve_auto=False verwendet (auto_solver standardmäßig aus).
        """
        # Merke vorherigen auto_solver, falls preserve gewünscht
        old_auto = self.auto_solver if preserve_auto else False

        if w is None or h is None or mines is None:
            if self.selected_difficulty != "Custom":
                pw, ph, pm = GUI.DIFFICULTIES[self.selected_difficulty]
                w, h, mines = pw, ph, pm
            else:
                # Aus den Textboxen lesen (oder Default fallback)
                mines = self.custom_mines_box.get_int(10)
                w = self.custom_width_box.get_int(9)
                h = self.custom_height_box.get_int(9)

        # Validierung
        w = max(5, min(60, w))
        h = max(5, min(40, h))
        max_mines = max(1, w*h - 1)
        mines = max(1, min(max_mines, mines))

        self.GRID_W = w
        self.GRID_H = h
        self.WIDTH = self.GRID_W * self.CELL_SIZE
        self.HEIGHT = self.GRID_H * self.CELL_SIZE + 50

        # Resize Fenster entsprechend
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption(f"Minesweeper {w}x{h} - {mines} Minen")

        # Neues Game + Solver binden
        self.game = Minesweeper(self.GRID_W, self.GRID_H, mines)
        self.solver = RuleBasedSolver(self.game)

        # Setze auto_solver je nach preserve-Flag zurück oder wiederherstellen
        self.auto_solver = bool(old_auto)

        self.state = "playing"

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

    def draw_menu(self):
        self.screen.fill(self.COLORS["menu_bg"])
        title = self.font.render("Minesweeper - Schwierigkeit wählen", True, (255,255,255))
        self.screen.blit(title, (self.MENU_W//2 - title.get_width()//2, 40))

        # Draw difficulty buttons
        mouse_pos = pygame.mouse.get_pos()
        for name, rect in self.diff_buttons:
            is_hover = rect.collidepoint(mouse_pos)
            color = self.COLORS["button_hover"] if is_hover else self.COLORS["button"]
            if name == self.selected_difficulty:
                # Hervorhebung des aktuell ausgewählten
                pygame.draw.rect(self.screen, (80, 140, 80), rect)
            else:
                pygame.draw.rect(self.screen, color, rect)
            txt = self.font.render(name, True, self.COLORS["button_text"])
            self.screen.blit(txt, (rect.x + 10, rect.y + 12))

        # Wenn Custom ausgewählt: zeige Textfelder
        if self.selected_difficulty == "Custom":
            hint = self.font.render("Gib Custom Werte ein (Ganzzahlen). Validierung erfolgt beim Start.", True, (220,220,220))
            self.screen.blit(hint, (200, 200))
            self.custom_mines_box.draw(self.screen, self.font, self.COLORS)
            self.custom_width_box.draw(self.screen, self.font, self.COLORS)
            self.custom_height_box.draw(self.screen, self.font, self.COLORS)

        # Start / Quit Buttons
        pygame.draw.rect(self.screen, self.COLORS["button"], self.start_btn)
        s_txt = self.font.render("Start", True, self.COLORS["button_text"])
        self.screen.blit(s_txt, (self.start_btn.x + 40, self.start_btn.y + 10))

        pygame.draw.rect(self.screen, self.COLORS["button"], self.quit_btn)
        q_txt = self.font.render("Beenden", True, self.COLORS["button_text"])
        self.screen.blit(q_txt, (self.quit_btn.x + 30, self.quit_btn.y + 6))

    def handle_menu_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            # Difficulty buttons
            for name, rect in self.diff_buttons:
                if rect.collidepoint(pos):
                    self.selected_difficulty = name
            # Start button
            if self.start_btn.collidepoint(pos):
                # Start with selected preset / custom values
                self.create_new_game()
            # Quit
            if self.quit_btn.collidepoint(pos):
                pygame.quit()
                sys.exit()
        # NOTE: TextBox-Events werden nicht hier verarbeitet (verhindert doppelte Eingabe).

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
        if self.state != "playing":
            return
        if self.restart_btn.collidepoint(pos):
            # Neustart mit gleichen Parametern, AUTO-SOLVER-STATUS BEIBEHALTEN
            self.create_new_game(self.GRID_W, self.GRID_H, self.game.N, preserve_auto=True)
            print("Neustart (solver-status beibehalten)!")
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

                if self.state == "menu":
                    # Menu-interaktionen
                    if event.type in (pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN):
                        self.handle_menu_event(event)
                else:
                    # Spielinteraktionen
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.pos[1] >= self.HEIGHT - 50:
                            self.check_ui_click(event.pos)
                        else:
                            self.handle_click(event.pos, event.button)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            if self.solver:
                                self.solver.step()

                # Textboxes werden zentral im Hauptloop verarbeitet (nur einmal).
                if self.state == "menu":
                    if event.type in (pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN):
                        self.custom_mines_box.handle_event(event)
                        self.custom_width_box.handle_event(event)
                        self.custom_height_box.handle_event(event)

            if self.state == "menu":
                self.draw_menu()
            else:
                if self.auto_solver and self.solver:
                    self.solver.step()
                self.screen.fill(self.COLORS["bg"])
                if self.game:
                    self.draw_board()
                self.draw_ui()

            pygame.display.flip()
            self.clock.tick(30)

if __name__ == "__main__":
    GUI().main()
