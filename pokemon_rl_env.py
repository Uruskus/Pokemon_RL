import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from pyboy import PyBoy
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import time
import os

class PokemonRedEnv(gym.Env):
    """
    Pokemon Rot Reinforcement Learning Environment
    """

    def __init__(self, rom_path, render_mode=None, savestate_path=None):
        super().__init__()

        # PyBoy Setup - ROM Pfad speichern für Reset!
        self.rom_path = rom_path
        self.render_mode = render_mode  # Speichere render_mode für reset()
        self.savestate_path = savestate_path  # Optional: Savestate verwenden
        window_type = "SDL2" if render_mode == "human" else "null"
        self.pyboy = PyBoy(rom_path, window=window_type)
        self.pyboy.set_emulation_speed(0)  # Unbegrenzte Geschwindigkeit

        # Savestate sofort beim Start laden falls vorhanden
        if self.savestate_path and os.path.exists(self.savestate_path):
            print(f"Lade initialen Savestate: {self.savestate_path}")
            with open(self.savestate_path, "rb") as f:
                self.pyboy.load_state(f)
            print("✓ Savestate geladen - Spiel bereit!")
        else:
            # Nur wenn kein Savestate: Intro überspringen
            if not self.savestate_path:
                print("Kein Savestate - überspringe Intro...")
                self._skip_intro()
                print("✓ Intro übersprungen!")

        # Action Space: 9 Aktionen (8 Buttons + No-Op)
        # 0: No-Op, 1: A, 2: B, 3: Start, 4: Select, 5: Up, 6: Down, 7: Left, 8: Right
        self.action_space = spaces.Discrete(9)

        # Observation Space: Grayscale Game Boy Screen (144x160)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(144, 160, 1),
            dtype=np.uint8
        )

        # Stats Tracking
        self.stats = {
            'total_steps': 0,
            'total_episodes': 0,
            'total_battles_won': 0,
            'total_battles_lost': 0,
            'current_episode_steps': 0,
            'start_time': datetime.now(),
            'current_episode_start': datetime.now(),
            'pokemon_party': [],
            'player_position': (0, 0),
            'badges': 0,
            'last_reward': 0
        }

        # RAM Adressen für Pokemon Rot (Wichtige Werte)
        self.RAM_ADDRESSES = {
            'player_x': 0xD362,
            'player_y': 0xD361,
            'map_id': 0xD35E,
            'battle_type': 0xD057,
            'party_count': 0xD163,
            'badges': 0xD356,
            'money': 0xD347,  # 3 Bytes BCD
            'num_bag_items': 0xD31D,
            'bag_items': 0xD31E,
            # Pokemon Party
            'party_species': 0xD164,
            'party_level_1': 0xD18C,
            'party_level_2': 0xD1B8,
            'party_level_3': 0xD1E4,
            'party_level_4': 0xD210,
            'party_level_5': 0xD23C,
            'party_level_6': 0xD268,
            # Pokedex
            'pokedex_owned': 0xD2F7,
            'pokedex_seen': 0xD30A,
            # Events (wichtige Story-Flags)
            'event_flags': 0xD747,  # Event-Bitfeld
            # HMs und Field Moves
            'hm_flags': 0xD803,  # Welche HMs wurden benutzt
        }

        # Wichtige Map IDs (für Boosting)
        self.IMPORTANT_MAP_IDS = {
            # Gyms
            54: {'name': 'Pewter Gym', 'boost': 2.0, 'condition': lambda self: self._read_ram(self.RAM_ADDRESSES['badges']) < 1},
            59: {'name': 'Cerulean Gym', 'boost': 2.0, 'condition': lambda self: self._read_ram(self.RAM_ADDRESSES['badges']) < 2},
            192: {'name': 'Vermillion Gym', 'boost': 2.0, 'condition': lambda self: self._read_ram(self.RAM_ADDRESSES['badges']) < 3},
            183: {'name': 'Celadon Gym', 'boost': 2.0, 'condition': lambda self: self._read_ram(self.RAM_ADDRESSES['badges']) < 4},
            182: {'name': 'Fuchsia Gym', 'boost': 2.0, 'condition': lambda self: self._read_ram(self.RAM_ADDRESSES['badges']) < 5},
            214: {'name': 'Saffron Gym', 'boost': 2.0, 'condition': lambda self: self._read_ram(self.RAM_ADDRESSES['badges']) < 6},
            165: {'name': 'Cinnabar Gym', 'boost': 2.0, 'condition': lambda self: self._read_ram(self.RAM_ADDRESSES['badges']) < 7},
            45: {'name': 'Viridian Gym', 'boost': 2.0, 'condition': lambda self: self._read_ram(self.RAM_ADDRESSES['badges']) < 8},
            # Wichtige Story-Orte
            199: {'name': 'Rocket Hideout', 'boost': 1.5, 'condition': lambda self: True},
            181: {'name': 'Silph Co', 'boost': 1.5, 'condition': lambda self: True},
            142: {'name': 'Pokemon Tower', 'boost': 1.5, 'condition': lambda self: True},
            # Safari Zone
            217: {'name': 'Safari Zone', 'boost': 2.0, 'condition': lambda self: True},
        }

        # Reward Konfiguration (OPTIMIERT gegen Kreisen)
        self.reward_config = {
            'exploration': 0.005,         # REDUZIERT! Sonst läuft KI nur im Kreis
            'exploration_boost': 0.01,    # Boost für wichtige Maps
            'new_map': 10.0,              # ERHÖHT - Neue Maps sind wichtig!
            'seen_pokemon': 2.0,          # ERHÖHT - Pokemon sehen ist gut
            'caught_pokemon': 50.0,       # MASSIV ERHÖHT - Das ist das Hauptziel!
            'badges': 500.0,              # MASSIV ERHÖHT - Badges sind kritisch!
            'level': 0.5,                 # REDUZIERT - Nicht nur grinden
            'battle_start': 3.0,          # Leicht reduziert
            'battle_won': 15.0,           # NEU - Kämpfe gewinnen!
            'item': 5.0,                  # ERHÖHT - Items sind wichtig
            'heal': 10.0,                 # Pokemon Center benutzen
            'event': 30.0,                # ERHÖHT - Story-Events wichtig
            'hm_use': 20.0,               # ERHÖHT - HMs sind kritisch für Progress
            'talked_to_npc': 1.0,         # NEU - Mit NPCs reden
            'menu_opened': 0.5,           # NEU - Menüs öffnen lernen
            'stillstand_penalty': -1.0,   # ERHÖHT - Stillstand ist schlecht
            'revisit_penalty': -0.1,      # NEU - Zu oft gleiche Zone
            'time_penalty': -0.001,       # NEU - Zeit-Druck
        }

        # Previous state für Reward Calculation
        self.prev_state = {
            'position': (0, 0),
            'map_id': 0,
            'party_count': 0,
            'badges': 0,
            'battle_type': 0,
            'money': 0,
            'party_levels': [0, 0, 0, 0, 0, 0],
            'pokedex_seen': set(),
            'pokedex_owned': set(),
            'num_items': 0,
            'event_count': 0,
            'hm_flags': 0,
            'in_battle': False,
            'battle_outcome': 0,
        }

        # Visited zones für Exploration-Bonus (per Episode)
        self._visited_zones = set()
        self._zone_visit_count = {}  # Wie oft jede Zone besucht wurde

        # Global visited mask (über alle Episodes)
        self._global_visited_map = {}  # {map_id: set of (x, y)}

        # Stats für Tracking
        self.total_pokemon_seen = 0
        self.total_pokemon_caught = 0
        self.total_events_completed = 0
        self.episode_step_count = 0  # Für Time-Penalty

        # Stats Window
        self.stats_window = None
        self.stats_thread = None

    def _get_screen(self):
        """
        Holt den aktuellen Bildschirm als NumPy Array mit 2 Channels:
        - Channel 1: Grayscale Screen
        - Channel 2: Visited Mask (zeigt wo KI schon war)
        """
        # Channel 1: Normaler Bildschirm
        screen = self.pyboy.screen.image
        screen = np.array(screen)
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

        # Channel 2: Visited Mask erstellen
        map_id = self._read_ram(self.RAM_ADDRESSES['map_id'])
        visited_mask = np.zeros((144, 160), dtype=np.uint8)

        # Markiere besuchte Bereiche auf aktueller Map
        if map_id in self._global_visited_map:
            for (vx, vy) in self._global_visited_map[map_id]:
                # Zeichne 8x8 Block für besuchte Tile
                x_start = vx * 8
                y_start = vy * 8
                x_end = min(x_start + 8, 160)
                y_end = min(y_start + 8, 144)
                visited_mask[y_start:y_end, x_start:x_end] = 255

        # Kombiniere beide Channels
        screen = np.stack([screen, visited_mask], axis=-1)
        return screen

    def _read_ram(self, address):
        """Liest einen Wert aus dem RAM"""
        return self.pyboy.memory[address]  # FIXED: memory[address] statt get_memory_value

    def _read_bit(self, address, bit):
        """Liest ein spezifisches Bit aus einer RAM-Adresse"""
        value = self._read_ram(address)
        return (value >> bit) & 1

    def _count_events_completed(self):
        """Zählt abgeschlossene Story-Events (vereinfacht)"""
        # Event Flags sind Bitfelder - jedes Bit = ein Event
        # Für Pokemon Red gibt es ~80 required events
        event_count = 0
        for i in range(32):  # Erste 32 Bytes der Event Flags
            byte_val = self._read_ram(self.RAM_ADDRESSES['event_flags'] + i)
            # Zähle gesetzte Bits
            event_count += bin(byte_val).count('1')
        return event_count

    def _get_hm_usage_count(self):
        """Zählt wie oft HMs benutzt wurden"""
        hm_flags = self._read_ram(self.RAM_ADDRESSES['hm_flags'])
        return bin(hm_flags).count('1')

    def _get_party_levels(self):
        """Liest alle Party Pokemon Level"""
        levels = []
        party_count = self._read_ram(self.RAM_ADDRESSES['party_count'])
        level_addrs = [
            self.RAM_ADDRESSES['party_level_1'],
            self.RAM_ADDRESSES['party_level_2'],
            self.RAM_ADDRESSES['party_level_3'],
            self.RAM_ADDRESSES['party_level_4'],
            self.RAM_ADDRESSES['party_level_5'],
            self.RAM_ADDRESSES['party_level_6'],
        ]
        for i in range(min(party_count, 6)):
            levels.append(self._read_ram(level_addrs[i]))
        while len(levels) < 6:
            levels.append(0)
        return levels

    def _get_pokedex_stats(self):
        """Liest Pokedex Seen und Owned"""
        seen = set()
        owned = set()

        # Pokedex ist bitfield - 151 Pokemon
        for i in range(19):  # 19 Bytes für 151 Pokemon
            seen_byte = self._read_ram(self.RAM_ADDRESSES['pokedex_seen'] + i)
            owned_byte = self._read_ram(self.RAM_ADDRESSES['pokedex_owned'] + i)

            for bit in range(8):
                if seen_byte & (1 << bit):
                    seen.add(i * 8 + bit)
                if owned_byte & (1 << bit):
                    owned.add(i * 8 + bit)

        return seen, owned

    def _calculate_level_reward(self, levels):
        """
        Level Reward Formel aus PokeRL:
        - Early game: Summe aller Level
        - Late game: Reduced scaling um Exploration zu fördern
        """
        total_level = sum(levels)

        if total_level < 15:
            return total_level
        else:
            return 30 + (total_level - 15) / 4.0

    def _get_pokemon_party(self):
        """Liest Pokemon Party aus RAM"""
        party = []
        party_count = self._read_ram(self.RAM_ADDRESSES['party_count'])

        # Pokemon Party startet bei 0xD164
        for i in range(min(party_count, 6)):
            species_addr = 0xD164 + i
            species_id = self._read_ram(species_addr)

            # Level auslesen (0xD18C + i * 0x2C)
            level_addr = 0xD18C + (i * 44)
            level = self._read_ram(level_addr)

            # HP auslesen (Current HP: 2 Bytes)
            hp_addr = 0xD16C + (i * 44)
            current_hp = (self._read_ram(hp_addr) << 8) | self._read_ram(hp_addr + 1)
            max_hp = (self._read_ram(hp_addr + 2) << 8) | self._read_ram(hp_addr + 3)

            party.append({
                'species_id': species_id,
                'level': level,
                'hp': current_hp,
                'max_hp': max_hp
            })

        return party

    def _calculate_reward(self):
        """Berechnet die Belohnung basierend auf Spielfortschritt"""
        reward = 0

        # Aktuelle Position
        x = self._read_ram(self.RAM_ADDRESSES['player_x'])
        y = self._read_ram(self.RAM_ADDRESSES['player_y'])
        map_id = self._read_ram(self.RAM_ADDRESSES['map_id'])
        party_count = self._read_ram(self.RAM_ADDRESSES['party_count'])
        badges = self._read_ram(self.RAM_ADDRESSES['badges'])

        # Reward für Bewegung (Exploration)
        if (x, y) != self.prev_state['position']:
            reward += 0.1

        # Reward für neue Map
        if map_id != self.prev_state['map_id']:
            reward += 5.0

        # Reward für neues Pokemon
        if party_count > self.prev_state['party_count']:
            reward += 20.0
            self.stats['total_battles_won'] += 1

        # Reward für Badge
        if badges > self.prev_state['badges']:
            reward += 100.0

        # Penalty für Stillstand
        if (x, y) == self.prev_state['position'] and map_id == self.prev_state['map_id']:
            reward -= 0.1  # Stärkere Penalty

        # Update previous state
        self.prev_state = {
            'position': (x, y),
            'map_id': map_id,
            'party_count': party_count,
            'badges': badges
        }

        return reward

    def _skip_intro(self):
        """Helper-Funktion: Überspringt das Intro"""
        # Phase 1: Title Screen überspringen
        for _ in range(5):
            self.pyboy.button_press('start')
            for _ in range(30): self.pyboy.tick()
            self.pyboy.button_release('start')
            for _ in range(30): self.pyboy.tick()

        # Phase 2: A spammen
        for _ in range(100):
            self.pyboy.button_press('a')
            for _ in range(10): self.pyboy.tick()
            self.pyboy.button_release('a')
            for _ in range(10): self.pyboy.tick()

        # Phase 3: warten
        for _ in range(1000):
            self.pyboy.tick()

    def reset(self, seed=None, options=None):
        """Startet eine neue Episode"""
        super().reset(seed=seed)

        # Reset Emulator (lädt ROM neu)
        self.pyboy.stop()
        window_type = "SDL2" if self.render_mode == "human" else "null"
        self.pyboy = PyBoy(self.rom_path, window=window_type)
        self.pyboy.set_emulation_speed(0)

        # ===== INTRO ÜBERSPRINGEN =====
        print("Überspringe Intro...")

        # Phase 1: Title Screen überspringen (Start drücken)
        for _ in range(5):
            self.pyboy.button_press('start')
            for _ in range(30):
                self.pyboy.tick()
            self.pyboy.button_release('start')
            for _ in range(30):
                self.pyboy.tick()

        # Phase 2: Intro-Sequenz und Dialoge überspringen (A spammen)
        for _ in range(100):
            self.pyboy.button_press('a')
            for _ in range(10):
                self.pyboy.tick()
            self.pyboy.button_release('a')
            for _ in range(10):
                self.pyboy.tick()

        # Phase 3: Warten bis Spiel wirklich startet
        for _ in range(1000):
            self.pyboy.tick()

        print("Intro übersprungen! Spiel läuft.")
        # ===== ENDE INTRO SKIP =====

        # Stats Update
        self.stats['total_episodes'] += 1
        self.stats['current_episode_steps'] = 0
        self.stats['current_episode_start'] = datetime.now()

        observation = self._get_screen()
        info = {}

        return observation, info

    def step(self, action):
        """Führt eine Aktion aus"""
        # Button Mapping
        button_map = {
            0: None,  # No-Op
            1: 'a',
            2: 'b',
            3: 'start',
            4: 'select',
            5: 'up',
            6: 'down',
            7: 'left',
            8: 'right'
        }

        # Button drücken
        if action != 0:
            self.pyboy.button_press(button_map[action])

        # Emulation für einige Frames
        for _ in range(8):  # 8 Frames pro Aktion
            self.pyboy.tick()

        # Button loslassen
        if action != 0:
            self.pyboy.button_release(button_map[action])

        # Stats Update
        self.stats['total_steps'] += 1
        self.stats['current_episode_steps'] += 1

        # Observation, Reward, Done
        observation = self._get_screen()
        reward = self._calculate_reward()
        self.stats['last_reward'] = reward

        # Update Pokemon Party
        self.stats['pokemon_party'] = self._get_pokemon_party()

        # Update Position & Badges
        x = self._read_ram(self.RAM_ADDRESSES['player_x'])
        y = self._read_ram(self.RAM_ADDRESSES['player_y'])
        self.stats['player_position'] = (x, y)
        self.stats['badges'] = self._read_ram(self.RAM_ADDRESSES['badges'])

        # Episode beenden - DYNAMISCHE LÄNGE basierend auf Progress
        # Early Training: Kurze Episodes (schnelles Reset)
        # Late Training: Längere Episodes (mehr Zeit für Progress)
        base_episode_length = 10000
        badge_bonus = badges * 5000  # +5k Steps pro Badge
        max_episode_length = min(base_episode_length + badge_bonus, 50000)

        done = self.stats['current_episode_steps'] >= max_episode_length
        truncated = False

        info = {}

        return observation, reward, done, truncated, info

    def render(self):
        """Rendert das Spiel (optional)"""
        pass

    def close(self):
        """Schließt die Umgebung"""
        self.pyboy.stop()
        if self.stats_window:
            self.stats_window.destroy()

    def start_stats_window(self):
        """Startet das Stats-Fenster in separatem Thread"""
        self.stats_thread = threading.Thread(target=self._run_stats_window, daemon=True)
        self.stats_thread.start()

    def _run_stats_window(self):
        """Erstellt und aktualisiert das Stats-Fenster"""
        self.stats_window = tk.Tk()
        self.stats_window.title("Pokemon RL Training Stats")
        self.stats_window.geometry("500x600")

        # Style
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Stats.TLabel', font=('Arial', 10))

        # Main Frame
        main_frame = ttk.Frame(self.stats_window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Labels
        ttk.Label(main_frame, text="🎮 Pokemon RL Training", style='Title.TLabel').grid(row=0, column=0, columnspan=2, pady=10)

        # Stats Labels
        labels = {}
        row = 1

        stats_to_show = [
            ('Gesamtzeit', 'total_time'),
            ('Episode Zeit', 'episode_time'),
            ('Total Steps', 'total_steps'),
            ('Episode Steps', 'current_episode_steps'),
            ('Total Episodes', 'total_episodes'),
            ('Kämpfe Gewonnen', 'total_battles_won'),
            ('Kämpfe Verloren', 'total_battles_lost'),
            ('Badges', 'badges'),
            ('Position', 'player_position'),
            ('Letzter Reward', 'last_reward')
        ]

        for label_text, key in stats_to_show:
            ttk.Label(main_frame, text=f"{label_text}:", style='Stats.TLabel').grid(row=row, column=0, sticky=tk.W, pady=2)
            labels[key] = ttk.Label(main_frame, text="0", style='Stats.TLabel')
            labels[key].grid(row=row, column=1, sticky=tk.W, pady=2)
            row += 1

        # Pokemon Party Frame
        ttk.Label(main_frame, text="Pokemon Party:", style='Title.TLabel').grid(row=row, column=0, columnspan=2, pady=10)
        row += 1

        pokemon_frame = ttk.Frame(main_frame)
        pokemon_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E))

        pokemon_text = tk.Text(pokemon_frame, height=10, width=50, font=('Arial', 9))
        pokemon_text.pack()

        # Update Loop
        def update_stats():
            # Zeit Berechnung
            total_time = datetime.now() - self.stats['start_time']
            episode_time = datetime.now() - self.stats['current_episode_start']

            labels['total_time'].config(text=str(total_time).split('.')[0])
            labels['episode_time'].config(text=str(episode_time).split('.')[0])
            labels['total_steps'].config(text=f"{self.stats['total_steps']:,}")
            labels['current_episode_steps'].config(text=f"{self.stats['current_episode_steps']:,}")
            labels['total_episodes'].config(text=f"{self.stats['total_episodes']:,}")
            labels['total_battles_won'].config(text=f"{self.stats['total_battles_won']:,}")
            labels['total_battles_lost'].config(text=f"{self.stats['total_battles_lost']:,}")
            labels['badges'].config(text=f"{self.stats['badges']}/8")
            labels['player_position'].config(text=f"{self.stats['player_position']}")
            labels['last_reward'].config(text=f"{self.stats['last_reward']:.2f}")

            # Pokemon Party
            pokemon_text.delete(1.0, tk.END)
            if self.stats['pokemon_party']:
                for i, pokemon in enumerate(self.stats['pokemon_party'], 1):
                    pokemon_text.insert(tk.END,
                        f"#{i}: ID {pokemon['species_id']} | Lv.{pokemon['level']} | "
                        f"HP: {pokemon['hp']}/{pokemon['max_hp']}\n")
            else:
                pokemon_text.insert(tk.END, "Keine Pokemon gefangen")

            self.stats_window.after(100, update_stats)

        update_stats()
        self.stats_window.mainloop()


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    # Environment erstellen
    env = PokemonRedEnv(rom_path="pokemon_red.gb")

    # Stats Window starten
    env.start_stats_window()

    # Training Loop (Beispiel mit Random Actions)
    for episode in range(10):
        observation, info = env.reset()
        done = False

        while not done:
            # Random Action (später: Model Prediction)
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)

            time.sleep(0.01)  # Für Visualisierung

    env.close()
