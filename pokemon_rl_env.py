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

class PokemonRedEnv(gym.Env):
    """
    Pokemon Rot Reinforcement Learning Environment
    """

    def __init__(self, rom_path, render_mode=None):
        super().__init__()

        # PyBoy Setup - ROM Pfad speichern für Reset!
        self.rom_path = rom_path
        self.render_mode = render_mode  # Speichere render_mode für reset()
        window_type = "SDL2" if render_mode == "human" else "null"
        self.pyboy = PyBoy(rom_path, window=window_type)
        self.pyboy.set_emulation_speed(0)  # Unbegrenzte Geschwindigkeit

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
            'money': 0xD347  # 3 Bytes BCD
        }

        # Previous state für Reward Calculation
        self.prev_state = {
            'position': (0, 0),
            'map_id': 0,
            'party_count': 0,
            'badges': 0
        }

        # Stats Window
        self.stats_window = None
        self.stats_thread = None

    def _get_screen(self):
        """Holt den aktuellen Bildschirm als NumPy Array"""
        screen = self.pyboy.screen.image  # FIXED: screen.image statt screen_image()
        screen = np.array(screen)
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        screen = np.expand_dims(screen, axis=-1)
        return screen

    def _read_ram(self, address):
        """Liest einen Wert aus dem RAM"""
        return self.pyboy.memory[address]  # FIXED: memory[address] statt get_memory_value

    def _read_bit(self, address, bit):
        """Liest ein spezifisches Bit aus einer RAM-Adresse"""
        value = self._read_ram(address)
        return (value >> bit) & 1

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

        # Episode beenden nach X Steps (oder andere Bedingung)
        done = self.stats['current_episode_steps'] >= 10000
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
