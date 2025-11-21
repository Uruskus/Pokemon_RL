# 🎮 Pokemon Red RL Agent

Ein Reinforcement Learning Agent, der lernt, Pokemon Rot zu spielen und zum Pokemon-Liga Champion zu werden.

## 📋 Inhaltsverzeichnis

- [Über das Projekt](#über-das-projekt)
- [Features](#features)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Projektstruktur](#projektstruktur)
- [Wie funktioniert es?](#wie-funktioniert-es)
- [Training Konfiguration](#training-konfiguration)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

## 🎯 Über das Projekt

Dieses Projekt nutzt **Reinforcement Learning** (speziell den PPO-Algorithmus), um eine KI zu trainieren, die Pokemon Rot selbstständig spielt. Die KI lernt durch Trial-and-Error, indem sie Belohnungen für Fortschritte erhält (neue Gebiete erkunden, Pokemon fangen, Badges gewinnen).

### Technologie-Stack

- **Python 3.13**
- **PyBoy** - Game Boy Emulator
- **Stable-Baselines3** - Reinforcement Learning Framework (PPO)
- **Gymnasium** - RL Environment Standard
- **OpenCV** - Bildverarbeitung
- **PyTorch** - Deep Learning Backend

## ✨ Features

✅ **Automatisches Intro-Überspringen** - Startet direkt im Spiel  
✅ **Intelligentes Reward-System** - Belohnungen für Exploration, Pokemon fangen, Badges  
✅ **Checkpoint-System** - Speichert automatisch alle 50.000 Steps  
✅ **Resume-Funktion** - Training fortsetzbar nach Unterbrechung  
✅ **RAM-basierte Belohnungen** - Präzise Fortschrittserkennung über Game Boy RAM  
✅ **Headless Training** - Training ohne GUI für maximale Performance  
✅ **Watch Mode** - KI beim Spielen live zuschauen  

## 📦 Installation

### Voraussetzungen

- **Linux** (getestet auf Arch Linux)
- **Python 3.13+**
- **CUDA** (optional, für GPU-Training)
- Pokemon Red ROM-Datei (legal nur mit Original-Spielmodul)

### Schritt 1: Repository klonen

```bash
git clone https://github.com/Uruskus/Pokemon_RL.git
cd Pokemon_RL
```

### Schritt 2: Virtuelle Umgebung erstellen

```bash
python -m venv venv
source venv/bin/activate.fish  # für fish shell
# oder
source venv/bin/activate       # für bash/zsh
```

### Schritt 3: Dependencies installieren

```bash
pip install --upgrade pip
pip install gymnasium pyboy opencv-python numpy stable-baselines3 torch tensorboard
```

**Optional für GPU-Support (NVIDIA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Schritt 4: ROM-Datei hinzufügen

Platziere deine Pokemon Red ROM-Datei im Projektordner:
```
Pokemon_RL/
├── pokemon_red.gb  ← Hier!
├── pokemon_rl_env.py
├── train.py
└── watch.py
```

**Wichtig:** Du musst eine Original-Pokemon-Rot-Kassette besitzen, um die ROM legal zu verwenden.

## 🚀 Verwendung

### Training starten

```bash
python train.py
```

**Das Training:**
- Läuft für **10 Millionen Steps** (dauert mehrere Stunden/Tage)
- Speichert automatisch alle 50.000 Steps
- Kann jederzeit mit `CTRL+C` pausiert werden
- Fortsetzung beim Neustart automatisch

**Ausgabe:**
```
==================================================
ITERATION 1/10
Progress: 0.0% | Steps: 0/10,000,000
==================================================

fps: 300 | ep_rew_mean: 1810 | total_timesteps: 136,793
```

### Training pausieren

```bash
# Im laufenden Training:
CTRL+C
```

Das Model wird automatisch als `models/pokemon_model_backup.zip` gespeichert.

### Training fortsetzen

```bash
python train.py
```

Das Script lädt automatisch das letzte Model (`models/pokemon_model_latest.zip`).

### KI beim Spielen zuschauen

```bash
python watch.py
```

**Was passiert:**
- Lädt das trainierte Model
- Öffnet ein Fenster mit dem Spiel
- Die KI spielt live vor deinen Augen
- Console zeigt Actions und Rewards

**Abbrechen:** `CTRL+C`

## 📁 Projektstruktur

```
Pokemon_RL/
├── pokemon_rl_env.py      # Gymnasium Environment (Spiel-Interface)
├── train.py               # Training Script
├── watch.py               # Visualisierung Script
├── pokemon_red.gb         # Pokemon Red ROM (nicht im Git!)
├── .gitignore            # Ignorierte Dateien
├── README.md             # Diese Datei
│
├── models/               # Trainierte Models (automatisch erstellt)
│   ├── pokemon_model_latest.zip
│   ├── pokemon_model_1m_steps.zip
│   ├── pokemon_model_2m_steps.zip
│   └── pokemon_checkpoint_*.zip
│
└── logs/                 # Tensorboard Logs (automatisch erstellt)
    └── PPO_0/
```

## 🧠 Wie funktioniert es?

### 1. Environment (`pokemon_rl_env.py`)

Die KI interagiert mit dem Spiel über ein **Gymnasium Environment**:

**Observations (Was sieht die KI?):**
- Grayscale Screenshot (144x160 Pixel)
- Optional: RAM-Werte (Position, Pokemon, Badges)

**Actions (Was kann die KI tun?):**
- 0: No-Op (nichts)
- 1: A
- 2: B
- 3: Start
- 4: Select
- 5: Up
- 6: Down
- 7: Left
- 8: Right

**Rewards (Wofür bekommt die KI Punkte?):**
- +0.1 für Bewegung (Exploration)
- +5.0 für neue Map
- +20.0 für neues Pokemon gefangen
- +100.0 für neuen Badge
- -0.1 für Stillstand (Penalty)

### 2. Training Algorithm (PPO)

**Proximal Policy Optimization (PPO):**
- Moderner RL-Algorithmus
- Stabil und effizient
- Lernt aus vergangenen Episoden
- Verwendet CNN (Convolutional Neural Network) für Bildverarbeitung

### 3. Training Loop

```
1. Start Episode
2. Überspringe Intro automatisch
3. Für 10.000 Steps:
   - KI wählt Action
   - Spiel führt Action aus
   - KI bekommt Reward
   - KI lernt aus Erfahrung
4. Episode beendet → Reset
5. Wiederhole mit besserer Policy
```

Nach Millionen von Steps lernt die KI:
- Menüs zu navigieren
- Zielgerichtet zu laufen
- Mit NPCs zu interagieren
- Pokemon zu fangen und zu trainieren
- Arenen zu besiegen

## ⚙️ Training Konfiguration

### Training-Parameter anpassen

In `train.py` kannst du folgendes ändern:

```python
# Gesamt-Steps
total_steps = 10_000_000  # Standard: 10 Millionen

# Checkpoint-Frequenz
save_freq=50000  # Speichert alle 50k Steps

# PPO Hyperparameter
model = PPO(
    "CnnPolicy",
    env,
    n_steps=2048,        # Steps pro Update
    batch_size=64,       # Batch Size
    learning_rate=0.0003,  # Learning Rate
    n_epochs=10,         # Epochs pro Update
)
```

### Episode-Länge ändern

In `pokemon_rl_env.py` Zeile ~258:

```python
# Aktuelle Einstellung: 10.000 Steps pro Episode
done = self.stats['current_episode_steps'] >= 10000

# Für längere Episodes:
done = self.stats['current_episode_steps'] >= 50000
```

### Reward-System anpassen

In `pokemon_rl_env.py` in der `_calculate_reward()` Funktion:

```python
# Beispiel: Höhere Belohnung für neue Maps
if map_id != self.prev_state['map_id']:
    reward += 10.0  # Statt 5.0
```

## 🔧 Troubleshooting

### Problem: "ImportError: libtk8.6.so"

**Lösung:**
```bash
sudo pacman -S tk python-tk
```

### Problem: "ROM nicht gefunden"

**Lösung:**
```bash
# Prüfe ob ROM im richtigen Ordner ist
ls -la pokemon_red.gb

# Stelle sicher der Name exakt "pokemon_red.gb" ist
```

### Problem: Training zu langsam

**Lösungen:**
- Nutze GPU (CUDA):
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```
- Reduziere `n_steps` in `train.py` von 2048 auf 1024
- Schließe andere Programme

### Problem: KI macht keinen Fortschritt

**Mögliche Gründe:**
- **Zu wenig Training** - Mindestens 1-5 Millionen Steps nötig
- **Reward-System unbalanciert** - Mehr Belohnungen für Zwischenziele
- **Episode zu kurz** - Erhöhe auf 50.000 Steps

**Verbesserungen:**
```python
# In pokemon_rl_env.py - Stärkere Rewards
if party_count > self.prev_state['party_count']:
    reward += 50.0  # Statt 20.0
```

### Problem: "CUDA out of memory"

**Lösung:**
```bash
# Reduziere Batch Size in train.py
batch_size=32  # Statt 64
```

## 🗺️ Roadmap

### ✅ Abgeschlossen
- [x] Basic Environment Setup
- [x] Intro-Skip Automatisierung
- [x] Reward-System Implementation
- [x] Checkpoint-System
- [x] Training & Watch Scripts

### 🚧 In Arbeit
- [ ] 10 Millionen Steps Training
- [ ] Hyperparameter Tuning
- [ ] Reward-System Optimierung

### 📝 Geplant
- [ ] Savestate-System (statt immer vom Anfang)
- [ ] Curriculum Learning (schrittweise schwieriger)
- [ ] Multi-Environment Training (mehrere Instanzen parallel)
- [ ] Web-Dashboard für Live-Monitoring
- [ ] Pre-trained Model zum Download
- [ ] Twitch-Integration (Live-Stream der KI)

## 📊 Erwartete Ergebnisse

**Nach verschiedenen Training-Stufen:**

| Steps | Erwartetes Verhalten |
|-------|---------------------|
| 100k | Zufälliges Herumlaufen, manchmal Menüs öffnen |
| 1M | Verlässt das erste Haus, erkundet Alabastia |
| 5M | Erreicht erste Stadt, fängt Pokemon |
| 10M | Kann erste Arena herausfordern |
| 50M+ | Mehrere Badges, strategisches Kämpfen |

## 📝 Lizenz

Dieses Projekt ist für Bildungszwecke. Pokemon ist ein Trademark von Nintendo/Game Freak.

**ROM-Hinweis:** Du musst eine originale Pokemon Red Kassette besitzen, um die ROM legal zu verwenden.

## 🤝 Contributing

Contributions sind willkommen! Bitte:
1. Fork das Repository
2. Erstelle einen Feature Branch
3. Commit deine Changes
4. Push zum Branch
5. Öffne einen Pull Request

## 📧 Kontakt

Bei Fragen oder Problemen öffne ein Issue auf GitHub!

---

**Viel Erfolg beim Training deiner Pokemon-KI!** 🎮🤖
