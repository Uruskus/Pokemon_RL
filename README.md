[INFO]
Übersetzung der kompletten Datei ins Englische. Inhalt unverändert, nur Sprache konvertiert.

---

## ENGLISH TRANSLATION

# 🎮 Pokemon Red RL Agent

A reinforcement learning agent that learns to play Pokémon Red and become the Pokémon League Champion.

## 📋 Table of Contents

* [About the Project](#about-the-project)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [How It Works](#how-it-works)
* [Training Configuration](#training-configuration)
* [Troubleshooting](#troubleshooting)
* [Roadmap](#roadmap)

## 🎯 About the Project

This project uses **Reinforcement Learning** (specifically the PPO algorithm) to train an AI that plays Pokémon Red autonomously. The AI learns through trial and error, receiving rewards for progress (exploring new areas, catching Pokémon, winning badges).

### Technology Stack

* **Python 3.13**
* **PyBoy** – Game Boy emulator
* **Stable-Baselines3** – Reinforcement learning framework (PPO)
* **Gymnasium** – RL environment standard
* **OpenCV** – Image processing
* **PyTorch** – Deep learning backend

## ✨ Features

✅ **Automatic intro skipping** – starts directly in the game
✅ **Intelligent reward system** – exploration, Pokémon catches, badges
✅ **Checkpoint system** – saves automatically every 50,000 steps
✅ **Resume function** – training can continue after interruption
✅ **RAM-based rewards** – precise progress detection via GB RAM
✅ **Headless training** – no GUI for maximum performance
✅ **Watch mode** – see the AI play live

## 📦 Installation

### Requirements

* **Linux** (tested on Arch Linux)
* **Python 3.13+**
* **CUDA** (optional, for GPU training)
* Pokémon Red ROM file (legally only with original cartridge)

### Step 1: Clone repository

```bash
git clone https://github.com/Uruskus/Pokemon_RL.git
cd Pokemon_RL
```

### Step 2: Create virtual environment

```bash
python -m venv venv
source venv/bin/activate.fish  # fish shell
# or
source venv/bin/activate       # bash/zsh
```

### Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install gymnasium pyboy opencv-python numpy stable-baselines3 torch tensorboard
```

**Optional GPU support (NVIDIA):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Add ROM file

Place your Pokémon Red ROM in the project folder:

```
Pokemon_RL/
├── pokemon_red.gb  ← HERE!
├── pokemon_rl_env.py
├── train.py
└── watch.py
```

**Important:** You must legally own an original Pokémon Red cartridge.

## 🚀 Usage

### Start training

```bash
python train.py
```

**Training:**

* Runs for **10 million steps** (hours/days)
* Saves every 50k steps
* Can be paused with `CTRL+C`
* Automatically resumes on restart

**Example output:**

```
==================================================
ITERATION 1/10
Progress: 0.0% | Steps: 0/10,000,000
==================================================

fps: 300 | ep_rew_mean: 1810 | total_timesteps: 136,793
```

### Pause training

```bash
CTRL+C
```

Auto-saves as `models/pokemon_model_backup.zip`.

### Resume training

```bash
python train.py
```

Loads `models/pokemon_model_latest.zip` automatically.

### Watch the AI play

```bash
python watch.py
```

What happens:

* Loads trained model
* Opens a window with the game
* AI plays live
* Console shows actions + rewards

Cancel: `CTRL+C`

## 📁 Project Structure

```
Pokemon_RL/
├── pokemon_rl_env.py
├── train.py
├── watch.py
├── pokemon_red.gb
├── .gitignore
├── README.md
│
├── models/
│   ├── pokemon_model_latest.zip
│   ├── pokemon_model_1m_steps.zip
│   ├── pokemon_model_2m_steps.zip
│   └── pokemon_checkpoint_*.zip
│
└── logs/
    └── PPO_0/
```

## 🧠 How It Works

### 1. Environment (`pokemon_rl_env.py`)

The AI interacts with the game through a **Gymnasium environment**.

**Observations (what the AI sees):**

* Grayscale screenshot (144×160 px)
* Optional: RAM values (position, Pokémon, badges)

**Actions (what the AI can do):**

* 0: No-op
* 1: A
* 2: B
* 3: Start
* 4: Select
* 5: Up
* 6: Down
* 7: Left
* 8: Right

**Rewards:**

* +0.1 for movement
* +5.0 for new map
* +20.0 for catching Pokémon
* +100.0 for a new badge
* -0.1 for standing still

### 2. Training Algorithm (PPO)

* Modern RL algorithm
* Stable and efficient
* Learns from past episodes
* Uses a CNN to process images

### 3. Training Loop

```
1. Start episode
2. Auto-skip intro
3. For 10,000 steps:
   - AI selects action
   - Game executes action
   - AI receives reward
   - AI updates model
4. Reset episode
5. Repeat with improved policy
```

After millions of steps the AI learns:

* Navigation
* Menu interactions
* NPC interactions
* Catching and training Pokémon
* Beating gym leaders

## ⚙️ Training Configuration

Edit `train.py` to adjust:

```python
total_steps = 10_000_000
save_freq = 50000

model = PPO(
    "CnnPolicy",
    env,
    n_steps=2048,
    batch_size=64,
    learning_rate=0.0003,
    n_epochs=10,
)
```

### Episode length

`pokemon_rl_env.py` ~line 258:

```python
done = self.stats['current_episode_steps'] >= 10000
```

Change to:

```python
done = self.stats['current_episode_steps'] >= 50000
```

### Reward system

In `_calculate_reward()`:

```python
if map_id != self.prev_state['map_id']:
    reward += 10.0
```

## 🔧 Troubleshooting

### "ImportError: libtk8.6.so"

```
sudo pacman -S tk python-tk
```

### ROM not found

```
ls -la pokemon_red.gb
```

### Training slow

* Use GPU
* Reduce `n_steps` to 1024
* Close other programs

### AI not progressing

Possible reasons:

* Too little training
* Unbalanced rewards
* Episode too short

Fix example:

```python
if party_count > self.prev_state['party_count']:
    reward += 50.0
```

### CUDA out of memory

```python
batch_size = 32
```

## 🗺️ Roadmap

### Completed

* Basic environment
* Intro skip
* Reward system
* Checkpoints
* Training & watch scripts

### In progress

* 10M step training
* Hyperparameter tuning
* Reward optimization

### Planned

* Savestates
* Curriculum learning
* Multi-environment
* Web dashboard
* Pre-trained model
* Twitch integration

## 📊 Expected Results

| Steps | Expected Behavior                 |
| ----- | --------------------------------- |
| 100k  | Random wandering                  |
| 1M    | Leaves house, explores first town |
| 5M    | Reaches cities, catches Pokémon   |
| 10M   | Beats first gym                   |
| 50M+  | Multiple badges, strategies       |

## 📝 License

For educational purposes. Pokémon is a trademark of Nintendo/Game Freak.
You must own an original cartridge to use the ROM legally.

## 🤝 Contributing

1. Fork repo
2. Create feature branch
3. Commit changes
4. Push
5. Open pull request

## 📧 Contact

Open an issue on GitHub for questions.

---

**Good luck training your Pokémon AI!** 🎮🤖

