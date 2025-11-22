from pokemon_rl_env import PokemonRedEnv
from stable_baselines3 import PPO
import time
import os

print("=" * 60)
print("POKEMON RL - WATCH MODE")
print("=" * 60)

# Savestate Konfiguration (optional)
SAVESTATE_NAME = None  # Setze z.B. "start" um von Savestate zu starten
# SAVESTATE_NAME = "start"  # Aktiviere diese Zeile

if SAVESTATE_NAME:
    savestate_path = f"savestates/{SAVESTATE_NAME}.state"
    if os.path.exists(savestate_path):
        print(f"\n✓ Nutze Savestate: {savestate_path}")
    else:
        print(f"\n⚠️  Savestate nicht gefunden: {savestate_path}")
        savestate_path = None
else:
    savestate_path = None

print("\nLade Environment...")
env = PokemonRedEnv(
    rom_path="pokemon_red.gb",
    render_mode="human",  # Mit Fenster
    savestate_path=savestate_path
)

print("Lade trainiertes Model...")
try:
    model = PPO.load("models/pokemon_model_latest.zip")
    print("✓ Model erfolgreich geladen!")
except Exception as e:
    print(f"❌ Fehler beim Laden: {e}")
    print("\nVerfügbare Models:")
    if os.path.exists("models"):
        for f in os.listdir("models"):
            if f.endswith(".zip"):
                print(f"  - {f}")
    exit()

print("\n" + "=" * 60)
print("🎮 KI SPIELT POKEMON")
print("=" * 60)
print("\nDrücke CTRL+C zum Beenden\n")

obs, info = env.reset()

try:
    step_count = 0
    episode_reward = 0

    for i in range(100000):  # Lange genug zum Zuschauen
        action, _states = model.predict(obs, deterministic=True)
        action = int(action.item())

        obs, reward, done, truncated, info = env.step(action)

        step_count += 1
        episode_reward += reward

        # Alle 100 Steps ausgeben
        if step_count % 100 == 0:
            button_names = ["NoOp", "A", "B", "Start", "Select", "Up", "Down", "Left", "Right"]
            print(f"Step {step_count:5d} | Action: {button_names[action]:6s} | "
                  f"Reward: {reward:6.2f} | Total: {episode_reward:8.2f}")

        time.sleep(0.01)  # Langsamer machen zum Zuschauen

        if done or truncated:
            print(f"\n{'='*60}")
            print(f"Episode beendet!")
            print(f"Total Steps: {step_count}")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"{'='*60}\n")

            obs, info = env.reset()
            step_count = 0
            episode_reward = 0

except KeyboardInterrupt:
    print("\n\n" + "=" * 60)
    print("⏸️  Watch Mode beendet!")
    print("=" * 60)
    print(f"\nGespielte Steps: {step_count}")
    print(f"Episode Reward: {episode_reward:.2f}")

finally:
    env.close()
    print("\n✓ Fertig!")
