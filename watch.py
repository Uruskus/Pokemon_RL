from pokemon_env import PokemonRedEnv
from stable_baselines3 import PPO
import time

print("Lade Environment...")
env = PokemonRedEnv(rom_path="pokemon_red.gb")

print("Lade trainiertes Model...")
try:
    model = PPO.load("pokemon_model")
    print("Model erfolgreich geladen!")
except Exception as e:
    print(f"Fehler beim Laden: {e}")
    exit()

print("Starte Spiel...")
obs, info = env.reset()

try:
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        action = int(action.item())

        obs, reward, done, truncated, info = env.step(action)

        if i % 100 == 0:  # Alle 100 Steps ausgeben
            print(f"Step {i}, Action: {action}, Reward: {reward:.2f}")

        time.sleep(0.01)  # Langsamer machen zum Zuschauen

        if done:
            print("Episode beendet! Neustart...")
            obs, info = env.reset()

except KeyboardInterrupt:
    print("\nStoppe...")
except Exception as e:
    print(f"Fehler: {e}")
    import traceback
    traceback.print_exc()
finally:
    env.close()
    print("Fertig!")
