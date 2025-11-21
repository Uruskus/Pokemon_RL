from pokemon_rl_env import PokemonRedEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import time
import os

# Erstelle Ordner für Models
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Environment erstellen
print("=" * 50)
print("POKEMON RL TRAINING")
print("=" * 50)
print("\nErstelle Pokemon Environment...")
env = PokemonRedEnv(rom_path="pokemon_red.gb", render_mode=None)  # Kein Fenster für schnelleres Training
print("✓ Environment erstellt (Training läuft ohne GUI für maximale Performance)")

# Prüfe ob bereits ein Model existiert
model_path = "models/pokemon_model_latest.zip"
if os.path.exists(model_path):
    print(f"\n✓ Gefundenes Model wird geladen: {model_path}")
    print("Training wird fortgesetzt!")
    model = PPO.load(model_path, env=env)
else:
    print("\nKein bestehendes Model gefunden. Erstelle neues Model...")
    # PPO Model erstellen mit optimierten Parametern
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=0.0003,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./logs/"
    )

# Callbacks für automatisches Speichern
checkpoint_callback = CheckpointCallback(
    save_freq=50000,  # Speichere alle 50k Steps
    save_path="./models/",
    name_prefix="pokemon_checkpoint",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

print("\n" + "=" * 50)
print("TRAINING KONFIGURATION")
print("=" * 50)
print(f"Gesamt Steps: 10,000,000")
print(f"Checkpoint alle: 50,000 Steps")
print(f"Training in: 10 Iterationen à 1 Million Steps")
print(f"Modelle werden gespeichert in: ./models/")
print("=" * 50)
print("\n⚠️  WICHTIG: Drücke CTRL+C um Training zu pausieren und zu speichern")
print("=" * 50)

# Training starten mit Iterationen
try:
    total_steps = 10_000_000
    steps_per_iteration = 1_000_000
    iterations = total_steps // steps_per_iteration
    
    print(f"\n🚀 Starte Training mit {total_steps:,} Steps...\n")
    
    for i in range(iterations):
        print(f"\n{'='*50}")
        print(f"ITERATION {i+1}/{iterations}")
        print(f"Progress: {(i/iterations)*100:.1f}% | Steps: {i*steps_per_iteration:,}/{total_steps:,}")
        print(f"{'='*50}\n")
        
        # Training für diese Iteration
        model.learn(
            total_timesteps=steps_per_iteration,
            callback=checkpoint_callback,
            reset_num_timesteps=False  # Wichtig: Zähler nicht zurücksetzen!
        )
        
        # Speichere nach jeder Iteration
        iteration_model_path = f"models/pokemon_model_{(i+1)}m_steps.zip"
        model.save(iteration_model_path)
        model.save("models/pokemon_model_latest.zip")  # Überschreibe "latest"
        
        print(f"\n✓ Checkpoint gespeichert: {iteration_model_path}")
        print(f"✓ Latest Model aktualisiert")
        
        # Zeige Fortschritt
        completion = ((i+1) / iterations) * 100
        print(f"\n📊 Training {completion:.1f}% abgeschlossen!")
        print(f"   {(i+1)*steps_per_iteration:,} / {total_steps:,} Steps")
    
    # Training abgeschlossen
    print("\n" + "=" * 50)
    print("🎉 TRAINING ABGESCHLOSSEN!")
    print("=" * 50)
    
    # Finales Model speichern
    final_model_path = "models/pokemon_model_final_10m.zip"
    model.save(final_model_path)
    print(f"\n✓ Finales Model gespeichert: {final_model_path}")
    print(f"✓ Latest Model: models/pokemon_model_latest.zip")
    print("\n💡 Nutze 'python watch.py' um die KI spielen zu sehen!")

except KeyboardInterrupt:
    print("\n\n" + "=" * 50)
    print("⏸️  TRAINING UNTERBROCHEN!")
    print("=" * 50)
    
    # Speichere aktuellen Stand
    backup_path = "models/pokemon_model_backup.zip"
    model.save(backup_path)
    model.save("models/pokemon_model_latest.zip")
    
    print(f"\n✓ Backup gespeichert: {backup_path}")
    print(f"✓ Latest Model aktualisiert")
    print("\n💡 Starte 'python train.py' erneut um fortzusetzen!")
    print("   (Das Model wird automatisch geladen)")

except Exception as e:
    print(f"\n❌ FEHLER beim Training: {e}")
    import traceback
    traceback.print_exc()
    
    # Speichere trotzdem
    try:
        error_path = "models/pokemon_model_error_backup.zip"
        model.save(error_path)
        print(f"\n✓ Notfall-Backup gespeichert: {error_path}")
    except:
        print("❌ Konnte kein Backup erstellen!")

finally:
    print("\nSchließe Environment...")
    env.close()
    print("✓ Fertig!")
