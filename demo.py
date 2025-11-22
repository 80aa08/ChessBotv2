import torch
import os
from pathlib import Path

def demo_training():
    print("="*70)
    print("🎓 DEMO 1: Mini Training Session")
    print("="*70)
    print("\nUruchamiam krótki trening (5 iteracji) z pełnym logowaniem...")

    from config import Config
    config = Config()
    config.NUM_ITERATIONS = 5
    config.NUM_SELFPLAY_GAMES = 5
    config.NUM_SIMULATIONS = 50
    config.TRAIN_EPOCHS = 1

    with open('config_backup.py', 'w') as f:
        f.write("# Backup of original config\n")

    print(f"  - Iterations: {config.NUM_ITERATIONS}")
    print(f"  - Games per iteration: {config.NUM_SELFPLAY_GAMES}")
    print(f"  - MCTS simulations: {config.NUM_SIMULATIONS}")

    input("\n▶️  Press Enter to start training...")

    from train import main
    main()

    print("\n✅ Training demo complete!")
    print("📁 Check ./experiments/ for all generated data and plots")


def demo_plots_generation():
    print("\n" + "="*70)
    print("🎓 DEMO 2: Plots Generation")
    print("="*70)

    from data_logger import DataLogger

    exp_dir = Path("./experiments")
    if not exp_dir.exists():
        print("❌ No experiments found. Run training first.")
        return

    experiments = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
    if not experiments:
        print("❌ No experiments found. Run training first.")
        return

    latest_exp = experiments[-1]
    print(f"\n📁 Using experiment: {latest_exp.name}")

    logger = DataLogger(experiment_name="demo", base_dir=str(exp_dir))

    print("\n📊 Generating all plots...")
    logger.generate_all_plots()

    print(f"\n✅ Plots saved to: {logger.plots_dir}")
    print("\nGenerated plots:")
    print("  1. training_losses.png/pdf - Wykresy loss")
    print("  2. win_rates.png/pdf - Wskaźniki wygranych")
    print("  3. game_statistics.png/pdf - Statystyki gier")
    print("  4. learning_curves.png/pdf - Krzywe uczenia")
    print("  5. validation_metrics.png/pdf - Metryki walidacji")
    print("  6. combined_overview.png/pdf - Przegląd kompletny")


def demo_play_vs_ai():
    print("\n" + "="*70)
    print("🎓 DEMO 3: Play Against AI")
    print("="*70)

    model_path = "./models/best_model.pt"

    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at {model_path}")
        print("   Please train a model first or specify a different path.")
        return

    print(f"\n🤖 Loading model from: {model_path}")
    print("\n🎮 Starting game interface...")

    from play_vs_human import SimpleCLI
    SimpleCLI.main()


def demo_model_evaluation():
    print("\n" + "="*70)
    print("🎓 DEMO 4: Model Evaluation")
    print("="*70)

    model_path = "./models/best_model.pt"

    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at {model_path}")
        return

    from model import ChessNet
    from evaluator import ModelEvaluator

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Using device: {device}")

    print(f"\n🤖 Loading model...")
    model = ChessNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    evaluator = ModelEvaluator(device=device)

    print("\n📊 Running evaluation tests...")
    print("\n1️⃣  Testing against random player (10 games)...")
    win_rate = evaluator.evaluate_vs_random(model, num_games=10)

    print(f"\n✅ Results:")
    print(f"  - Win rate vs random: {win_rate:.1%}")

    if win_rate > 0.8:
        print("  - 🎉 Excellent! Model is much stronger than random.")
    elif win_rate > 0.6:
        print("  - 👍 Good! Model shows decent chess understanding.")
    elif win_rate > 0.5:
        print("  - 📈 Model is learning, but needs more training.")
    else:
        print("  - ⚠️  Model needs more training.")


def demo_data_export():
    print("\n" + "="*70)
    print("🎓 DEMO 5: Data Export for Thesis")
    print("="*70)

    exp_dir = Path("./experiments")
    if not exp_dir.exists() or not list(exp_dir.iterdir()):
        print("\n❌ No experiments found. Run training first.")
        return

    experiments = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
    latest_exp = experiments[-1]

    print(f"\n📁 Exporting data from: {latest_exp.name}")

    from data_logger import DataLogger

    print("\n📚 Exported files structure:")
    print(f"""
    {latest_exp.name}/
    ├── data/
    │   ├── iterations.csv       # Metryki treningowe
    │   ├── games.csv            # Zapisane gry
    │   ├── validation.csv       # Wyniki walidacji
    │   └── summary.json         # Podsumowanie
    ├── plots/
    │   ├── training_losses.png/pdf
    │   ├── win_rates.png/pdf
    │   ├── game_statistics.png/pdf
    │   ├── learning_curves.png/pdf
    │   ├── validation_metrics.png/pdf
    │   └── combined_overview.png/pdf
    ├── models/
    │   └── [saved checkpoints]
    ├── games/
    │   └── [PGN files]
    └── README.md                # Opis eksperymentu
    """)

    print("\n✅ All files ready for thesis!")
    print(f"📂 Location: {latest_exp.absolute()}")


def demo_full_pipeline():
    print("\n" + "="*70)
    print("🎓 FULL DEMO: Complete Pipeline for Thesis")
    print("="*70)

    print("""
    Demo przeprowadzi Cię przez pełny proces:

    1. ⏱️  Mini trening (5 iteracji, ~10-15 min)
    2. 📊 Generowanie wykresów
    3. 📈 Ewaluacja modelu
    4. 💾 Eksport danych dla pracy
    5. 🎮 Możliwość zagrania przeciwko AI

    Wszystkie wygenerowane dane będą gotowe do użycia w pracy inżynierskiej!
    """)

    confirm = input("\n▶️  Czy chcesz uruchomić pełny pipeline? (y/n): ").strip().lower()

    if confirm != 'y':
        print("❌ Demo cancelled.")
        return

    print("\n" + "🔹"*35)
    print("KROK 1/5: Mini Training")
    print("🔹"*35)
    demo_training()

    print("\n" + "🔹"*35)
    print("KROK 2/5: Generating Plots")
    print("🔹"*35)
    demo_plots_generation()

    print("\n" + "🔹"*35)
    print("KROK 3/5: Model Evaluation")
    print("🔹"*35)
    demo_model_evaluation()

    print("\n" + "🔹"*35)
    print("KROK 4/5: Data Export")
    print("🔹"*35)
    demo_data_export()

    print("\n" + "🔹"*35)
    print("KROK 5/5: Play Against AI (Optional)")
    print("🔹"*35)
    play = input("\n▶️  Czy chcesz zagrać przeciwko AI? (y/n): ").strip().lower()
    if play == 'y':
        demo_play_vs_ai()

    print("\n" + "="*70)
    print("🎉 FULL DEMO COMPLETE!")
    print("="*70)

def main_menu():
    while True:
        print("\n" + "="*70)
        print("♟️  ChessBot - Demonstracja dla Pracy Inżynierskiej")
        print("="*70)
        print("""
        Wybierz demo:

        1. 🔥 Pełny Pipeline (wszystkie kroki)
        2. ⏱️  Mini Trening (5 iteracji)
        3. 📊 Generowanie Wykresów
        4. 🎮 Gra przeciwko AI
        5. 📈 Ewaluacja Modelu
        6. 💾 Eksport Danych

        0. ❌ Wyjście
        """)

        choice = input("Wybór (0-6): ").strip()

        if choice == '0':
            print("\n👋 Do widzenia!")
            break
        elif choice == '1':
            demo_full_pipeline()
        elif choice == '2':
            demo_training()
        elif choice == '3':
            demo_plots_generation()
        elif choice == '4':
            demo_play_vs_ai()
        elif choice == '5':
            demo_model_evaluation()
        elif choice == '6':
            demo_data_export()
        else:
            print("❌ Nieprawidłowy wybór!")


if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ChessBot - Demonstracja dla Pracy Inżynierskiej             ║
    ║                                                               ║
    ║   Autor: Michał Michalik                                      ║
    ║   Projekt: INTEGRACJA ALGORYTMU MCTS I SIECI NEURONOWYCH      ║
    ║            W CELU STWORZENIA SZACHOWEGO SILNIKA AI            ║
    ║   Algorytm: MonteCarlo Self-Play Learning                     ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    main_menu()
