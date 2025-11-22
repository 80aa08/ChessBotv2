import torch
import os
from pathlib import Path

def demo_training():
    print("="*70)
    print("🎓 DEMO 1: Minimalna sesja treningu")
    print("="*70)
    print("\nUruchamiam krótki trening (2 iteracje) z pełnym logowaniem...")

    from config_demo import Config
    config = Config()

    print(f"  - Iteracje: {config.NUM_ITERATIONS}")
    print(f"  - Gry na iteracje: {config.NUM_SELFPLAY_GAMES}")
    print(f"  - MCTS symulacje: {config.NUM_SIMULATIONS}")

    input("\n▶️  Wciśniej enter aby rozpocząć...")

    from train import main
    main(config)

    print("\n✅ Trening zakończony!")
    print("📁 Sprawdź ./experiments/ dla wygenerowych danych i plików")


def demo_plots_generation():
    print("\n" + "="*70)
    print("🎓 DEMO 2: Generowanie wykresów")
    print("="*70)

    exp_dir = Path("./experiments")
    if not exp_dir.exists():
        print("❌ Nie znaleziono experiments. Najpierw uruchom trening.")
        return

    experiments = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
    if not experiments:
        print("❌ Nie znaleziono experiments. Najpierw uruchom trening.")
        return

    demo_exps = [d for d in experiments if d.name.startswith("demo_")]
    if demo_exps:
        latest_exp = demo_exps[-1]
    else:
        latest_exp = experiments[-1]

    print(f"\n📁 Używam eksperymentu: {latest_exp.name}")

    plots_dir = latest_exp / "plots"
    if not plots_dir.exists():
        print("⚠️ W tym eksperymencie nie ma jeszcze wygenerowanych wykresów.")
        print("   (upewnij się, że trening doszedł do walidacji i wywołał export_for_thesis).")
        return

    print(f"\n✅ Wykresy są zapisane w: {plots_dir}")
    print("\nWygenerowane wykresy:")
    print("  - training_losses.png/pdf")
    print("  - win_rates.png/pdf")
    print("  - game_statistics.png/pdf")
    print("  - learning_curves.png/pdf")
    print("  - validation_metrics.png/pdf")
    print("  - combined_overview.png/pdf")

def demo_play_vs_ai():
    print("\n" + "="*70)
    print("🎓 DEMO 3: Gra przeciwko AI")
    print("="*70)

    model_path = "./models/best_model.pt"

    if not os.path.exists(model_path):
        print(f"\n❌ Nie znaleziono modelu {model_path}")
        print("   Wykonaj trening lub wybierz inną ścieżkę do modelu.")
        return

    print(f"\n🤖 Ładowanie modelu: {model_path}")
    print("\n🎮 Uruchamianie interfejsu gry...")

    from play_vs_human import SimpleCLI
    SimpleCLI.main()


def demo_model_evaluation():
    print("\n" + "="*70)
    print("🎓 DEMO 4: Ocena modelu")
    print("="*70)

    model_path = "./models/best_model.pt"

    if not os.path.exists(model_path):
        print(f"\n❌ Nie znaleziono modelu {model_path}")
        return

    from model import ChessNet
    from evaluator import ModelEvaluator

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")

    print(f"\n🤖 Ładowanie modelu...")
    model = ChessNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    evaluator = ModelEvaluator(device=device)

    print("\n📊 Uruchamianie testów oceny...")
    print("\n1️⃣  Testowanie przeciwko losowemu przeciwnikowi (10 gier)...")
    win_rate = evaluator.evaluate_vs_random(model, num_games=10)

    print(f"\n✅ Wyniki:")
    print(f"  - Wskaźnik zwycięstw vs losowy: {win_rate:.1%}")


def demo_data_export():
    print("\n" + "="*70)
    print("🎓 DEMO 5: Export danych")
    print("="*70)

    exp_dir = Path("./experiments")
    if not exp_dir.exists() or not list(exp_dir.iterdir()):
        print("❌ Nie znaleziono experiments. Najpierw uruchom trening.")
        return

    experiments = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
    latest_exp = experiments[-1]

    print(f"\n📁 Exportowanie danych: {latest_exp.name}")

    from data_logger import DataLogger

    print("\n📚 Export struktury:")
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

    print(f"📂 Lokalizacja: {latest_exp.absolute()}")


def demo_full_pipeline():
    print("\n" + "="*70)
    print("🎓 Kompletna wersja demonstracyjna")
    print("="*70)

    print("""
    Demo przeprowadzi Cię przez pełny proces:

    1. ⏱️  Mini trening (2 iteracje, ~10-15 min)
    2. 📊 Generowanie wykresów
    3. 📈 Ewaluacja modelu
    4. 💾 Eksport danych
    5. 🎮 Możliwość zagrania przeciwko AI

    """)

    confirm = input("\n▶️  Czy chcesz uruchomić pełny pipeline? (y/n): ").strip().lower()

    if confirm != 'y':
        print("❌ Anulowanie dema.")
        return

    print("\n" + "🔹"*35)
    print("KROK 1/5: Mini Training")
    print("🔹"*35)
    demo_training()

    print("\n" + "🔹"*35)
    print("KROK 2/5: Generowanie wykresów")
    print("🔹"*35)
    demo_plots_generation()

    print("\n" + "🔹"*35)
    print("KROK 3/5: Ocena modelu")
    print("🔹"*35)
    demo_model_evaluation()

    print("\n" + "🔹"*35)
    print("KROK 4/5: Export danych")
    print("🔹"*35)
    demo_data_export()

    print("\n" + "🔹"*35)
    print("KROK 5/5: Gra przeciwko AI")
    print("🔹"*35)
    play = input("\n▶️  Czy chcesz zagrać przeciwko AI? (y/n): ").strip().lower()
    if play == 'y':
        demo_play_vs_ai()

    print("\n" + "="*70)
    print("🎉 Koniec wersji demonstracyjnej! (Reszta w DLC)")
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
            print("\n👋 Koniec!")
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
