import torch
import chess
import chess.svg
from chess_env import ChessEnv
from model import ChessNet
from mcts import MCTS
from config import Config
from utils import temperature_sample
import os

class ChessGame:

    def __init__(self, model_path, device='cpu', human_color='white'):
        self.device = torch.device(device)
        self.human_color = chess.WHITE if human_color.lower() == 'white' else chess.BLACK

        print(f"🤖 Loading model from {model_path}...")
        self.model = ChessNet().to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✅ Model loaded successfully!")
        else:
            print("⚠️  Model file not found, using random initialized model")

        self.model.eval()

        self.config = Config()
        self.config.NUM_SIMULATIONS = 400
        self.mcts = MCTS(self.model, self.config, self.device)

        self.env = ChessEnv(device=self.device)
        self.env.reset()

        print(f"👤 You are playing as: {human_color.upper()}")
        print(f"🤖 AI is playing as: {'BLACK' if human_color.lower() == 'white' else 'WHITE'}")
    
    def display_board(self):
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        board_str = str(self.env.board)
        rows = board_str.split("\n")

        print("\n" + "=" * 50)
        print()

        print("   " + " ".join(letters))

        for rank, row in zip(range(8, 0, -1), rows):
            print(f"{rank}  {row}")

        print("   " + " ".join(letters))

        print("=" * 50)
        print(f"Move: {self.env.board.fullmove_number}")
        print(f"Turn: {'WHITE' if self.env.board.turn == chess.WHITE else 'BLACK'}")
        if self.env.board.is_check():
            print("⚠️  CHECK!")
        print()
        

    def get_human_move(self):
        legal_moves = list(self.env.board.legal_moves)

        print("Legal moves:")
        moves_str = [move.uci() for move in legal_moves]
        for i in range(0, len(moves_str), 8):
            print("  " + "  ".join(moves_str[i:i+8]))

        while True:
            move_input = input("\n👤 Your move (e.g., 'e2e4' or 'q' to quit): ").strip().lower()

            if move_input == 'q':
                return None

            if move_input == 'hint':
                print("💡 Getting hint from AI...")
                hint_move = self.get_ai_move(show_thinking=True)
                print(f"   Suggested move: {hint_move.uci()}")
                continue

            try:
                move = chess.Move.from_uci(move_input)
                if move in legal_moves:
                    return move
                else:
                    print("❌ Illegal move! Try again.")
            except:
                print("❌ Invalid format! Use format like 'e2e4' or 'e7e8q' for promotion")

    def get_ai_move(self, show_thinking=False):
        print("🤖 AI is thinking...")

        if show_thinking:
            print(f"   Running {self.config.NUM_SIMULATIONS} MCTS simulations...")

        visit_counts = self.mcts.search(self.env, add_noise=False)

        best_move = temperature_sample(visit_counts, temperature=0.1)

        if show_thinking:
            sorted_moves = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            print("\n   Top 3 moves considered:")
            for i, (move, visits) in enumerate(sorted_moves, 1):
                percentage = (visits / sum(visit_counts.values())) * 100
                print(f"   {i}. {move.uci()} ({visits} visits, {percentage:.1f}%)")

        return best_move

    def play(self):
        print("\n🎮 Game started! Good luck!")
        print("Commands: move in UCI format (e.g., 'e2e4'), 'hint' for suggestion, 'q' to quit")

        move_count = 0

        while not self.env.board.is_game_over() and move_count < 200:
            self.display_board()

            if self.env.board.turn == self.human_color:
                move = self.get_human_move()
                if move is None:
                    print("👋 Game aborted by player")
                    return
            else:
                move = self.get_ai_move(show_thinking=True)
                print(f"🤖 AI plays: {move.uci()}")
                input("Press Enter to continue...")

            try:
                self.env.step(move)
                move_count += 1
            except Exception as e:
                print(f"❌ Error executing move: {e}")
                continue

        self.display_board()
        self.show_result()

    def show_result(self):
        print("\n" + "="*50)
        print("🏁 GAME OVER!")
        print("="*50)

        result = self.env.board.result()

        if result == '1-0':
            winner = "WHITE"
            if self.human_color == chess.WHITE:
                print("🎉 Congratulations! You won!")
            else:
                print("😔 AI won this time. Better luck next time!")
        elif result == '0-1':
            winner = "BLACK"
            if self.human_color == chess.BLACK:
                print("🎉 Congratulations! You won!")
            else:
                print("😔 AI won this time. Better luck next time!")
        else:
            winner = "DRAW"
            print("🤝 It's a draw! Well played!")

        print(f"\nResult: {result} ({winner})")
        print(f"Total moves: {self.env.board.fullmove_number}")

        self.save_game()

    def save_game(self):
        from utils import save_pgn
        save_pgn(self.env, path="./human_games", prefix="human_vs_ai")
        print(f"💾 Game saved to ./human_games/")


class SimpleCLI:

    @staticmethod
    def main():
        print("="*60)
        print("♟️  ChessBot - Play Against AI")
        print("="*60)

        print("\n📁 Available models:")
        model_dir = "./models"
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith('.pt') or f.endswith('.pth') or f.endswith('.tar')]
            if models:
                for i, model in enumerate(models, 1):
                    print(f"  {i}. {model}")

                choice = input(f"\nSelect model (1-{len(models)}) or press Enter for best_model.pt: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(models):
                    model_path = os.path.join(model_dir, models[int(choice)-1])
                else:
                    model_path = os.path.join(model_dir, "best_model.pt")
            else:
                print("  No models found in ./models/")
                model_path = os.path.join(model_dir, "best_model.pt")
        else:
            model_path = "./models/best_model.pt"

        print("\n🎨 Choose your color:")
        print("  1. White (you play first)")
        print("  2. Black (AI plays first)")

        color_choice = input("Select (1-2) [default: 1]: ").strip()
        human_color = 'white' if color_choice != '2' else 'black'

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n💻 Using device: {device.upper()}")

        print("\n🚀 Starting game...")
        game = ChessGame(model_path, device=device, human_color=human_color)
        game.play()

        print("\n" + "="*60)
        play_again = input("Play again? (y/n): ").strip().lower()
        if play_again == 'y':
            SimpleCLI.main()
        else:
            print("👋 Thanks for playing! Goodbye!")


def quick_play(model_path="./models/best_model.pt", human_color="white", device="cpu"):
    game = ChessGame(model_path, device=device, human_color=human_color)
    game.play()


if __name__ == '__main__':
    SimpleCLI.main()
