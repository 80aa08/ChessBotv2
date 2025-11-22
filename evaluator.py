import torch
import numpy as np
import chess
from tqdm import tqdm
from chess_env import ChessEnv
from mcts import MCTS
from config import Config
from utils import temperature_sample, save_pgn


class ModelEvaluator:
    def __init__(self, device='cpu'):
        self.device = device
        self.config = Config()

    def play_game(self, model1, model2, verbose=False):
        env = ChessEnv(device=self.device)
        mcts1 = MCTS(model1, self.config, self.device)
        mcts2 = MCTS(model2, self.config, self.device)

        state = env.reset()
        move_count = 0

        while not env.board.is_game_over() and move_count < self.config.MAX_GAME_LENGTH:
            mcts = mcts1 if env.board.turn == chess.WHITE else mcts2

            visit_counts = mcts.search(env, add_noise=False)
            move = temperature_sample(visit_counts, temperature=0.1)

            state, reward, done = env.step(move)
            move_count += 1

            if verbose and move_count % 10 == 0:
                print(f"Ruch {move_count}")

        result_str = env.board.result()
        if result_str == '1-0':
            result = 1
        elif result_str == '0-1':
            result = -1
        else:
            result = 0

        return result, move_count, env

    def evaluate_vs_model(self, current_model, opponent_model, num_games=20,
                          save_games=False, games_dir='./eval_games'):
        current_model.eval()
        opponent_model.eval()

        wins = 0
        losses = 0
        draws = 0
        game_lengths = []

        for i in tqdm(range(num_games), desc="Evaluating"):
            if i < num_games // 2:
                result, length, env = self.play_game(current_model, opponent_model)
            else:
                result, length, env = self.play_game(opponent_model, current_model)
                result = -result

            game_lengths.append(length)

            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1

            if save_games and i % 5 == 0:
                save_pgn(env, path=games_dir, prefix=f"eval_game_{i}")

        total = wins + losses + draws

        results = {
            'win_rate': wins / total,
            'loss_rate': losses / total,
            'draw_rate': draws / total,
            'avg_game_length': np.mean(game_lengths),
            'total_games': total,
            'wins': wins,
            'losses': losses,
            'draws': draws
        }

        return results

    def evaluate_vs_random(self, model, num_games=10):
        model.eval()
        wins = 0

        for i in range(num_games):
            env = ChessEnv(device=self.device)
            mcts = MCTS(model, self.config, self.device)

            state = env.reset()
            move_count = 0

            while not env.board.is_game_over() and move_count < self.config.MAX_GAME_LENGTH:
                if env.board.turn == chess.WHITE:
                    visit_counts = mcts.search(env, add_noise=False)
                    move = temperature_sample(visit_counts, temperature=0.1)
                else:
                    legal_moves = list(env.board.legal_moves)
                    move = np.random.choice(legal_moves)

                state, reward, done = env.step(move)
                move_count += 1

            if env.board.result() == '1-0':
                wins += 1

        return wins / num_games

    def measure_policy_quality(self, model, test_positions):
        model.eval()
        correct = 0

        for fen, best_move_uci in test_positions:
            board = chess.Board(fen)
            env = ChessEnv(device=self.device)
            env.board = board

            mcts = MCTS(model, self.config, self.device)
            visit_counts = mcts.search(env, add_noise=False)

            best_move = max(visit_counts.items(), key=lambda x: x[1])[0]

            if best_move.uci() == best_move_uci:
                correct += 1

        return correct / len(test_positions)

    def estimate_elo(self, model, reference_models_elos, num_games_each=20):
        results = []

        for ref_model, ref_elo in reference_models_elos.items():
            eval_results = self.evaluate_vs_model(
                model, ref_model,
                num_games=num_games_each,
                save_games=False
            )

            score = eval_results['win_rate'] + 0.5 * eval_results['draw_rate']
            results.append((ref_elo, score))

        estimated_elos = []
        for ref_elo, score in results:
            if score > 0.99:
                score = 0.99
            if score < 0.01:
                score = 0.01

            elo_diff = -400 * np.log10(1 / score - 1)
            estimated_elo = ref_elo + elo_diff
            estimated_elos.append(estimated_elo)

        return np.mean(estimated_elos)

    def tournament(self, models, names, num_games_per_pair=10):
        n = len(models)
        scores = np.zeros(n)  # Win = 1, Draw = 0.5, Loss = 0

        print(f"Uruchamianie zawodów {n} modeli...")

        for i in range(n):
            for j in range(i + 1, n):
                print(f"\n{names[i]} vs {names[j]}")

                for game_idx in range(num_games_per_pair):
                    if game_idx % 2 == 0:
                        result, _, _ = self.play_game(models[i], models[j])
                        if result == 1:
                            scores[i] += 1
                        elif result == -1:
                            scores[j] += 1
                        else:
                            scores[i] += 0.5
                            scores[j] += 0.5
                    else:
                        result, _, _ = self.play_game(models[j], models[i])
                        if result == 1:
                            scores[j] += 1
                        elif result == -1:
                            scores[i] += 1
                        else:
                            scores[i] += 0.5
                            scores[j] += 0.5

        results = []
        for i in range(n):
            results.append({
                'name': names[i],
                'score': scores[i],
                'games_played': num_games_per_pair * (n - 1)
            })

        results.sort(key=lambda x: x['score'], reverse=True)

        print("\nWyniki:")
        print("-" * 50)
        for i, r in enumerate(results):
            print(f"{i + 1}. {r['name']}: {r['score']:.1f}/{r['games_played']}")

        return results


def test_model_strength(model, device='cpu'):
    evaluator = ModelEvaluator(device=device)

    print("Testowanie siły modelu...")

    print("1. Zachowanie przeciwko losowym ruchom...")
    random_win_rate = evaluator.evaluate_vs_random(model, num_games=10)
    print(f"   Wskaźnik zwycięstw vs losowy: {random_win_rate:.1%}")

    results = {
        'random_win_rate': random_win_rate,
        # 'policy_accuracy': policy_accuracy,
    }

    return results


if __name__ == '__main__':
    from model import ChessNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ChessNet().to(device)
    checkpoint = torch.load('./models/best_model.pt')
    model.load_state_dict(checkpoint)

    evaluator = ModelEvaluator(device=device)

    random_wr = evaluator.evaluate_vs_random(model, num_games=20)
    print(f"Wskażnik zwycięstw: {random_wr:.1%}")
