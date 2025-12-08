import chess
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from chess_env import ChessEnv
from mcts import MCTS
from utils import temperature_sample, get_policy_from_visits, save_pgn


class Arena:
    def __init__(self, device="cpu") -> None:
        self.device = device

    def play_game(
            self,
            model_white,
            model_black,
            mcts_white: MCTS,
            mcts_black: MCTS,
            config,
            save_path=None,
            game_id=0,
    ):
        env = ChessEnv(device=self.device)
        state = env.reset()

        examples_white = []
        examples_black = []
        move_count = 0

        while not env.board.is_game_over() and move_count < config.MAX_GAME_LENGTH:
            if env.board.turn == chess.WHITE:
                mcts = mcts_white
                current_examples = examples_white
            else:
                mcts = mcts_black
                current_examples = examples_black

            visit_counts = mcts.search(env, add_noise=False)

            legal_moves = env.legal_moves()
            policy = get_policy_from_visits(visit_counts, legal_moves)

            current_examples.append((state.cpu().clone(), policy, None))

            move = temperature_sample(visit_counts, temperature=0.1)

            state, reward, done = env.step(move)
            move_count += 1

        if env.board.is_game_over():
            result_str = env.board.result()
            if result_str == "1-0":
                result_white = 1
            elif result_str == "0-1":
                result_white = -1
            else:
                result_white = 0
        else:
            result_white = 0

        result_black = -result_white

        examples_white = [(s, p, result_white) for (s, p, _) in examples_white]

        examples_black = [(s, p, result_black) for (s, p, _) in examples_black]

        if save_path:
            save_pgn(env, path=save_path, prefix=f"arena_game_{game_id}")

        return result_white, move_count, examples_white, examples_black

    def compare_models(
            self,
            new_model,
            best_model,
            config,
            num_games=20,
            save_path=None,
    ):
        new_model.eval()
        best_model.eval()

        new_wins = 0
        best_wins = 0
        draws = 0
        all_examples = []

        print(f"\n🏆 Arena: new_model vs best_model ({num_games} gier)")

        for game_num in tqdm(range(num_games), desc="Arena games"):
            if game_num % 2 == 0:
                mcts_white = MCTS(new_model, config, self.device)
                mcts_black = MCTS(best_model, config, self.device)

                result_white, length, ex_white, ex_black = self.play_game(
                    model_white=new_model,
                    model_black=best_model,
                    mcts_white=mcts_white,
                    mcts_black=mcts_black,
                    config=config,
                    save_path=save_path,
                    game_id=game_num,
                )

                if result_white == 1:
                    new_wins += 1
                elif result_white == -1:
                    best_wins += 1
                else:
                    draws += 1

                all_examples.extend(ex_white)
            else:
                mcts_white = MCTS(best_model, config, self.device)
                mcts_black = MCTS(new_model, config, self.device)

                result_white, length, ex_white, ex_black = self.play_game(
                    model_white=best_model,
                    model_black=new_model,
                    mcts_white=mcts_white,
                    mcts_black=mcts_black,
                    config=config,
                    save_path=save_path,
                    game_id=game_num,
                )

                if result_white == -1:
                    new_wins += 1
                elif result_white == 1:
                    best_wins += 1
                else:
                    draws += 1

                all_examples.extend(ex_black)

        win_rate = (new_wins + 0.5 * draws) / num_games

        results = {
            "new_wins": new_wins,
            "best_wins": best_wins,
            "draws": draws,
            "win_rate": win_rate,
            "examples": all_examples,
            "total_games": num_games,
        }

        print("\n📊 Arena – wyniki:")
        print(f"   New model:  {new_wins} wygranych")
        print(f"   Best model: {best_wins} wygranych")
        print(f"   Remisy:     {draws}")
        print(f"   Win rate nowego modelu: {win_rate:.1%}")

        return results

    def should_replace_best_model(self, arena_results, threshold=0.55):
        win_rate = arena_results["win_rate"]

        if win_rate >= threshold:
            print(f"✅ New model lepszy: {win_rate:.1%} >= {threshold:.1%}")
            return True
        else:
            print(f"❌ Pozostaje stary best model: {win_rate:.1%} < {threshold:.1%}")
            return False
