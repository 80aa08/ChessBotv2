import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from tqdm import tqdm
import chess
import time

from config import Config
from model import ChessNet
from chess_env import ChessEnv
from mcts import MCTS
from utils import (
    save_pgn,
    ReplayBuffer,
    get_policy_from_visits,
    temperature_sample
)
from data_logger import DataLogger
from arena import Arena
import copy


class ChessDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, policy, value = self.examples[idx]

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(policy, torch.Tensor):
            policy = torch.tensor(policy, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value], dtype=torch.float32)

        return state, policy, value


def play_game(model, config, device, iteration, game_num=None, exp_games_dir=None):
    env = ChessEnv(device=device)
    mcts = MCTS(model, config, device)

    state = env.reset()
    examples = []
    move_count = 0

    while not env.board.is_game_over() and move_count < config.MAX_GAME_LENGTH:
        visit_counts = mcts.search(env, add_noise=True)

        legal_moves = env.legal_moves()
        policy = get_policy_from_visits(visit_counts, legal_moves)

        examples.append((state.cpu().clone(), policy, None))

        temperature = 1.0 if move_count < config.TEMP_THRESHOLD else 0.1
        move = temperature_sample(visit_counts, temperature)

        state, reward, done = env.step(move)
        move_count += 1

    final_result = reward
    for i in range(len(examples)):
        s, p, _ = examples[i]
        moves_from_end = len(examples) - i - 1
        value = final_result if moves_from_end % 2 == 0 else -final_result
        examples[i] = (s, p, value)

    save_pgn(env, path=config.GAMES_DIR, prefix="selfplay", iteration=iteration)

    if exp_games_dir is not None:
        if game_num is not None:
            prefix = f"iter{iteration}-game{game_num}"
        else:
            prefix = f"iter{iteration}"

        save_pgn(env, path=str(exp_games_dir), prefix=prefix, iteration=iteration)

    return examples


def train_model(model, optimizer, dataset, config, device):
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    num_batches = 0

    for states, policies, values in dataloader:
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)

        policy_logits, pred_values = model(states)

        log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.mean(torch.sum(policies * log_probs, dim=1))

        value_loss = torch.mean((values.squeeze() - pred_values.squeeze()) ** 2)

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

    return (
        total_loss / num_batches,
        total_policy_loss / num_batches,
        total_value_loss / num_batches
    )


def validate_model(model, config, device):
    model.eval()

    wins = 0
    total = config.NUM_VALIDATION_GAMES

    for i in range(total):
        env = ChessEnv(device=device)
        mcts = MCTS(model, config, device)

        state = env.reset()
        move_count = 0

        while not env.board.is_game_over() and move_count < config.MAX_GAME_LENGTH:
            visit_counts = mcts.search(env, add_noise=False)

            move = temperature_sample(visit_counts, temperature=0.1)

            state, reward, done = env.step(move)
            move_count += 1

        if env.board.result() == '1-0':
            wins += 1

        if i == 0:
            save_pgn(env, path="./validation", prefix="validation")

    return wins / total


def main(config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not config:
        config = Config()

    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.GAMES_DIR, exist_ok=True)

    print("📊 Initializing logging systems...")

    writer = SummaryWriter(config.LOG_DIR)

    from metrics import MetricsTracker
    metrics = MetricsTracker(log_dir=config.LOG_DIR)

    experiment_name = getattr(config, "EXPERIMENT_NAME", "chess_training")
    logger = DataLogger(experiment_name=experiment_name, base_dir="./experiments")

    print(f"✅ Logs will be saved to:")
    print(f"   - Tensorboard: {config.LOG_DIR}")
    print(f"   - Experiments: {logger.exp_dir}")

    init_model_path = getattr(config, "INIT_MODEL_PATH", "").strip()

    model = ChessNet().to(device)

    best_model = None
    arena = Arena(device=device)

    if os.path.isfile(init_model_path):
        print(f"🧠 Ładuję istniejący best model z: {init_model_path}")
        best_model = ChessNet().to(device)
        best_model.load_state_dict(torch.load(init_model_path, map_location=device))
    else:
        print("ℹ️ Brak zapisanego best_model.pt – zostanie ustawiony przy pierwszej walidacji.")

    if init_model_path:
        if os.path.isfile(init_model_path):
            try:
                model.load_state_dict(torch.load(init_model_path, map_location=device))
                print(f"Załadowano model {init_model_path}")
            except FileNotFoundError:
                pass

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    replay_buffer = ReplayBuffer(max_size=config.REPLAY_BUFFER_SIZE)

    best_val_win_rate = 0.0
    patience_counter = 0

    for iteration in range(1, config.NUM_ITERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration}/{config.NUM_ITERATIONS}")
        print(f"{'=' * 60}")

        iteration_start_time = time.time()

        print(f"Playing {config.NUM_SELFPLAY_GAMES} self-play games...")
        iteration_examples = []

        for game_num in tqdm(range(config.NUM_SELFPLAY_GAMES), desc="Self-play"):
            examples = play_game(model, config, device, iteration, game_num=game_num, exp_games_dir=logger.games_dir)
            iteration_examples.extend(examples)

            if examples:
                game_result = examples[-1][2]
                logger.log_game(
                    iteration=iteration,
                    game_id=game_num,
                    result=game_result,
                    length=len(examples),
                    opening="Unknown"
                )

        print(f"Generated {len(iteration_examples)} training examples")

        replay_buffer.add(iteration_examples)
        print(f"Replay buffer size: {len(replay_buffer)}")

        if len(replay_buffer) < config.MIN_BUFFER_SIZE:
            print(f"Waiting for more examples ({len(replay_buffer)}/{config.MIN_BUFFER_SIZE})")
            continue

        print("Training model...")
        for epoch in range(config.TRAIN_EPOCHS):
            train_examples = replay_buffer.get_all()
            dataset = ChessDataset(train_examples)

            loss, policy_loss, value_loss = train_model(
                model, optimizer, dataset, config, device
            )

            print(f"  Epoch {epoch + 1}/{config.TRAIN_EPOCHS}: "
                  f"Loss={loss:.4f}, Policy={policy_loss:.4f}, Value={value_loss:.4f}")

            global_step = iteration * config.TRAIN_EPOCHS + epoch
            writer.add_scalar('Loss/total', loss, global_step)
            writer.add_scalar('Loss/policy', policy_loss, global_step)
            writer.add_scalar('Loss/value', value_loss, global_step)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_rate', current_lr, iteration)

        iteration_time = time.time() - iteration_start_time
        avg_game_length = len(iteration_examples) / config.NUM_SELFPLAY_GAMES if config.NUM_SELFPLAY_GAMES > 0 else 0

        iteration_data = {
            'train_loss': loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'learning_rate': current_lr,
            'num_games': config.NUM_SELFPLAY_GAMES,
            'avg_game_length': avg_game_length,
            'replay_buffer_size': len(replay_buffer),
            'training_time_sec': iteration_time,
            'avg_policy_entropy': 0,
            'value_accuracy': 0
        }

        metrics.log_iteration(iteration, iteration_data)
        logger.log_iteration(iteration, iteration_data)

        print(f"⏱️  Iteration time: {iteration_time:.1f}s")

        if iteration % config.VALIDATION_INTERVAL == 0:
            print("\n=== VALIDATION (ARENA) ===")

            if best_model is None:
                print("⚠️  Brak best_model – ustawiam aktualny model jako bazowy.")
                best_model = copy.deepcopy(model)
                torch.save(
                    best_model.state_dict(),
                    os.path.join(config.MODEL_DIR, "best_model.pt")
                )
            else:
                arena_results = arena.compare_models(
                    new_model=model,
                    best_model=best_model,
                    config=config,
                    num_games=config.ARENA_GAMES,
                    save_path=str(logger.games_dir)
                )

                new_win_rate = arena_results["win_rate"]

                validation_results = {
                    "win_rate": new_win_rate,
                    "loss_rate": 1.0 - new_win_rate,
                    "draw_rate": arena_results["draws"] / arena_results["total_games"],
                    "avg_game_length": 0,
                    "elo_estimate": 0,
                    "vs_random_win_rate": 0
                }
                logger.log_validation(iteration, validation_results)

                print(f"\nArena win rate (current vs best): {new_win_rate:.1%}")

                if new_win_rate >= config.ARENA_THRESHOLD:
                    print("✅ New model beat best model – updating best_model.pt")
                    best_model = copy.deepcopy(model)
                    patience_counter = 0

                    torch.save(
                        best_model.state_dict(),
                        os.path.join(config.MODEL_DIR, "best_model.pt")
                    )
                else:
                    print("❌ Keeping previous best model")
                    patience_counter += 1
                    print(f"Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
                    if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                        print("⛔ Early stopping triggered (no arena improvement).")
                        break

        if iteration % (config.VALIDATION_INTERVAL * 2) == 0:
            print("📊 Generating plots...")
            logger.generate_all_plots()
            metrics.plot_training_progress()

        if iteration % config.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(
                config.MODEL_DIR,
                f'checkpoint_iter_{iteration}.pt'
            )
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'replay_buffer': replay_buffer,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("\nTraining complete!")

    print("\n📚 Exporting all data for thesis...")
    logger.export_for_thesis()

    print("\n" + "=" * 60)
    print("✅ ALL DATA SAVED!")
    print("=" * 60)
    print(f"📁 Experiment directory: {logger.exp_dir}")
    print(f"📊 Plots: {logger.plots_dir}")
    print(f"📄 Data (CSV): {logger.data_dir}")
    print(f"🎮 Games (PGN): {logger.games_dir}")
    print("=" * 60)

    writer.close()


if __name__ == '__main__':
    main()
