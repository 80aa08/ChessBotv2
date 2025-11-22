import json
import csv
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DataLogger:
    def __init__(self, experiment_name="chess_training", base_dir="./experiments"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.exp_dir = Path(base_dir) / f"{experiment_name}_{self.timestamp}"
        self.data_dir = self.exp_dir / "data"
        self.plots_dir = self.exp_dir / "plots"
        self.models_dir = self.exp_dir / "models"
        self.games_dir = self.exp_dir / "games"

        for dir_path in [self.data_dir, self.plots_dir, self.models_dir, self.games_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.iteration_log = self.data_dir / "iterations.csv"
        self.game_log = self.data_dir / "games.csv"
        self.validation_log = self.data_dir / "validation.csv"
        self.summary_file = self.data_dir / "summary.json"

        self._init_csv_files()

        self.iterations_data = []
        self.games_data = []
        self.validation_data = []

        print(f"📁 Experiment directory: {self.exp_dir}")

    def _init_csv_files(self):
        with open(self.iteration_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'timestamp', 'train_loss', 'policy_loss', 'value_loss',
                'learning_rate', 'num_games', 'avg_game_length', 'replay_buffer_size',
                'avg_policy_entropy', 'value_accuracy', 'training_time_sec'
            ])

        with open(self.game_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'game_id', 'result', 'length', 'white_win',
                'black_win', 'draw', 'avg_move_time', 'opening'
            ])

        with open(self.validation_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'timestamp', 'win_rate', 'loss_rate', 'draw_rate',
                'avg_game_length', 'elo_estimate', 'vs_random_win_rate'
            ])

    def log_iteration(self, iteration, data):
        timestamp = datetime.now().isoformat()

        with open(self.iteration_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                timestamp,
                data.get('train_loss', 0),
                data.get('policy_loss', 0),
                data.get('value_loss', 0),
                data.get('learning_rate', 0),
                data.get('num_games', 0),
                data.get('avg_game_length', 0),
                data.get('replay_buffer_size', 0),
                data.get('avg_policy_entropy', 0),
                data.get('value_accuracy', 0),
                data.get('training_time_sec', 0)
            ])

        data['iteration'] = iteration
        data['timestamp'] = timestamp
        self.iterations_data.append(data)

        print(f"📊 Iteracja {iteration} zapisana")

    def log_game(self, iteration, game_id, result, length, opening="Unknown"):
        with open(self.game_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                game_id,
                result,
                length,
                1 if result == 1 else 0,
                1 if result == -1 else 0,
                1 if result == 0 else 0,
                0,
                opening
            ])

        self.games_data.append({
            'iteration': iteration,
            'game_id': game_id,
            'result': result,
            'length': length,
            'opening': opening
        })

    def log_validation(self, iteration, results):
        timestamp = datetime.now().isoformat()

        with open(self.validation_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                timestamp,
                results.get('win_rate', 0),
                results.get('loss_rate', 0),
                results.get('draw_rate', 0),
                results.get('avg_game_length', 0),
                results.get('elo_estimate', 0),
                results.get('vs_random_win_rate', 0)
            ])

        results['iteration'] = iteration
        results['timestamp'] = timestamp
        self.validation_data.append(results)

    def generate_all_plots(self):
        print("📈 Generowanie wykresów...")

        if not self.iterations_data:
            print("⚠️ Brak danych")
            return

        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

        self._plot_training_losses()
        self._plot_win_rates()
        self._plot_game_statistics()
        self._plot_learning_curves()
        self._plot_validation_metrics()
        self._plot_combined_overview()

        print(f"✅ Wszystkie zapisane: {self.plots_dir}")

    def _plot_training_losses(self):
        iterations = [d['iteration'] for d in self.iterations_data]
        train_loss = [d.get('train_loss', 0) for d in self.iterations_data]
        policy_loss = [d.get('policy_loss', 0) for d in self.iterations_data]
        value_loss = [d.get('value_loss', 0) for d in self.iterations_data]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(iterations, train_loss, 'b-', linewidth=2, label='Total Loss', marker='o', markersize=4)
        ax.plot(iterations, policy_loss, 'r--', linewidth=2, label='Policy Loss', alpha=0.7)
        ax.plot(iterations, value_loss, 'g--', linewidth=2, label='Value Loss', alpha=0.7)

        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
        ax.set_title('Training Loss Over Time', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_losses.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'training_losses.pdf', bbox_inches='tight')
        plt.close()

    def _plot_win_rates(self):
        if not self.validation_data:
            return

        iterations = [d['iteration'] for d in self.validation_data]
        win_rates = [d.get('win_rate', 0) * 100 for d in self.validation_data]
        draw_rates = [d.get('draw_rate', 0) * 100 for d in self.validation_data]
        loss_rates = [d.get('loss_rate', 0) * 100 for d in self.validation_data]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(iterations, win_rates, 'g-', linewidth=3, label='Win Rate', marker='o', markersize=6)
        ax.plot(iterations, draw_rates, 'y-', linewidth=2, label='Draw Rate', marker='s', markersize=5)
        ax.plot(iterations, loss_rates, 'r-', linewidth=2, label='Loss Rate', marker='^', markersize=5)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Baseline')

        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Rate (%)', fontsize=14, fontweight='bold')
        ax.set_title('Model Performance: Win/Draw/Loss Rates', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'win_rates.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'win_rates.pdf', bbox_inches='tight')
        plt.close()

    def _plot_game_statistics(self):
        iterations = [d['iteration'] for d in self.iterations_data]
        avg_lengths = [d.get('avg_game_length', 0) for d in self.iterations_data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(iterations, avg_lengths, 'purple', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Moves', fontsize=12, fontweight='bold')
        ax1.set_title('Average Game Length', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        if self.games_data:
            results = [g['result'] for g in self.games_data]
            wins = results.count(1)
            losses = results.count(-1)
            draws = results.count(0)

            ax2.bar(['Wins\n(White)', 'Losses\n(Black)', 'Draws'],
                   [wins, losses, draws],
                   color=['green', 'red', 'yellow'],
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=2)
            ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax2.set_title('Game Outcomes Distribution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'game_statistics.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'game_statistics.pdf', bbox_inches='tight')
        plt.close()

    def _plot_learning_curves(self):
        iterations = [d['iteration'] for d in self.iterations_data]
        value_accuracy = [d.get('value_accuracy', 0) * 100 for d in self.iterations_data]
        policy_entropy = [d.get('avg_policy_entropy', 0) for d in self.iterations_data]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        ax1.plot(iterations, value_accuracy, 'teal', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Value Head Accuracy', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])

        ax2.plot(iterations, policy_entropy, 'brown', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Entropy (nats)', fontsize=12, fontweight='bold')
        ax2.set_title('Policy Entropy (Exploration)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'learning_curves.pdf', bbox_inches='tight')
        plt.close()

    def _plot_validation_metrics(self):
        if not self.validation_data:
            return

        iterations = [d['iteration'] for d in self.validation_data]
        elo = [d.get('elo_estimate', 0) for d in self.validation_data]
        vs_random = [d.get('vs_random_win_rate', 0) * 100 for d in self.validation_data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if any(elo):
            ax1.plot(iterations, elo, 'navy', linewidth=3, marker='D', markersize=6)
            ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax1.set_ylabel('ELO Rating', fontsize=12, fontweight='bold')
            ax1.set_title('Estimated ELO Rating', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)

        if any(vs_random):
            ax2.plot(iterations, vs_random, 'darkgreen', linewidth=3, marker='o', markersize=6)
            ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random baseline')
            ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
            ax2.set_title('Performance vs Random Player', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'validation_metrics.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'validation_metrics.pdf', bbox_inches='tight')
        plt.close()

    def _plot_combined_overview(self):
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        iterations = [d['iteration'] for d in self.iterations_data]

        ax1 = fig.add_subplot(gs[0, 0])
        train_loss = [d.get('train_loss', 0) for d in self.iterations_data]
        ax1.plot(iterations, train_loss, 'b-', linewidth=2)
        ax1.set_title('Training Loss', fontweight='bold')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        if self.validation_data:
            val_iters = [d['iteration'] for d in self.validation_data]
            win_rates = [d.get('win_rate', 0) * 100 for d in self.validation_data]
            ax2.plot(val_iters, win_rates, 'g-', linewidth=2, marker='o')
            ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5)
        ax2.set_title('Win Rate', fontweight='bold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Win Rate (%)')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        avg_lengths = [d.get('avg_game_length', 0) for d in self.iterations_data]
        ax3.plot(iterations, avg_lengths, 'purple', linewidth=2)
        ax3.set_title('Average Game Length', fontweight='bold')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Moves')
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        lr = [d.get('learning_rate', 0) for d in self.iterations_data]
        ax4.semilogy(iterations, lr, 'orange', linewidth=2)
        ax4.set_title('Learning Rate Schedule', fontweight='bold')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Learning Rate (log)')
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[2, :])
        policy_loss = [d.get('policy_loss', 0) for d in self.iterations_data]
        value_loss = [d.get('value_loss', 0) for d in self.iterations_data]
        ax5.plot(iterations, policy_loss, 'r-', linewidth=2, label='Policy Loss', alpha=0.8)
        ax5.plot(iterations, value_loss, 'g-', linewidth=2, label='Value Loss', alpha=0.8)
        ax5.set_title('Loss Components', fontweight='bold')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Loss')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        fig.suptitle('ChessBot Training Overview', fontsize=18, fontweight='bold', y=0.995)
        plt.savefig(self.plots_dir / 'combined_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'combined_overview.pdf', bbox_inches='tight')
        plt.close()

    def save_summary(self):
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.timestamp,
            'end_time': datetime.now().isoformat(),
            'total_iterations': len(self.iterations_data),
            'total_games': len(self.games_data),
            'total_validations': len(self.validation_data),
            'final_metrics': self.iterations_data[-1] if self.iterations_data else {},
            'best_win_rate': max([d.get('win_rate', 0) for d in self.validation_data]) if self.validation_data else 0,
            'experiment_directory': str(self.exp_dir)
        }

        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"💾 Podsumowanie zapisane: {self.summary_file}")
        return summary

    def export_for_thesis(self):
        print("📚 Export danych...")

        self.generate_all_plots()

        summary = self.save_summary()

        readme_path = self.exp_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"# ChessBot Experiment: {self.experiment_name}\n\n")
            f.write(f"**Date:** {self.timestamp}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- Total iterations: {summary['total_iterations']}\n")
            f.write(f"- Total games: {summary['total_games']}\n")
            f.write(f"- Best win rate: {summary['best_win_rate']:.2%}\n\n")
            f.write(f"## Data Files\n\n")
            f.write(f"- `data/iterations.csv` - Training metrics per iteration\n")
            f.write(f"- `data/games.csv` - Individual game records\n")
            f.write(f"- `data/validation.csv` - Validation results\n")
            f.write(f"- `data/summary.json` - Experiment summary\n\n")
            f.write(f"## Plots\n\n")
            f.write(f"All plots available in `plots/` directory in PNG (300 DPI) and PDF formats.\n")

        print(f"✅ Export ukończony!")
        print(f"📁 Wszystkie pliki w: {self.exp_dir}")
        print(f"📊 Wykresy: {self.plots_dir}")
        print(f"📄 Dane: {self.data_dir}")
