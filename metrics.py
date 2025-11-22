import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsTracker:
    """
    Comprehensive metrics tracking for ChessBot training
    Tracks: loss, win rates, game lengths, move quality, etc.
    """

    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.metrics = defaultdict(list)
        self.iteration_data = []

        self.metrics_file = os.path.join(log_dir, "metrics.json")
        self.plots_dir = os.path.join(log_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        self.load_metrics()

    def log_iteration(self, iteration, data):
        data['iteration'] = iteration
        data['timestamp'] = datetime.now().isoformat()

        for key, value in data.items():
            if key not in ['iteration', 'timestamp']:
                self.metrics[key].append(value)

        self.iteration_data.append(data)
        self.save_metrics()

    def log_game(self, iteration, game_data):
        if 'games' not in self.metrics:
            self.metrics['games'] = []

        game_data['iteration'] = iteration
        game_data['timestamp'] = datetime.now().isoformat()
        self.metrics['games'].append(game_data)

    def log_validation(self, iteration, val_data):
        val_data['iteration'] = iteration
        val_data['timestamp'] = datetime.now().isoformat()

        if 'validation' not in self.metrics:
            self.metrics['validation'] = []
        self.metrics['validation'].append(val_data)

    def save_metrics(self):
        data = {
            'metrics': dict(self.metrics),
            'iterations': self.iteration_data
        }
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_metrics(self):
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                self.metrics = defaultdict(list, data.get('metrics', {}))
                self.iteration_data = data.get('iterations', [])
            except:
                print("Could not load existing metrics")

    def plot_training_progress(self, save=True):
        if not self.iteration_data:
            print("No data to plot yet")
            return

        iterations = [d['iteration'] for d in self.iteration_data]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ChessBot Training Progress', fontsize=16, fontweight='bold')

        ax = axes[0, 0]
        if 'train_loss' in self.metrics:
            ax.plot(iterations, self.metrics['train_loss'], 'b-', label='Total Loss', linewidth=2)
        if 'policy_loss' in self.metrics:
            ax.plot(iterations, self.metrics['policy_loss'], 'r--', label='Policy Loss', alpha=0.7)
        if 'value_loss' in self.metrics:
            ax.plot(iterations, self.metrics['value_loss'], 'g--', label='Value Loss', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if self.metrics.get('validation'):
            val_iters = [v['iteration'] for v in self.metrics['validation']]
            win_rates = [v.get('win_rate', 0) for v in self.metrics['validation']]
            draw_rates = [v.get('draw_rate', 0) for v in self.metrics['validation']]

            ax.plot(val_iters, win_rates, 'g-o', label='Win Rate', linewidth=2)
            ax.plot(val_iters, draw_rates, 'y-s', label='Draw Rate', linewidth=2)
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% baseline')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Rate')
            ax.set_title('Validation Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        ax = axes[0, 2]
        if 'avg_game_length' in self.metrics:
            ax.plot(iterations, self.metrics['avg_game_length'], 'purple', linewidth=2)
            ax.fill_between(iterations, self.metrics['avg_game_length'], alpha=0.3)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Moves')
            ax.set_title('Average Game Length')
            ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if 'learning_rate' in self.metrics:
            ax.semilogy(iterations, self.metrics['learning_rate'], 'orange', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Learning Rate (log scale)')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if 'value_accuracy' in self.metrics:
            ax.plot(iterations, self.metrics['value_accuracy'], 'teal', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Accuracy')
            ax.set_title('Value Head Accuracy')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        ax = axes[1, 2]
        if 'avg_policy_entropy' in self.metrics:
            ax.plot(iterations, self.metrics['avg_policy_entropy'], 'brown', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Entropy (nats)')
            ax.set_title('Average Policy Entropy')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.plots_dir, 'training_progress.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved training progress plot to {filepath}")

        return fig

    def plot_game_outcomes(self, save=True):
        if 'games' not in self.metrics or not self.metrics['games']:
            print("No game data available")
            return

        games = self.metrics['games']
        iterations = sorted(set(g['iteration'] for g in games))

        wins = []
        losses = []
        draws = []

        for it in iterations:
            iter_games = [g for g in games if g['iteration'] == it]
            total = len(iter_games)
            if total > 0:
                wins.append(sum(1 for g in iter_games if g.get('result', 0) == 1) / total)
                losses.append(sum(1 for g in iter_games if g.get('result', 0) == -1) / total)
                draws.append(sum(1 for g in iter_games if g.get('result', 0) == 0) / total)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(iterations))
        width = 0.6

        ax.bar(x, wins, width, label='Wins', color='green', alpha=0.8)
        ax.bar(x, draws, width, bottom=wins, label='Draws', color='yellow', alpha=0.8)
        ax.bar(x, losses, width, bottom=np.array(wins) + np.array(draws),
               label='Losses', color='red', alpha=0.8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Proportion')
        ax.set_title('Game Outcomes Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(iterations)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        if save:
            filepath = os.path.join(self.plots_dir, 'game_outcomes.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved game outcomes plot to {filepath}")

        return fig

    def plot_move_quality_heatmap(self, save=True):
        pass

    def generate_report(self):
        if not self.iteration_data:
            return "No training data available yet."

        latest = self.iteration_data[-1]

        report = []
        report.append("=" * 60)
        report.append("ChessBot Training Report")
        report.append("=" * 60)
        report.append(f"Total Iterations: {len(self.iteration_data)}")
        report.append(f"Latest Iteration: {latest['iteration']}")
        report.append(f"Timestamp: {latest['timestamp']}")
        report.append("")

        report.append("Latest Metrics:")
        report.append("-" * 40)
        for key, value in latest.items():
            if key not in ['iteration', 'timestamp']:
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")

        report.append("")
        report.append("Training Summary:")
        report.append("-" * 40)

        if 'train_loss' in self.metrics and len(self.metrics['train_loss']) > 1:
            initial_loss = self.metrics['train_loss'][0]
            final_loss = self.metrics['train_loss'][-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            report.append(f"  Loss Improvement: {improvement:.2f}%")

        if self.metrics.get('validation'):
            best_wr = max(v.get('win_rate', 0) for v in self.metrics['validation'])
            report.append(f"  Best Validation Win Rate: {best_wr:.2%}")

        if 'games' in self.metrics:
            total_games = len(self.metrics['games'])
            report.append(f"  Total Games Played: {total_games}")

        report.append("=" * 60)

        report_text = "\n".join(report)

        report_file = os.path.join(self.log_dir, 'training_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)

        return report_text

    def get_summary_stats(self):
        if not self.iteration_data:
            return {}

        stats = {
            'total_iterations': len(self.iteration_data),
            'total_games': len(self.metrics.get('games', [])),
        }

        if 'train_loss' in self.metrics:
            stats['final_loss'] = self.metrics['train_loss'][-1]
            stats['best_loss'] = min(self.metrics['train_loss'])

        if self.metrics.get('validation'):
            win_rates = [v.get('win_rate', 0) for v in self.metrics['validation']]
            stats['best_win_rate'] = max(win_rates)
            stats['current_win_rate'] = win_rates[-1]

        return stats


def create_comparison_plot(metrics_files, labels, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for metrics_file, label in zip(metrics_files, labels):
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        metrics = data.get('metrics', {})
        iterations = data.get('iterations', [])

        if iterations:
            iters = [d['iteration'] for d in iterations]

            if 'train_loss' in metrics:
                axes[0].plot(iters, metrics['train_loss'], label=label, linewidth=2)

            val_data = metrics.get('validation', [])
            if val_data:
                val_iters = [v['iteration'] for v in val_data]
                win_rates = [v.get('win_rate', 0) for v in val_data]
                axes[1].plot(val_iters, win_rates, label=label, linewidth=2, marker='o')

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Win Rate')
    axes[1].set_title('Validation Win Rate Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    return fig
