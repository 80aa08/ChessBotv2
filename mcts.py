import math
import torch
import numpy as np
from config import Config
from utils import move_to_index, get_legal_move_mask
import chess

class Node:
    """Node in MCTS tree"""
    def __init__(self, prior):
        self.P = prior          # Prior probability from NN
        self.N = 0              # Visit count
        self.W = 0              # Total action value
        self.Q = 0              # Mean action value (W/N)
        self.children = {}      # Map: chess.Move -> Node

    def expanded(self):
        return len(self.children) > 0

    def select_child(self, c_puct):
        """Select child with highest UCB score"""
        total_N = sum(child.N for child in self.children.values())

        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            U = c_puct * child.P * math.sqrt(total_N) / (1 + child.N)
            score = child.Q + U

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def expand(self, policy, legal_moves):
        for move in legal_moves:
            idx = move_to_index(move)
            prior = policy[idx]
            self.children[move] = Node(prior)

    def update(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N


class MCTS:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

    def search(self, env, add_noise=False):
        root = Node(prior=0)

        state = env._get_state().unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.model(state)
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        legal_moves = env.legal_moves()

        legal_mask = get_legal_move_mask(env.board)
        policy = policy * legal_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            policy = legal_mask / legal_mask.sum()

        root.expand(policy, legal_moves)

        if add_noise:
            noise = np.random.dirichlet([self.config.DIRICHLET_ALPHA] * len(legal_moves))
            epsilon = self.config.EPSILON

            for i, move in enumerate(legal_moves):
                child = root.children[move]
                child.P = (1 - epsilon) * child.P + epsilon * noise[i]

        for _ in range(self.config.NUM_SIMULATIONS):
            env_copy = env.clone()
            self._simulate(env_copy, root)

        visit_counts = {move: child.N for move, child in root.children.items()}
        return visit_counts

    def _simulate(self, env, node):
        if env.board.is_game_over():
            return self._get_game_result(env)

        if not node.expanded():
            return self._evaluate_and_expand(env, node)

        move, child = node.select_child(self.config.C_PUCT)

        env.step(move)

        value = -self._simulate(env, child)

        child.update(value)

        return value

    def _evaluate_and_expand(self, env, node):
        state = env._get_state().unsqueeze(0)

        with torch.no_grad():
            policy_logits, value = self.model(state)
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()

        legal_moves = env.legal_moves()
        legal_mask = get_legal_move_mask(env.board)

        policy = policy * legal_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            policy = legal_mask / legal_mask.sum()

        node.expand(policy, legal_moves)

        return value

    def _get_game_result(self, env):
        result = env.board.result()

        if result == '1-0':
            return 1.0 if env.board.turn == chess.WHITE else -1.0
        elif result == '0-1':
            return -1.0 if env.board.turn == chess.WHITE else 1.0
        else:
            return 0.0


class MCTSBatched(MCTS):

    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.eval_queue = []

    def search(self, env, add_noise=False):
        """Run MCTS with batched evaluation"""
        return super().search(env, add_noise)
