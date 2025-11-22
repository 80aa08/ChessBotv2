import datetime
import os
import chess
import numpy as np
import torch
import chess.pgn as pgn
from config import Config

def move_to_index(move: chess.Move) -> int:
    """
    Convert chess.Move to policy index (0-4671)
    Encoding:
    - Normal moves: from_square * 64 + to_square (0-4095)
    - Underpromotions: 4096 + from_square * 9 + promotion_offset (4096-4671)

    Note: Queen promotions use the normal move encoding
    """
    from_sq = move.from_square
    to_sq = move.to_square

    if move.promotion is None or move.promotion == chess.QUEEN:
        # Normal move or queen promotion
        return from_sq * 64 + to_sq
    else:
        # Underpromotion (knight, bishop, rook)
        promotion_map = {
            chess.KNIGHT: 0,
            chess.BISHOP: 1,
            chess.ROOK: 2,
        }
        # Each from_square has 9 possible underpromotions (3 pieces * 3 directions)
        # Actually simpler: just 3 pieces per from_square
        base = 4096
        promo_offset = promotion_map[move.promotion]
        # For simplicity: from_square * 3 + promo_type
        return base + from_sq * 3 + promo_offset


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Convert policy index back to chess.Move
    Returns None if move is illegal in current position
    """
    if index < 4096:
        # Normal move
        from_sq = index // 64
        to_sq = index % 64
        move = chess.Move(from_sq, to_sq)

        # Check if it's a pawn promotion to queen
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(to_sq)
            if (piece.color == chess.WHITE and rank == 7) or \
               (piece.color == chess.BLACK and rank == 0):
                move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

        return move if move in board.legal_moves else None
    else:
        # Underpromotion
        offset = index - 4096
        from_sq = offset // 3
        promo_type = offset % 3

        promotion_map = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        promotion = promotion_map[promo_type]

        # Find valid to_square (pawn promotion)
        piece = board.piece_at(from_sq)
        if not piece or piece.piece_type != chess.PAWN:
            return None

        # Check possible promotion squares
        for to_sq in [from_sq + 8, from_sq - 8, from_sq + 7, from_sq - 7,
                      from_sq + 9, from_sq - 9]:
            if 0 <= to_sq < 64:
                move = chess.Move(from_sq, to_sq, promotion=promotion)
                if move in board.legal_moves:
                    return move

        return None


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """
    Create a mask of legal moves for the current position
    Returns: numpy array of shape (4672,) with 1 for legal moves, 0 otherwise
    """
    mask = np.zeros(Config.POLICY_OUTPUT_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = 1.0
    return mask


def get_policy_from_visits(visit_counts: dict, legal_moves: list) -> np.ndarray:
    """
    Convert MCTS visit counts to policy vector

    Args:
        visit_counts: dict mapping chess.Move to visit count
        legal_moves: list of legal chess.Move objects

    Returns:
        numpy array of shape (4672,) representing policy distribution
    """
    policy = np.zeros(Config.POLICY_OUTPUT_SIZE, dtype=np.float32)
    total_visits = sum(visit_counts.values())

    if total_visits == 0:
        return policy

    for move in legal_moves:
        if move in visit_counts:
            idx = move_to_index(move)
            policy[idx] = visit_counts[move] / total_visits

    return policy


def save_pgn(env, path="./games", prefix="game", iteration=None):
    """Save game to PGN file"""
    os.makedirs(path, exist_ok=True)

    game = pgn.Game()
    game.headers["Event"] = "Self-Play Training"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "ChessBot"
    game.headers["Black"] = "ChessBot"

    if iteration is not None:
        game.headers["Round"] = str(iteration)

    # Add moves
    node = game
    board = chess.Board()
    for move in env.move_history:
        node = node.add_variation(move)
        board.push(move)

    game.headers["Result"] = board.result()

    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{timestamp}.pgn"

    with open(os.path.join(path, fname), 'w') as f:
        print(game, file=f)


class ReplayBuffer:
    """
    Circular buffer for storing (state, policy, value) examples
    """
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def add(self, examples):
        """Add list of (state, policy, value) tuples"""
        for example in examples:
            if len(self.buffer) < self.max_size:
                self.buffer.append(example)
            else:
                self.buffer[self.position] = example
                self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        """Sample random batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

    def get_all(self):
        """Return all examples"""
        return self.buffer.copy()


def temperature_sample(visit_counts: dict, temperature: float = 1.0) -> chess.Move:
    """
    Sample move from visit count distribution with temperature

    Args:
        visit_counts: dict mapping chess.Move to visit count
        temperature:
            - temp=1: proportional to visit counts
            - temp→0: choose most visited (argmax)
            - temp>1: more exploration

    Returns:
        Selected chess.Move
    """
    moves = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)

    if temperature < 0.01:
        # Greedy: choose most visited
        return moves[np.argmax(counts)]

    # Apply temperature
    counts = counts ** (1.0 / temperature)
    probs = counts / counts.sum()

    # Sample
    return np.random.choice(moves, p=probs)
