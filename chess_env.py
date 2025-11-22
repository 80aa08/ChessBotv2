import chess
import torch
import numpy as np
from config import Config

class ChessEnv:
    def __init__(self, device=torch.device('cpu')):
        self.board = chess.Board()
        self.device = device
        self.move_history = []

    def reset(self):
        self.board.reset()
        self.move_history = []
        return self._get_state()

    def step(self, move):
        if move not in self.board.legal_moves:
            raise ValueError(f"Nielegalny ruch: {move}")

        self.board.push(move)
        self.move_history.append(move)

        reward = 0
        done = self.board.is_game_over()

        if done:
            result = self.board.result()
            if result == '1-0':
                reward = 1 if self.board.turn == chess.BLACK else -1
            elif result == '0-1':
                reward = -1 if self.board.turn == chess.BLACK else 1
            else:
                reward = 0  # Draw

        return self._get_state(), reward, done

    def _get_state(self):
        """
        Encode board state as tensor:
        - 12 planes for pieces (6 white, 6 black)
        - 1 plane for current player color
        - 2 planes for castling rights (current player, opponent)
        - 1 plane for en passant
        - 1 plane for move count (50-move rule)
        Total: 17 channels
        """
        planes = np.zeros((17, 8, 8), dtype=np.float32)

        for sq, piece in self.board.piece_map().items():
            idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
            row, col = divmod(sq, 8)
            planes[idx, row, col] = 1

        planes[12, :, :] = int(self.board.turn)

        if self.board.turn == chess.WHITE:
            planes[13, :, :] = int(self.board.has_kingside_castling_rights(chess.WHITE))
            planes[13, :, :] += int(self.board.has_queenside_castling_rights(chess.WHITE))
            planes[14, :, :] = int(self.board.has_kingside_castling_rights(chess.BLACK))
            planes[14, :, :] += int(self.board.has_queenside_castling_rights(chess.BLACK))
        else:
            planes[13, :, :] = int(self.board.has_kingside_castling_rights(chess.BLACK))
            planes[13, :, :] += int(self.board.has_queenside_castling_rights(chess.BLACK))
            planes[14, :, :] = int(self.board.has_kingside_castling_rights(chess.WHITE))
            planes[14, :, :] += int(self.board.has_queenside_castling_rights(chess.WHITE))

        if self.board.ep_square is not None:
            row, col = divmod(self.board.ep_square, 8)
            planes[15, row, col] = 1

        planes[16, :, :] = self.board.halfmove_clock / 100.0

        return torch.tensor(planes, dtype=torch.float32, device=self.device)

    def legal_moves(self):
        return list(self.board.legal_moves)

    def clone(self):
        new_env = ChessEnv(device=self.device)
        new_env.board = self.board.copy()
        new_env.move_history = self.move_history.copy()
        return new_env
