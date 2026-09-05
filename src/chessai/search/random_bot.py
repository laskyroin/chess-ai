import chess
import random as rd
from .base import Bot

class RandomBot(Bot):

    name = "random"

    def __init__(self, seed):
        self.rng = rd.Random(seed)

    def select_move(self, board):
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("no legal move available in this position")
        return self.rng.choice(moves)