import chess
import random as rd

class Bot:
    name: str = "bot"

    def select_move(self, board):
        raise NotImplementedError

    def __repr__(self):
         return f"<{type(self).__name__} name={self.name!r}>"


