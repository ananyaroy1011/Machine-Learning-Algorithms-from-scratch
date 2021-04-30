import random

class RandomPlayer():

    def __init__(self):
        self.moves = []
    
    
    def generate_moves(self):
        self.moves = [i for i in range(1,10)]


    def draw_bead(self):
        move = random.choice(self.moves)
        self.moves.remove(move)
        return move

