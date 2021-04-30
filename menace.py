import itertools
import random
import json
import os

class Menace():

    def __init__(self,reset_weights=False,weights_path='weights/menace1.json'):
        if not os.path.exists('weights'):
            os.makedirs('weights')
        self.matchboxes_to_update = []
        self.MENACE_PATH = weights_path
        if reset_weights is False:
            self.load_state()
        else:
            self.matchboxes = self.generate_matchboxes_with_beads()


    def generate_matchboxes_with_beads(self):
        boards = [list(p) for p in itertools.product([list(p) for p in itertools.product(['X','O','-'], repeat=3)], repeat=3)]
        initial_beads = [i for i in range(1,10)]
        
        matchboxes = []
        for i in range(len(boards)):
            # beads = [initial_beads for i in range(len(boards))]
            matchboxes.append({'board':boards[i],'beads':[1,2,3,4,5,6,7,8,9]})
        return matchboxes


    def draw_bead(self,board):
        for mb in self.matchboxes:
            if mb['board'] == board:
                if len(mb['beads']) == 0:
                    return random.choice([1,2,3,4,5,6,7,8,9])
                chosen_bead = random.choice(mb['beads'])
                mb['beads'].remove(chosen_bead)
                return chosen_bead
     
    
    def view_matchbox(self,board):
        for mb in self.matchboxes:
            if mb['board'] == board:
                print('MATCHBOX',mb['beads'])
                break

    
    def menace_is_the_winner(self,matchboxes_to_update):
        for k in matchboxes_to_update:
            # print(k)
            for mb in self.matchboxes:
                if mb['board'] == k['board']:
                    if len(mb['beads']) <= 50:
                        mb['beads'].extend([k['move'] for _ in range(3)])
                        # print(mb)
                    break
            # print()

    
    def game_draw(self,matchboxes_to_update):
        for k in matchboxes_to_update:
            for mb in self.matchboxes:
                if mb['board'] == k['board']:
                    mb['beads'].append(k['move'])
                    break

    
    def save_state(self):
        with open(self.MENACE_PATH, 'w') as file:
            json.dump(self.matchboxes,file)


    def load_state(self):
        with open(self.MENACE_PATH, 'r') as file:
            self.matchboxes = json.load(file)
                    

