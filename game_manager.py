import os
import re
from copy import deepcopy

from tqdm import tqdm

from menace import Menace
from random_player import RandomPlayer
import argparse

class GameManager:

    def __init__(self,save_weights=False, quiet=True):
        self.move_map = {
            1: (0,0),
            2: (0,1),
            3: (0,2),
            4: (1,0),
            5: (1,1),
            6: (1,2),
            7: (2,0),
            8: (2,1),
            9: (2,2),
        }
        self.save_weights = save_weights
        self.quiet = quiet


    def generate_empty_board(self):
        board = []
        for i in range(3):
            row = []
            for j in range(3):
                row.append('-')
            board.append(row)
        return board


    def display_board(self,board):
        for i in board:
            for j in i:
                print(j,end='     ')
            print('\n')


    def is_game_over(self,board):
        # First row
        if (board[0][0] == board[0][1] == board[0][2]) and (board[0][0] != '-' and board[0][1] != '-' and board[0][2] != '-'):
            return board[0][0]
        
        # Second row
        if (board[1][0] == board[1][1] == board[1][2]) and (board[1][0] != '-' and board[1][1] != '-' and board[1][2] != '-'):
            return board[1][0]
   
        # Third row
        if (board[2][0] == board[2][1] == board[2][2]) and (board[2][0] != '-' and board[2][1] != '-' and board[2][2] != '-'):
            return board[2][0]

        # First column
        if (board[0][0] == board[1][0] == board[2][0]) and (board[0][0] != '-' and board[1][0] != '-' and board[2][0] != '-'):
            return board[0][0]
        
        # Second column
        if (board[0][1] == board[1][1] == board[2][1]) and (board[0][1] != '-' and board[1][1] != '-' and board[2][1] != '-'):
            return board[0][1]

        # Third column
        if (board[0][2] == board[1][2] == board[2][2]) and (board[0][2] != '-' and board[1][2] != '-' and board[2][2] != '-'):
            return board[0][2]

        # Top left to bottom right diagonal
        if (board[0][0] == board[1][1] == board[2][2]) and (board[0][0] != '-' and board[1][1] != '-' and board[2][2] != '-'):
            return board[0][0]

        # Top right to bottom left diagonal
        if (board[0][2] == board[1][1] == board[2][0]) and (board[0][2] != '-' and board[1][1] != '-' and board[2][0] != '-'):
            return board[0][2]

        # Draw
        if board[0][0] != '-' and board[0][1] != '-' and board[0][2] != '-' and board[1][0] != '-' and board[1][1] != '-' and board[1][2] != '-' and board[2][0] != '-' and board[2][1] != '-' and board[2][2] != '-':
            return '-'

        return None
        

    def is_move_valid(self,move,board):
        if re.match("^[1-9]$", str(move)) and board[self.move_map[move][0]][self.move_map[move][1]] == '-':
            return True
        return False


    def update_board(self,move,current_player,board):
        board[self.move_map[move][0]][self.move_map[move][1]] = current_player
        return board
        

    def menace_vs_menace(self,M1,M2,num_games):
        num_losses = 0
        for n in tqdm(range(num_games)):
            M1.matchboxes_to_update = []
            M2.matchboxes_to_update = []
            board = self.generate_empty_board()
            current_player = 'X'
            
            while 1:
                # time.sleep(0.5)
                not self.quiet and self.display_board(board)
                result = self.is_game_over(board)

                if result is not None:   
                    if result == 'X':
                        not self.quiet and print('MENACE is the winner!')
                        M1.menace_is_the_winner(M1.matchboxes_to_update)
                    elif result == 'O':
                        M2.menace_is_the_winner(M2.matchboxes_to_update)
                        not self.quiet and print('MENACE lost!')
                        num_losses += 1
                    else:
                        not self.quiet and print('Draw!')
                        M1.game_draw(M1.matchboxes_to_update)
                        M2.game_draw(M2.matchboxes_to_update)
                    break

                else:
                    move_successful = False
                    not self.quiet and print(current_player+"'s turn")

                    if current_player == 'X':          
                        move = M1.draw_bead(board)
                        not self.quiet and print('X PLAYED:',move)
                        if self.is_move_valid(move,board):
                            M1.matchboxes_to_update.append({'move':move,'board':deepcopy(board)})
                            move_successful = True

                    elif current_player == 'O':
                        move = M2.draw_bead(board)
                        not self.quiet and print('O PLAYED:',move)
                        if self.is_move_valid(move,board):
                            M2.matchboxes_to_update.append({'move':move,'board':deepcopy(board)})
                            move_successful = True
                    
                    if move_successful is True:
                        board = self.update_board(move,current_player,board)
                        # os.system('cls' if os.name == 'nt' else 'clear')
                        not self.quiet and print(current_player + ' played at '+str(move)+'\n')
                        current_player = 'O' if current_player == 'X' else 'X'
                    else:
                        not self.quiet and print('\nInvalid move')

        if self.save_weights is True:
            M1.save_state()
            M2.save_state()
        print('Loss Rate:',num_losses/num_games)


    def menace_vs_random(self,M1,R,num_games):
        num_losses = 0
        for n in tqdm(range(num_games)):
            R.generate_moves()
            M1.matchboxes_to_update = []
            board = self.generate_empty_board()
            current_player = 'X'
            
            while 1:
                # time.sleep(0.5)
                not self.quiet and self.display_board(board)
                result = self.is_game_over(board)

                if result is not None:   
                    if result == 'X':
                        not self.quiet and print('MENACE is the winner!')
                        M1.menace_is_the_winner(M1.matchboxes_to_update)
                    elif result == 'O':
                        not self.quiet and print('MENACE lost!')
                        num_losses += 1
                    else:
                        not self.quiet and print('Draw!')
                        M1.game_draw(M1.matchboxes_to_update)
                    # return result
                    break

                else:
                    move_successful = False
                    not self.quiet and print(current_player+"'s turn")

                    if current_player == 'X':          
                        move = M1.draw_bead(board)
                        not self.quiet and print('X PLAYED:',move)
                        if self.is_move_valid(move,board):
                            M1.matchboxes_to_update.append({'move':move,'board':deepcopy(board)})
                            move_successful = True

                    elif current_player == 'O':
                        move = R.draw_bead()
                        if self.is_move_valid(move,board):
                            move_successful = True
                            
                    if move_successful is True:
                        board = self.update_board(move,current_player,board)
                        # os.system('cls' if os.name == 'nt' else 'clear')
                        not self.quiet and print(current_player + ' played at '+str(move)+'\n')
                        current_player = 'O' if current_player == 'X' else 'X'
                    else:
                        not self.quiet and print('\nInvalid move')

        if self.save_weights is True:
            M1.save_state()
        print('Loss Rate:',num_losses/num_games)


    def menace_vs_human(self,M1):
        M1.matchboxes_to_update = []
        board = self.generate_empty_board()
        current_player = 'X'
        
        while 1:
            # time.sleep(0.5)
            self.display_board(board)
            result = self.is_game_over(board)

            if result is not None:   
                if result == 'X':
                    not self.quiet and print('MENACE is the winner!')
                    M1.menace_is_the_winner(M1.matchboxes_to_update)
                elif result == 'O':
                    not self.quiet and print('MENACE lost!')
                else:
                    not self.quiet and print('Draw!')
                    M1.game_draw(M1.matchboxes_to_update)
                # return result
                break

            else:
                move_successful = False
                not self.quiet and print(current_player+"'s turn")

                if current_player == 'X':          
                    move = M1.draw_bead(board)
                    not self.quiet and print('X PLAYED:',move)
                    if self.is_move_valid(move,board):
                        M1.matchboxes_to_update.append({'move':move,'board':deepcopy(board)})
                        move_successful = True

                elif current_player == 'O':
                    move = int(input('Enter move: '))
                    if self.is_move_valid(move,board):
                        move_successful = True
                
                if move_successful is True:
                    board = self.update_board(move,current_player,board)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    not self.quiet and print(current_player + ' played at '+str(move)+'\n')
                    current_player = 'O' if current_player == 'X' else 'X'
                else:
                    not self.quiet and print('\nInvalid move')

        if self.save_weights is True:
            M1.save_state()


os.system('cls' if os.name == 'nt' else 'clear')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help="'play' against MENACE or 'train' MENACE")
parser.add_argument('--trainer', help="train MENACE using a 'random' player or another 'menace'",default=None)
parser.add_argument('--episodes', help="number of episodes for training",default=1000)
args = parser.parse_args()

if args.mode == 'play':
    G = GameManager(save_weights=True, quiet=False)
    M1 = Menace(reset_weights=False, weights_path='weights/menace1.json')
    G.menace_vs_human(M1)

elif args.mode == 'train':
    if args.trainer == 'random':
        G = GameManager(save_weights=True, quiet=True)
        M1 = Menace(reset_weights=True, weights_path='weights/menace1.json')
        R = RandomPlayer()
        G.menace_vs_random(M1, R, num_games=int(args.episodes))

    elif args.trainer == 'menace':
        G = GameManager(save_weights=True, quiet=True)
        M1 = Menace(reset_weights=True, weights_path='weights/menace1.json')
        M2 = Menace(reset_weights=True, weights_path='weights/menace2.json')
        G.menace_vs_menace(M1, M2, int(args.episodes))
    print('Trained!')

else:
    print('Invalid Arguments')