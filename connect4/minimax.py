# Connect 4 implementation for MCTS and Minimax.
# University of Essex.
# M. Fairbank November 2021 for course CE811 Game Artificial Intelligence
# 
# Acknowedgements: 
# All of the graphics and some other code for the main game loop and minimax came from https://github.com/KeithGalli/Connect4-Python
# Some of the connect4Board logic and MCTS algorithm came from https://github.com/floriangardin/connect4-mcts 
# Other designs are implemented from the Millington and Funge Game AI textbook chapter on Minimax.
import random
from connect4Board import Board
import math

def static_evaluator(board, piece):
    # static evaluator function, to estimate how good the board is from the point of view of player "piece"
    # On entry, piece will be either 1 or 2

    grid = board.grid
    opponent = 2 if piece == 1 else 1
    score = 0
    lines_of_4 = []

    # find all possible horizontal lines of 4
    for i in range(0, 6):
        for j in range(0, 4):
          lines_of_4.append({"type": "horizontal", "indices": [(i, j), (i, j+1), (i, j+2), (i, j+3)]})

    # find all possible vertical lines of 4
    for i in range(0, 7):
        for j in range(0, 3):
            lines_of_4.append({"type": "vertical", "indices": [(j, i), (j+1, i), (j+2, i), (j+3, i)]})
    
    # find all diagonals part 1
    for i in range(3, 6):
        for j in range(4):
            lines_of_4.append({"type": "diagonal", "indices": [(i, j), (i-1, j+1), (i-2,j+2), (i-3,j+3)]})

    # find all diagonals part 2
    for i in range(0, 3):
        for j in range(0, 4):
            lines_of_4.append({"type": "diagonal", "indices": [(i, j), (i+1, j+1), (i+2,j+2), (i+3,j+3)]})

    # process the found lines into a score
    weights=[100, 300, 900, 8100]
    for line_of_4 in lines_of_4:
        line = [grid[i][j] for i, j in line_of_4["indices"]]
        opponent_pieces = line.count(opponent)
        my_pieces = line.count(piece)
        this_line_score = 0

        if my_pieces == 1:
            this_line_score += (weights[0] * (1 / math.pow((opponent_pieces + 1), 1)))
        if my_pieces == 2:
            this_line_score += (weights[1] * (1 / math.pow((opponent_pieces + 1), 2)))
        if my_pieces == 3:
            this_line_score += (weights[2] * (1 / math.pow((opponent_pieces + 1), 6)))
        if my_pieces == 4:
            this_line_score += weights[3]

        if opponent_pieces == 1:
            this_line_score -= (weights[0] * (1 / math.pow((my_pieces + 1), 1)))
        if opponent_pieces == 2:
            this_line_score -= (weights[1] * (1 / math.pow((my_pieces + 1), 2)))
        if opponent_pieces == 3:
            this_line_score -= (weights[2] * (1 / math.pow((my_pieces + 1), 6)))
        if opponent_pieces == 4:
            this_line_score -= weights[3]
               
        score += this_line_score

    return score


def minimax(board, current_depth, max_depth, player, alpha=-math.inf, beta=math.inf):
    
    is_terminal = board.is_game_over()
    if current_depth == max_depth or is_terminal:
        if is_terminal:
            opponent = 3 - board.get_player_turn()
            if board.get_victorious_player() == board.get_player_turn():
                return (None, +100000000)
            elif board.get_victorious_player() == opponent:
                return (None, -100000000)
            else: # Game is over, no more valid moves
                return (None, 0)
        else: # Depth is max_depth
            return (None, static_evaluator(board, board.get_player_turn()))
         
    valid_moves = board.valid_moves()
    best_move = None
    best_val = -math.inf # since it always maximises in negamax
    for move in valid_moves:
        # alpha / beta values can be negated and swapped to make them appropriate for the next player's turn
        new_move, new_val = minimax(board.play(move), current_depth+1, max_depth, player, -beta, -max(alpha, best_val))
        new_val = -new_val # negating the value makes it appropriate for the current player
        if new_val > best_val:
            best_val = new_val
            best_move = move
        if best_val >= beta: # no need to compute any more scores (prune the tree)
            break
    return best_move, best_val


# basic miniimax

# def minimax(board, current_depth, max_depth, player):
#     # This function needs to return a tuple (best_move, value), where value is the value of the board according to player=player
#     # See Millington and Funge, "Game Artificial Intelligence" texbook, 3rd edition, chapter 9.2 for pseudocode
#     # On Entry: 
#     #    board = Board object.
#     #    current_depth = level of the game tree we are currently at.
#     #    max_depth = the maximum search depth minimax is to be use, before resorting to the static_evaluator function
#     #    player = the player number (1 or 2) that the computer is playing as.
#     # On Exit, returns a tuple (best_move, value)
#     #    best_move = the move that minimax thinks is the best for the current player at the top of the tree
#     #    value = the board "value" that the game will reach if every player plays optimally from here on.
#     # print(current_depth, player, board.get_player_turn(), static_evaluator(board, player))

#     if player == board.get_player_turn():
#         maximiser=True # This means we are at the "maximiser" level of the game tree
#     else:
#         maximiser=False  # This means we are at the "minimiser" level of the game tree
    
#     # deal with easy case (the end-point of the recursion)...
#     is_terminal = board.is_game_over()
#     if current_depth == max_depth or is_terminal:
#         if is_terminal:
#             opponent=3-player
#             if board.get_victorious_player()==player:
#                 return (None, +100000000)
#             elif board.get_victorious_player()==opponent:
#                 return (None, -100000000)
#             else: # Game is over, no more valid moves
#                 return (None, 0)
#         else: # Depth is max_depth
#             return (None, static_evaluator(board, player))

#     # Use recursion to move down through the minimax levels and calculate the best_move and board value....            
#     valid_moves = board.valid_moves()
#     if maximiser:
#         best_move = None
#         best_val = -math.inf
#         for move in valid_moves:
#             new_move, new_val = minimax(board.play(move), current_depth+1, max_depth, player)
#             if new_val > best_val:
#               best_val = new_val
#               best_move = move
#         return best_move, best_val
    
#     else: # Minimising player
#         best_move = None
#         best_val = math.inf
#         for move in valid_moves:
#             new_move, new_val = minimax(board.play(move), current_depth+1, max_depth, player)
#             if new_val < best_val:
#               best_val = new_val
#               best_move = move
#         return best_move, best_val

# negamax without AB pruning

# def negamax(board, current_depth, max_depth, player):

#     is_terminal = board.is_game_over()
#     if current_depth == max_depth or is_terminal:
#         if is_terminal:
#             opponent = 3 - board.get_player_turn()
#             if board.get_victorious_player() == board.get_player_turn():
#                 return (None, +100000000)
#             elif board.get_victorious_player() == opponent:
#                 return (None, -100000000)
#             else: # Game is over, no more valid moves
#                 return (None, 0)
#         else: # Depth is max_depth
#             return (None, static_evaluator(board, board.get_player_turn()))
     
#     valid_moves = board.valid_moves()
#     best_move = None
#     best_val = -math.inf
#     for move in valid_moves:
#         new_move, new_val = negamax(board.play(move), current_depth+1, max_depth, player)
#         new_val = -new_val
#         if new_val > best_val:
#             best_val = new_val
#             best_move = move
#     return best_move, best_val
