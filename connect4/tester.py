import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt

from connect4_gui import Agents, create_empty_board
from mcts import expand_mcts_tree_once, expand_mcts_tree_repeatedly, build_initial_blank_mcts_tree
from minimax import minimax, negamax, static_evaluator, negamax_prune
from connect4Board import Board

def play_one_game(agent_1, agent_2, opts_agent_1=None, opts_agent_2=None):
  controllers = [agent_1, agent_2]
  game_over = False
  board = create_empty_board()

  if Agents.MCTS in controllers:
    mcts_tree = build_initial_blank_mcts_tree()
  else:
    mcts_tree = None

  while not game_over:
    turn = board.get_player_turn() # This will be 1 or 2, for player 1 or player 2, respectievly.
    current_agent = controllers[turn-1]
    move_choice=None

    opts = None
    if (opts_agent_1 and turn == 1) or (opts_agent_2 and turn == 2):
      opts = opts_agent_1 if turn == 1 else opts_agent_2

    # Make decision for AI players (if it's their turn...)
    if current_agent != Agents.USER and not game_over:
        if current_agent == Agents.MINIMAX:
            if opts:
              move_choice, minimax_score = minimax(board, current_depth=0, max_depth=opts["max_depth"], player=turn) # increase the max_depth to make a stronger player
            else:
              move_choice, minimax_score = minimax(board, current_depth=0, max_depth=3, player=turn) # increase the max_depth to make a stronger player
        elif current_agent == Agents.NEGAMAX:
          if opts:
            move_choice, minimax_score = negamax(board, current_depth=0, max_depth=opts["max_depth"], player=turn) # increase the max_depth to make a stronger player
          else:
             move_choice, minimax_score = negamax(board, current_depth=0, max_depth=3, player=turn) # increase the max_depth to make a stronger player
        elif current_agent == Agents.NEGAMAX_PRUNE:
          if opts:
            # print(f"Making negamax prune move with max depth = {opts["max_depth"]}")
            move_choice, minimax_score = negamax_prune(board, current_depth=0, max_depth=opts["max_depth"], player=turn, alpha=-math.inf, beta=math.inf) # increase the max_depth to make a stronger player
          else:
            move_choice, minimax_score = negamax_prune(board, current_depth=0, max_depth=3, player=turn, alpha=-math.inf, beta=math.inf) # increase the max_depth to make a stronger player
        
        elif current_agent == Agents.RANDOM:
          move_choice = random.choice(board.valid_moves())

        elif current_agent == Agents.STATIC_EVALUATOR:
            valid_moves = board.valid_moves()
            move_scores = np.array([static_evaluator(board.play(move), turn) for move in valid_moves]) # This is the one-step look ahead
            best_score = move_scores.max()
            best_score_indices = np.where(move_scores == best_score)[0]
            move_choice = valid_moves[random.choice(best_score_indices)]

        elif current_agent == Agents.MCTS:
            assert mcts_tree.board==board
            if opts:
              # print(f"Making mcts move with expansion time = {opts["expansion_time"]}")
              expand_mcts_tree_repeatedly(mcts_tree, tree_expansion_time_ms=opts["expansion_time"])# increase the expansion time to make a stronger player
              mcts_tree, move_choice = mcts_tree.select_best_move()
            else:
              expand_mcts_tree_repeatedly(mcts_tree, tree_expansion_time_ms=100)# increase the expansion time to make a stronger player
              mcts_tree, move_choice = mcts_tree.select_best_move()
        else:
            raise Exception("Unknown agent "+str(current_agent))
        
        assert move_choice!=None
        assert board.can_play(move_choice)
        
    if move_choice != None:
        assert board.can_play(move_choice)
        board = board.play(move_choice)

        if board.is_game_over():
            game_over = True

        if mcts_tree !=None and current_agent != Agents.MCTS:
            # update the MCTS tree to say it has a new root node.
            expand_mcts_tree_once(mcts_tree)
            mcts_tree=mcts_tree.get_child_with_move(move_choice)
            assert mcts_tree.board == board

        move_choice=None 
      
    if game_over:
      return board.get_victorious_player()

def measure_winrates(agent_1, agent_2, num_games, opts_agent_1=None, opts_agent_2=None):
  wins = [0, 0] # agent_1 wins in index 0, agent_2 wins in index 1
  for i in range(num_games):
    wins[play_one_game(agent_1, agent_2, opts_agent_1, opts_agent_2) - 1] += 1
    if i+1 == num_games / 2:
      print(f"Finished game {i+1} / {num_games}")
  return (wins[0] / num_games) * 100, (wins[1] / num_games) * 100

def assess_minimax(num_games, depths, do_minimax, do_negamax, do_negamax_prune):
  minimax_average_times = []
  negamax_average_times = []
  negamax_prune_average_times = []
  minimax_average_winrates = []
  negamax_average_winrates = []
  negamax_prune_average_winrates = []

  for depth in depths:
    print(f"Depth = {depth}")

    minimax_total_time = 0
    negamax_total_time = 0
    negamax_prune_total_time = 0
    minimax_total_winrate = 0
    negamax_total_winrate = 0
    negamax_prune_total_winrate = 0

    for i in range(num_games):

      if do_minimax:
        start_1 = time.time()
        winner = play_one_game(Agents.MCTS, Agents.MINIMAX, opts_agent_1={"expansion_time": 200}, opts_agent_2={"max_depth": depth})
        end_1 = time.time()
        minimax_total_time += end_1 - start_1
        minimax_total_winrate += 1 if winner == 2 else 0
        print(f"Finished minimax depth = {depth} in {end_1 - start_1}")
    
      if do_negamax:
        start_2 = time.time()
        winner = play_one_game(Agents.MCTS, Agents.NEGAMAX, opts_agent_1={"expansion_time": 200}, opts_agent_2={"max_depth": depth})
        end_2 = time.time()
        negamax_total_time += end_2 - start_2
        negamax_total_winrate += 1 if winner == 2 else 0
        print(f"Finished negamax depth = {depth} in {end_2 - start_2}")

      if do_negamax_prune:
        start_3 = time.time()
        winwinnerrates = play_one_game(Agents.MCTS, Agents.NEGAMAX_PRUNE, opts_agent_1={"expansion_time": 200}, opts_agent_2={"max_depth": depth})
        end_3 = time.time()
        negamax_prune_total_time += end_3 - start_3#
        negamax_prune_total_winrate += 1 if winner == 2 else 0
        print(f"Finished negamax prune depth = {depth} in {end_3 - start_3}")


    minimax_average_times.append(minimax_total_time / num_games)
    negamax_average_times.append(negamax_total_time / num_games)
    negamax_prune_average_times.append(negamax_prune_total_time / num_games)
    minimax_average_winrates.append(minimax_total_winrate / num_games)
    negamax_average_winrates.append(negamax_total_winrate / num_games)
    negamax_prune_average_winrates.append(negamax_prune_total_winrate / num_games)
  
  output = []
  if do_minimax: output.append(["Minimax", minimax_average_times, minimax_average_winrates])
  if do_negamax: output.append(["Negamax", negamax_average_times, negamax_average_winrates])
  if do_negamax_prune: output.append(["Negamax + AB pruning", negamax_prune_average_times, negamax_prune_average_winrates])
  return output

########################################################################################
########################################################################################
########################################################################################

print(measure_winrates(Agents.RANDOM, Agents.STATIC_EVALUATOR, 10000))
print(measure_winrates(Agents.STATIC_EVALUATOR, Agents.RANDOM, 10000))

# print(measure_winrates(Agents.RANDOM, Agents.MINIMAX, 100))
# print(measure_winrates(Agents.MINIMAX, Agents.RANDOM, 100))

# print(measure_winrates(Agents.MCTS, Agents.NEGAMAX_PRUNE, 100))
# print(measure_winrates(Agents.NEGAMAX_PRUNE, Agents.MCTS, 100))


#=============== ensure static evaluator is 0 sum ======================

board = Board(grid=np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 2, 0, 0, 0, 0],
                            [1, 2, 2, 2, 0, 0, 0]]))
# print(static_evaluator(board, 1), static_evaluator(board, 2), static_evaluator(board, 1) - static_evaluator(board, 2))
assert abs(static_evaluator(board, 1)) - abs(static_evaluator(board, 2)) == 0



# =============== measure MCTS vs minimax ======================

# possible_expansion_times = [1, 10, 100, 200, 500]
# possible_max_depths = [1, 2, 3, 4, 5, 6]
# num_games = 100
# winrates = [[(0, 0, 0, 0) for i in range(len(possible_max_depths))] for j in range(len(possible_expansion_times))]

# for i in range(0, len(possible_expansion_times)):
#   for j in range(0, len(possible_max_depths)):
#       print(f"Testing expansion_time: {possible_expansion_times[i]} with max_depth {possible_max_depths[j]}")
#       mcts_opts = { "expansion_time": possible_expansion_times[i] }
#       minimax_opts = { "max_depth": possible_max_depths[j] }
#       config_1 = measure_winrates(Agents.MCTS, Agents.NEGAMAX_PRUNE, num_games, mcts_opts, minimax_opts)
#       print(f"Finished config 1: {config_1}")
#       config_2 = measure_winrates(Agents.NEGAMAX_PRUNE, Agents.MCTS, num_games, minimax_opts, mcts_opts)
#       print(f"Finished config 2: {config_2}")
#       winrates[i][j] = (config_1[0], config_1[1], config_2[0], config_2[1])
#   print(f"Winrates so far {winrates}")
   
# print(f"\n\n\n Final winrates {winrates}")



# ================== assess minimax times / winrates ==================

# num_games = 10
# x = [1, 2, 3, 4]
# data = assess_minimax(num_games, x, True, True, True)

# print(data)