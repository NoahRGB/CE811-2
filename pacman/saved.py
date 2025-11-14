

# class ce811MyBestAgent(Agent): 
#   # python pacman.py -p ce811MyBestAgent -f -q -n 10
#   # python pacman.py -p ce811MyBestAgent 

#   def getAction(self, gameState):
#     legal_moves = gameState.getLegalActions()
#     good_moves = [move for move in legal_moves if move != Directions.STOP]
#     g_scores, parent_nodes = calculate_gscores(gameState.getWalls(), gameState.getPacmanPosition())

#     ghost_scared_times = [ghostState.scaredTimer for ghostState in gameState.getGhostStates()]
#     ghost_positions = gameState.getGhostPositions()

#     dangerous_ghost_paths = [calc_path_to_point(ghost_pos, parent_nodes) for ghost_pos, time in zip(ghost_positions, ghost_scared_times) if time <= 1]
#     path_to_closest_dangerous_ghost = min(dangerous_ghost_paths, key = lambda path: len(path)) if len(dangerous_ghost_paths) > 0 else []

#     scared_ghost_paths = [calc_path_to_point(ghost_pos, parent_nodes) for ghost_pos, time in zip(ghost_positions, ghost_scared_times) if time > 1]
#     path_to_closest_scared_ghost = min(scared_ghost_paths, key = lambda path: len(path)) if len(scared_ghost_paths) > 0 else []

#     food_paths = [calc_path_to_point(food_pos, parent_nodes) for food_pos in gameState.getFood().asList()]
#     path_to_closest_food = min(food_paths, key = lambda path: len(path)) if len(food_paths) > 0 else []

#     capsule_paths = [calc_path_to_point(capsule_pos, parent_nodes) for capsule_pos in gameState.getCapsules()]
#     path_to_closest_capsule = min(capsule_paths, key = lambda path: len(path)) if len(capsule_paths) > 0 else []

#     optimal_dir = None

#     optimal_dir = path_to_closest_food[0]
#     if len(scared_ghost_paths) > 0:
#        optimal_dir = path_to_closest_scared_ghost[0]
#     elif len(capsule_paths) > 0:
#        optimal_dir = path_to_closest_capsule[0]
      
#     if len(dangerous_ghost_paths) > 0:
#       if len(path_to_closest_dangerous_ghost) < 7:
#         good_moves.remove(path_to_closest_dangerous_ghost[0])

#     return optimal_dir if optimal_dir in good_moves else random.choice(good_moves) if len(good_moves) > 0 else random.choice(gameState.getLegalActions())