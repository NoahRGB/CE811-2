# ce811Agents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
#
# Adapted for CE811 by M. Fairbank
# Put your (non-tutorial) agents for CE811 pacman in here.

import random, util

import numpy as np

from util import manhattanDistance
from game import Agent, Directions

from dijkstra import calc_path_to_point, calculate_gscores

BY_LENGTH = lambda el: len(el)

class ce811GoWestAgent(Agent):
  def getAction(self, gameState):
    return Directions.WEST

class ce811GoSouthWestAgent(Agent):
  def getAction(self, gameState):
    return Directions.SOUTH if Directions.SOUTH in gameState.getLegalActions() else Directions.WEST

class ce811ManhattanFoodSeekerAgent(Agent):
  def getAction(self, gameState):
    food_pos = gameState.getFood().asList()[0]
    pacman_pos = gameState.getPacmanPosition()
    diff = (pacman_pos[0] - food_pos[0], pacman_pos[1] - food_pos[1])
    if diff[0] < 0:
      return Directions.EAST
    if diff[0] > 0:
      return Directions.WEST
    if diff[1] < 0:
      return Directions.SOUTH
    if diff[1] > 0:
      return Directions.NORTH
    return Directions.STOP
  
class ce811ManhattanGhostDodgerAgent(Agent):
  
  def getAction(self, gameState):
    legal_moves = gameState.getLegalActions()
    pacman_pos = gameState.getPacmanPosition()
    ghost_state = gameState.getGhostStates()[0]
    ghost_pos = ghost_state.getPosition()
    ghost_dir = ghost_state.getDirection()
    foods = gameState.getFood().asList()
    legal_moves.remove(Directions.STOP)
    ghost_diff = (ghost_pos[0] - pacman_pos[0], ghost_pos[1] - pacman_pos[1])

    # filter directions out of good_moves if they are going to lead to pacman hitting a ghost
    if Directions.EAST in legal_moves and len(legal_moves) >= 2 and (ghost_dir == Directions.WEST and ghost_diff[1] == 0):
      legal_moves.remove(Directions.EAST)
    if Directions.WEST in legal_moves and len(legal_moves) >= 2 and (ghost_dir == Directions.EAST and ghost_diff[1] == 0):
      legal_moves.remove(Directions.WEST)
    if Directions.NORTH in legal_moves and len(legal_moves) >= 2 and (ghost_dir == Directions.NORTH or ghost_diff[1] < 0):
      legal_moves.remove(Directions.NORTH)
    if Directions.SOUTH in legal_moves and len(legal_moves) >= 2 and (ghost_dir == Directions.SOUTH or ghost_diff[1] > 0):
      legal_moves.remove(Directions.SOUTH)

    # chase a food
    for food_pos in foods:
      diff = (food_pos[0] - pacman_pos[0], food_pos[1] - pacman_pos[1])
      if diff[0] > 0 and Directions.EAST in legal_moves:
        return Directions.EAST
      if diff[0] < 0 and Directions.WEST in legal_moves:
        return Directions.WEST
      if diff[1] < 0 and Directions.SOUTH in legal_moves:
        return Directions.SOUTH
      if diff[1] > 0 and Directions.NORTH in legal_moves:
        return Directions.NORTH
    
    # or return a random safe move
    return random.choice(legal_moves)
  
class ce811ManhattanGhostDodgerHunterAgent(Agent):
  def getAction(self, gameState):
    legal_moves = gameState.getLegalActions()
    good_moves = [move for move in legal_moves if move != Directions.STOP]
    pacman_pos = gameState.getPacmanPosition()

    ghost_states = gameState.getGhostStates()
    ghost_scared_times = [ghostState.scaredTimer for ghostState in ghost_states]
    ghost_dir = [ghostState.getDirection() for ghostState in ghost_states][0]
    ghost_positions = gameState.getGhostPositions()
    ghost_distances = [manhattanDistance(pacman_pos, ghost_pos) for ghost_pos in ghost_positions]
    ghost_distances_scared = [ghost_dist for ghost_dist,time in zip(ghost_distances,ghost_scared_times) if time > 1]
    ghost_distances_dangerous = [ghost_dist for ghost_dist,time in zip(ghost_distances,ghost_scared_times) if time <= 1]
    ghost_diff = (pacman_pos[0] - ghost_positions[0][0], pacman_pos[1] - ghost_positions[0][1])
    
    # locate closest food dot:
    food_locations = gameState.getFood().asList()
    food_distances = [manhattanDistance(pacman_pos, food_pos) for food_pos in food_locations]
    closest_food_location=food_locations[food_distances.index(min(food_distances))]
    closest_food_diff = (pacman_pos[0] - closest_food_location[0], pacman_pos[1] - closest_food_location[1])

    # find the best direction to go in order to eat the closest food
    optimal_dir = None
    if closest_food_diff[0] < 0:
      optimal_dir = Directions.EAST
    if closest_food_diff[0] > 0:
      optimal_dir = Directions.WEST
    if closest_food_diff[1] < 0:
      optimal_dir = Directions.SOUTH
    if closest_food_diff[1] > 0:
      optimal_dir = Directions.NORTH

    # filter directions out of good_moves if they are going to lead to pacman hitting a ghost
    if ghost_diff[1] == 0: # ghost and pacman are on the same row
      if Directions.EAST in good_moves and ghost_dir == Directions.WEST:
        good_moves.remove(Directions.EAST)
      if Directions.WEST in good_moves and ghost_dir == Directions.EAST:
        good_moves.remove(Directions.WEST)
    else: # ghost and pacman are not on the same row
      if Directions.EAST in good_moves and ghost_dir == Directions.EAST:
        good_moves.remove(Directions.EAST)
      if Directions.WEST in good_moves and ghost_dir == Directions.WEST:
        good_moves.remove(Directions.WEST)
      if Directions.NORTH in good_moves and (ghost_diff[0] < 0 and ghost_diff[1] > 0 and ghost_dir == Directions.WEST):
        good_moves.remove(Directions.NORTH)
      if Directions.NORTH in good_moves and (ghost_diff[0] > 0 and ghost_diff[1] > 0 and ghost_dir == Directions.EAST):
        good_moves.remove(Directions.NORTH)
      if Directions.SOUTH in good_moves and (ghost_diff[0] < 0 and ghost_diff[1] < 0 and ghost_dir == Directions.WEST):
        good_moves.remove(Directions.SOUTH)
      if Directions.SOUTH in good_moves and (ghost_diff[0] > 0 and ghost_diff[1] < 0 and ghost_dir == Directions.EAST):
        good_moves.remove(Directions.SOUTH)

    # if the optimal move is left, then take it
    if optimal_dir in good_moves:
      return optimal_dir
    # otherwise choose a random safe move
    return random.choice(good_moves)
    
class ce811OneStepLookaheadManhattanAgent(Agent):

  """
    A one-step lookahead agent which chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """

  def getAction(self, gameState):
    """  
    Just like in the tutorial, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()
    legalMoves.remove(Directions.STOP)
    # Choose one of the best actions
    scores = []
    scores = [self.evaluateBoardState(gameState.generatePacmanSuccessor(action)) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalMoves[chosenIndex]

  def evaluateBoardState(self, gameState):
    """
    Design a better evaluation function here.

    The code below extracts some useful information from the state, like the
    remaining food (food_positions) and Pacman position after moving (pacman_pos).
    ghost_scared_times holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    pacman_pos = gameState.getPacmanPosition()
    food_positions = gameState.getFood().asList()
    ghost_states = gameState.getGhostStates()
    ghost_positions = gameState.getGhostPositions()
    ghost_movement_directions = [ghostState.getDirection() for ghostState in ghost_states]
    ghost_scared_times = [ghostState.scaredTimer for ghostState in ghost_states]
    capsule_positions = gameState.getCapsules()

    capsule_distances = [manhattanDistance(pacman_pos, capsule_pos) for capsule_pos in capsule_positions]

    food_distances = [manhattanDistance(pacman_pos, food_pos) for food_pos in food_positions]
    min_food_distance = min(food_distances) if len(food_distances) > 0 else 0

    ghost_distances_dangerous = [manhattanDistance(pacman_pos, ghost_pos) for ghost_pos,time in zip(ghost_positions,ghost_scared_times) if time <= 1]
    min_dangerous_ghost_distance = min(ghost_distances_dangerous) if len(ghost_distances_dangerous) > 0 else 0

    ghost_distances_scared = [manhattanDistance(pacman_pos, ghost_pos) for ghost_pos,time in zip(ghost_positions,ghost_scared_times) if time > 1]
    min_scared_ghost_distance = min(ghost_distances_scared) if len(ghost_distances_scared) > 0 else 0

    # ====== calculate evaluation of gameState based on the above variables ======

    evaluators = []

    food_weight = 20.0
    food_parameter = -len(food_positions) # (1 / min_food_distance) if min_food_distance > 0 else 0
    evaluators.append([food_parameter, food_weight])

    capsule_weight = 30.0
    capsule_parameter = -len(capsule_positions)
    evaluators.append([capsule_parameter, capsule_weight])

    score_weight = 5.0
    score_parameter = gameState.getScore()
    evaluators.append([score_parameter, score_weight])

    dangerous_ghost_distance_weight = 10.0
    dangerous_ghost_distance_parameter = min_dangerous_ghost_distance
    evaluators.append([dangerous_ghost_distance_parameter, dangerous_ghost_distance_weight])

    scared_ghost_distance_weight = 70.0
    scared_ghost_distance_parameter = (1 / min_scared_ghost_distance) if min_scared_ghost_distance > 0 else 0 # -min_scared_ghost_distance
    evaluators.append([scared_ghost_distance_parameter, scared_ghost_distance_weight])

    evaluation = 0
    for evaluator in evaluators:
      evaluation += evaluator[0] * evaluator[1]

    return evaluation

class ce811OneStepLookaheadDijkstraAgent(Agent):
  """
    A one-step lookahead agent which chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """

  def getAction(self, gameState):
    """  
    Just like in the tutorial, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()
    legalMoves.remove(Directions.STOP)

    # Choose one of the best actions
    scores = [self.evaluateBoardState(gameState.generatePacmanSuccessor(action)) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalMoves[chosenIndex]

  def evaluateBoardState(self, gameState):
    """
    Design a better evaluation function here.

    Use your old evaluation function for the OneStepLookaheadAgent, but replace the calls to the manhattan
    distance function by a gScores array lookup.
    
    Note that ghost positions can be fractional, so round them to integers before doing the array lookup to 
    avoid getting an error.
    """
    # Useful information you can extract from a GameState (pacman.py)
    pacman_pos = gameState.getPacmanPosition()
    g_scores, parent_nodes = calculate_gscores(gameState.getWalls(), pacman_pos)
    food_positions = gameState.getFood().asList()
    ghost_states = gameState.getGhostStates()
    ghost_positions = gameState.getGhostPositions()
    ghost_movement_directions = [ghostState.getDirection() for ghostState in ghost_states]
    ghost_scared_times = [ghostState.scaredTimer for ghostState in ghost_states]
    capsule_positions = gameState.getCapsules()

    capsule_distances = [g_scores[capsule_pos[1]][capsule_pos[0]] for capsule_pos in capsule_positions]
    min_capsule_distance = min(capsule_distances) if len(capsule_distances) > 0 else 0

    food_distances = [g_scores[food_pos[1]][food_pos[0]] for food_pos in food_positions]
    min_food_distance = min(food_distances) if len(food_distances) > 0 else 0

    ghost_distances_dangerous = [g_scores[int(ghost_pos[1])][int(ghost_pos[0])] for ghost_pos, time in zip(ghost_positions,ghost_scared_times) if time <= 1]
    min_dangerous_ghost_distance = min(ghost_distances_dangerous) if len(ghost_distances_dangerous) > 0 else 0

    ghost_distances_scared = [g_scores[int(ghost_pos[1])][int(ghost_pos[0])] for ghost_pos, time in zip(ghost_positions,ghost_scared_times) if time > 1]
    min_scared_ghost_distance = min(ghost_distances_scared) if len(ghost_distances_scared) > 0 else 0

    # ====== calculate evaluation of gameState based on the above variables ======

    evaluators = []

    food_dist_weight = 406.0
    food_dist_parameter = (1 / min_food_distance) if min_food_distance > 0 else 0 # -len(food_positions)
    evaluators.append([food_dist_parameter, food_dist_weight, "Food dist"])

    food_count_weight = 200.0
    food_count_parameter = -len(food_positions)
    evaluators.append([food_count_parameter, food_count_weight, "Food count"])

    capsule_dist_weight = 200.0
    capsule_dist_parameter = (1 / min_capsule_distance) if min_capsule_distance > 0 else 0
    evaluators.append([capsule_dist_parameter, capsule_dist_weight, "Capsule dist"])

    capsule_count_weight = 130.0
    capsule_count_parameter = -len(capsule_positions)
    evaluators.append([capsule_count_parameter, capsule_count_weight, "Capsule count"])

    score_weight = 1.0
    score_parameter = gameState.getScore()
    evaluators.append([score_parameter, score_weight, "Score"])

    dangerous_ghost_distance_weight = 0.9
    dangerous_ghost_distance_parameter = pow(min_dangerous_ghost_distance, 2)
    evaluators.append([dangerous_ghost_distance_parameter, dangerous_ghost_distance_weight, "Dangerous ghosts"])

    scared_ghost_distance_weight = 280.0
    scared_ghost_distance_parameter = (1 / min_scared_ghost_distance) if min_scared_ghost_distance > 0 else 0
    evaluators.append([scared_ghost_distance_parameter, scared_ghost_distance_weight, "Scared ghosts"])

    evaluation = 0
    for evaluator in evaluators:
      evaluation += evaluator[0] * evaluator[1]

    return evaluation
  
class ce811DijkstraRuleAgent(Agent):

  def getAction(self, gameState):
    legal_moves = gameState.getLegalActions()
    good_moves = [move for move in legal_moves if move != Directions.STOP]
    g_scores, parent_nodes = calculate_gscores(gameState.getWalls(), gameState.getPacmanPosition())

    ghost_scared_times = [ghostState.scaredTimer for ghostState in gameState.getGhostStates()]
    ghost_positions = gameState.getGhostPositions()
    ghost_distances = [g_scores[int(ghost_pos[1])][int(ghost_pos[0])] for ghost_pos in ghost_positions]

    ghost_distances_dangerous = [ghost_dist for ghost_dist, time in zip(ghost_distances, ghost_scared_times) if time <= 1]
    min_dangerous_ghost_distance = min(ghost_distances_dangerous) if len(ghost_distances_dangerous) > 0 else 0
    closest_dangerous_ghost_position = [pos for pos, dist in zip(ghost_positions, ghost_distances) if dist == min_dangerous_ghost_distance][0] if len(ghost_distances_dangerous) > 0 else (0, 0)
    path_to_closest_dangerous_ghost = calc_path_to_point(closest_dangerous_ghost_position, parent_nodes) if len(ghost_distances_dangerous) > 0 else []

    ghost_distances_scared = [ghost_dist for ghost_dist, time in zip(ghost_distances, ghost_scared_times) if time > 1]
    min_scared_ghost_distance = min(ghost_distances_scared) if len(ghost_distances_scared) > 0 else 0
    closest_scared_ghost_position = [pos for pos, dist in zip(ghost_positions, ghost_distances) if dist == min_scared_ghost_distance][0] if len(ghost_distances_scared) > 0 else (0, 0)
    path_to_closest_scared_ghost = calc_path_to_point(closest_scared_ghost_position, parent_nodes) if len(ghost_distances_scared) > 0 else []
#
    food_locations = gameState.getFood().asList()
    food_distances = [g_scores[food_pos[1]][food_pos[0]] for food_pos in food_locations]
    closest_food_position = food_locations[food_distances.index(min(food_distances))]
    path_to_closest_food = calc_path_to_point(closest_food_position, parent_nodes) if len(food_locations) > 0 else []

    optimal_dir = None

    optimal_dir = path_to_closest_food[0]
    if len(ghost_distances_scared) > 0:
       optimal_dir = path_to_closest_scared_ghost[0]
      
    if len(ghost_distances_dangerous) > 0:
      if len(path_to_closest_dangerous_ghost) < 7:
        good_moves.remove(path_to_closest_dangerous_ghost[0])

    return optimal_dir if optimal_dir in good_moves else random.choice(good_moves) if len(good_moves) > 0 else random.choice(gameState.getLegalActions())
  
class ce811MyBestAgent(Agent): 
  # python pacman.py -p ce811MyBestAgent -f -q -n 10
  # python pacman.py -p ce811MyBestAgent 

  def getAction(self, gameState):
    legal_moves = gameState.getLegalActions()
    safe_moves = [move for move in legal_moves if move != Directions.STOP]
    g_scores, parent_nodes = calculate_gscores(gameState.getWalls(), gameState.getPacmanPosition())

    ghost_scared_times = [ghostState.scaredTimer for ghostState in gameState.getGhostStates()]
    ghost_positions = gameState.getGhostPositions()
    paths = {"dangerous_ghosts" : [ghost_pos for ghost_pos, time in zip(ghost_positions, ghost_scared_times) if time <= 1],
             "scared_ghosts" : [ghost_pos for ghost_pos, time in zip(ghost_positions, ghost_scared_times) if time > 1],
             "food" : [food_pos for food_pos in gameState.getFood().asList()],
             "capsules" : [capsule_pos for capsule_pos in gameState.getCapsules()]}
    
    for name, positions in paths.items():
      if len(positions) == 0:
         paths[name] = [[]]
      else: 
        new_paths = [calc_path_to_point(pos, parent_nodes) for pos in positions]
        new_paths.sort(key = BY_LENGTH)
        paths[name] = new_paths


              
    optimal_dir = None

    # optimal priority is: scared ghosts --> capsules --> food
    optimal_dir = paths["food"][0][0]
    if len(paths["scared_ghosts"][0]) != 0:
       optimal_dir = paths["scared_ghosts"][0][0]
    elif len(paths["capsules"][0]) != 0:
       optimal_dir = paths["capsules"][0][0]
    
    # remove the direction that takes pacman to the closest ghost
    if len(paths["dangerous_ghosts"][0]) != 0:
      if len(paths["dangerous_ghosts"][0]) < 7:
        safe_moves.remove(paths["dangerous_ghosts"][0][0])

    # priority is: optimal move --> safe move --> legal move
    return optimal_dir if optimal_dir in safe_moves else random.choice(safe_moves) if len(safe_moves) > 0 else random.choice(gameState.getLegalActions())