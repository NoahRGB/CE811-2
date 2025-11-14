# ce811TutorialAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
#
# Adapted for CE811 by M. Fairbank
# Put your agents for the CE811 pacman tutorial in here.

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


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
    