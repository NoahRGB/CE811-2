import numpy as np

from game import Directions

def calculate_neighbouring_nodes(node, maze):
    # for any node (y,x), calculates a dictionary of which nodes are neighbours in this maze array?
    maze_height = maze.height
    maze_width = maze.width
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    (x ,y) = node
    result_neigbours = {}
    for (dx,dy) in directions:
        neighbour_y = y + dy
        neighbour_x = x + dx
        # check if this potential neighbour goes off the edges of the maze:
        if neighbour_y >= 0 and neighbour_y < maze_height and neighbour_x >= 0 and neighbour_x<maze_width:
            # check if this potential neighbour is not a wall:
            if maze[neighbour_x][neighbour_y] == 0:
                # we have found a valid neighbouring node that is not a wall
                result_neigbours[(neighbour_x,neighbour_y)] = 1 # this says the distance to this neighbour is 1
                # Note that all neighbours in this problem are distance 1 away!
    return result_neigbours

def calculate_gscores(maze, start_node):
    assert str(type(maze))=="<class 'game.Grid'>"
    maze_height = maze.height
    maze_width = maze.width
    assert maze[start_node[0]][start_node[1]]==False,"start_node error "+str(start_node) # start node must point to a False in the maze array

    array_gScores = np.zeros((maze_height, maze_width), dtype=int)
    g_scores = { start_node: 0 }
    parent_nodes = { start_node: None}
    open_nodes = [start_node]
    closed_list = []
    while len(open_nodes) > 0:
        sorted_list_of_open_nodes_nodes = sorted(open_nodes, key = lambda n: g_scores[n])
        current_node = sorted_list_of_open_nodes_nodes[0]
        current_distance = g_scores[current_node]
        closed_list.append(current_node)
        array_gScores[current_node[1]][current_node[0]] = current_distance
        open_nodes.remove(current_node)
        neighbours = calculate_neighbouring_nodes(current_node, maze)
        for neighbour_node, neighbour_distance in neighbours.items():
            new_dist = (current_distance + neighbour_distance)
            if neighbour_node in g_scores:
                if new_dist < g_scores[neighbour_node]:
                    g_scores[neighbour_node] = new_dist
                    parent_nodes[neighbour_node] = current_node
            else:
                g_scores[neighbour_node] = new_dist
                parent_nodes[neighbour_node] = current_node
                open_nodes.append(neighbour_node)
            
    return [array_gScores, parent_nodes]

def calc_path_A_to_B(start_node, end_node, maze_walls):
    # This function is just a wrapper for calc_path_to_point below
    # This function is written for you - no need to change anything in this short function.
    gScores,parent_nodes=calculate_gscores(maze_walls, start_node)
    return calc_path_to_point(end_node, parent_nodes)
    
def get_direction_from_A_to_B(A, B):
    diff = (A[0] - B[0], A[1] - B[1])
    if diff[0] < 0:
        return Directions.WEST
    if diff[0] > 0:
        return Directions.EAST
    if diff[1] < 0:
        return Directions.NORTH
    if diff[1] > 0:
        return Directions.SOUTH
    return Directions.STOP

def calc_path_to_point(end_node, parent_nodes):    
    # TODO This is the main function you need to write.
    # Work backwards along the parent_nodes dictionary accumulating directions into a list.  
    # Then reverse your list before returning it.
    path = []
    current_node = (int(end_node[0]), int(end_node[1]))
    while parent_nodes[current_node] != None:
        new_node = parent_nodes[current_node]
        path.insert(0, get_direction_from_A_to_B(current_node, new_node))
        current_node = new_node
    return path
