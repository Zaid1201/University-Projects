# DO NOT CHANGE ANY DEFINITIONS IN THIS FILE

import matplotlib.pyplot as plt
import numpy as np

def paint_gridworld(gridworld_obj, figsize=(17, 12), ticksize=15):
  """
  Paint the gridworld
  gridworld_obj: GridWorld Object
  figsize: Figure size (Default is (17, 12))
  ticksize: Size of the ticks (Default is 15)
  """
  grid = get_grid(gridworld_obj)
  
  plt.figure(figsize=figsize)
  plt.imshow(grid)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.show()

def get_grid(gridworld_obj):
  """
  Return the grid with lava location, goal locations, starting locations, and walls
  This can be directly passed in to paint_gridworld to paint the map
  """
  shape = gridworld_obj.get_gridshape()
  lava_locs = gridworld_obj.get_lava_loc()
  goal_states = gridworld_obj.get_goal_loc()
  starting_loc = gridworld_obj.get_starting_loc()
  walls = gridworld_obj.get_walls_loc()

  grid = np.ones((shape[0], shape[1], 3), dtype=int)
  grid[:, :, :] = [255, 255, 255]
  for ob in lava_locs:
  # Lava
    grid[ob] = [178,34,34]

  for goal in goal_states:
    grid[goal] = [30,144,255]

  for wall in walls:
      grid[wall] = [0,0,0]

  grid[starting_loc] = [253, 218, 13]
  return grid