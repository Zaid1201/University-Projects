import numpy as np

# PLEASE DO NOT MODIFY THE DEFINITION OF THIS CLASS!
class GridWorld():
  def __init__(self,
               p_success=1.,
               lst_mean_std=[(500, 50), (450, 10)],
               goal_loc_lst=[(2,0), (6,9)]):
    """
    GridWorld initialisation.
    gamma: Discount factor
    p_success: probability of successfully executing an action
    lst_mean_std: List  of tuples (mean, and std) for stochastic rewards. The pair is respective to the goals location.
                  If this is empty or None, the rewards are deterministic
    goal_loc_lst: List of locations of the goal
    """
    self._prob_success = p_success

    # List of mean and stds
    self._lst_mean_std = lst_mean_std

    # List of goals
    self._goal_loc = goal_loc_lst

    # Build the GridWorld
    self._initialize_gridworld()


  # Functions used to build the GridWorld environment
  # You DO NOT NEED to modify them
  def _initialize_gridworld(self):
    """
    GridWorld initialisation.
    """

    # Properties of the GridWorld
    self._shape = (13, 10)

    # Location of Lava
    self._lava = [(0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0, 0),
                        (3,0), (4,0), (5,0), (6,0), (7,0),
                  (1,3), (1,4), (1,7), (1,8), (1,9), \
                  (8,0), \
                  (9,0)
                  ]

    # Location of wall
    self._walls = [(8, i) for i in range(4, 10)]

    self._terminal_locs = [i for i in self._goal_loc]
    self._terminal_locs.extend(self._lava)

    # Reward for goals
    self._terminal_rewards = [500] * len(self._goal_loc)
    # Reward for lava
    self._terminal_rewards.extend([-10] * len(self._lava))

    self._starting_loc = (10, 0)
    self._default_reward = 0
    self._max_t = 500

    # Actions
    self._action_size = 4
    self._direction_names = ['N','E','S','W']

    # States
    self._locations = []
    for i in range (self._shape[0]):
      for j in range (self._shape[1]):
        loc = (i,j)
        # Adding the state to locations if it is no obstacle
        if self._is_location(loc):
          self._locations.append(loc)
    self._state_size = len(self._locations)

    # All available starting locations
    self._avail_starting_loc = list(set(self._locations).difference(self._terminal_locs))

    # Neighbours - each line is a state, ranked by state-number, each column is a direction (N, E, S, W)
    self._neighbours = np.zeros((self._state_size, 4))
    for state in range(self._state_size):
      loc = self.get_loc_from_state(state)

      neighbour = (loc[0]-1, loc[1])
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('N')] = self._get_state_from_loc(neighbour)
      else:
        self._neighbours[state][self._direction_names.index('N')] = state

      neighbour = (loc[0], loc[1]+1)
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('E')] = self._get_state_from_loc(neighbour)
      else:
        self._neighbours[state][self._direction_names.index('E')] = state

      neighbour = (loc[0]+1, loc[1])
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('S')] = self._get_state_from_loc(neighbour)
      else:
        self._neighbours[state][self._direction_names.index('S')] = state

      neighbour = (loc[0], loc[1]-1)
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('W')] = self._get_state_from_loc(neighbour)
      else:
        self._neighbours[state][self._direction_names.index('W')] = state

    # Absorbing
    self._absorbing = np.zeros((1, self._state_size), dtype=bool)
    for a in self._terminal_locs:
      absorbing_state = self._get_state_from_loc(a)
      self._absorbing[0, absorbing_state] = True

    # Transition matrix
    self._T = np.zeros((self._state_size, self._state_size, self._action_size))
    for action in range(self._action_size):
      for outcome in range(4):

        if action == outcome:
          prob = 1 - 3.0 * ((1.0 - self._prob_success) / 3.0)

        else:
          prob = (1.0 - self._prob_success) / 3.0

        # Write this probability in the transition matrix
        for prior_state in range(self._state_size):
          if not self._absorbing[0, prior_state]:
            post_state = self._neighbours[prior_state, outcome]
            post_state = int(post_state)
            self._T[prior_state, post_state, action] += prob

    # Reward matrix
    self._R = np.ones((self._state_size, self._state_size, self._action_size))
    self._R = self._default_reward * self._R
    for i in range(len(self._terminal_rewards)):
      post_state = self._get_state_from_loc(self._terminal_locs[i])
      self._R[:,post_state,:] = self._terminal_rewards[i]

    # Reset the environment
    self.reset()
    
    
  def get_lava_loc(self):
    """
    Get the location of lava. USED ONLY FOR PLOTTING!
    """
    return self._lava
  
  def get_walls_loc(self):
    """
    Get the location of walls. USED ONLY FOR PLOTTING!
    """
    return self._walls

  def get_gridshape(self):
    """
    Get the shape of the gridworld. USED ONLY FOR PLOTTING!
    """
    return self._shape
  
  def get_starting_loc(self):
    """
    Get starting location. USED ONLY FOR PLOTTING!
    """
    return self._starting_loc
  
  def get_goal_loc(self):
    """
    Get the location(s) of goal(s). USED ONLY FOR PLOTTING!
    """
    return self._goal_loc

  def _is_location(self, loc):
    """
    Is the location a valid state (not out of GridWorld and not a wall)
    input: loc {tuple} -- location of the state
    output: _ {bool} -- is the location a valid state
    """
    if (loc[0] < 0 or loc[1] < 0 or loc[0] > self._shape[0]-1 or loc[1] > self._shape[1]-1):
      return False
    elif loc in self._walls:
      return False
    else:
      return True


  def _get_state_from_loc(self, loc):
    """
    Get the state number corresponding to a given location
    input: loc {tuple} -- location of the state
    output: index {int} -- corresponding state number
    """
    return self._locations.index(tuple(loc))


  def get_loc_from_state(self, state):
    """
    Get the state number corresponding to a given location
    input: index {int} -- state number
    output: loc {tuple} -- corresponding location
    """
    return self._locations[state]

  def is_terminal(self, state):
    """
    Returns if a state is terminal
    input: state to check
    """
    return self._absorbing[0, state]

  # You DO NOT NEED to modify them
  def get_graphics(self):
    return self._graphics

  def get_action_size(self):
    return self._action_size

  def get_state_size(self):
    return self._state_size

  # Functions used to perform episodes in the GridWorld environment
  def reset(self):
    """
    Reset the environment state to starting state
    input: /
    output:
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """
    self._t = 0
    self._state = self._get_state_from_loc(self._starting_loc)
    self._reward = 0
    self._done = False
    if self._lst_mean_std is not None and len(self._lst_mean_std) != 0:
      self.stochastic_reward()
    return self._t, self._state, self._reward, self._done

  def stochastic_reward(self):
    """
    Employ stochasticity in the rewards of the goals
    """
    for goal_loc, reward in zip(self._goal_loc, map(lambda x: np.random.normal(*x), self._lst_mean_std)):
      goal_state = self._get_state_from_loc(goal_loc)
      self._R[:, goal_state] = reward

  def step(self, action):
    """
    Perform an action in the environment
    input: action {int} -- action to perform
    output:
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """

    # If environment already finished, print an error
    if self._done or self._absorbing[0, self._state]:
      print("Please reset the environment")
      return self._t, self._state, self._reward, self._done

    # Drawing a random number used for probaility of next state
    probability_success = np.random.uniform(0,1)

    # Look for the first possible next states (so get a reachable state even if probability_success = 0)
    new_state = 0
    while self._T[self._state, new_state, action] == 0:
      new_state += 1
    assert self._T[self._state, new_state, action] != 0, "Selected initial state should be probability 0, something might be wrong in the environment."

    # Find the first state for which probability of occurence matches the random value
    total_probability = self._T[self._state, new_state, action]
    while (total_probability < probability_success) and (new_state < self._state_size-1):
     new_state += 1
     total_probability += self._T[self._state, new_state, action]
    assert self._T[self._state, new_state, action] != 0, "Selected state should be probability 0, something might be wrong in the environment."

    # Setting new t, state, reward and done
    self._t += 1
    self._reward = self._R[self._state, new_state, action]
    self._done = self._absorbing[0, new_state] or self._t > self._max_t
    self._state = new_state
    return self._t, self._state, self._reward, self._done
  
  
def get_task1_gridworld():
  """
  Return an instance of the environment to be used for task 1
  """
  return GridWorld(lst_mean_std=None, goal_loc_lst=[(2, 0)], p_success=1)

def get_task2_gridworld():
  """
  Return an instance of the environment to be used for task 2
  """
  return GridWorld(lst_mean_std=[(700, 100), (300, 70)], p_success=1)

def get_task3_gridworld():
  """
  Return an instance of the environment to be used for task 3
  """
  return GridWorld(lst_mean_std=[(700, 100), (300, 70)], p_success=0.8)
  
  