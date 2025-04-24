
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 380/480
#
#  MODIFIED BY: Jonesh Shrestha
# ===============================================
"""
from .state import State
from .map import CityMap
from .search_request import SearchRequest
from typing import List, Tuple

"""
    The Problem class. This is where most of the problem-specific logic (e.g. 
    the problem formulation) is done. For this problem in particular, we need
    two attributes: the city map and the test case (a Search Request). The 
    city map contains all the information related to the underlying navigation
    problem, while the Search Request includes all the info related to one
    specific search (e.g. starting point, targets locations, etc.). It is the 
    job of the Problem class to put these two together to generate the states,
    actions, transitions, costs and heuristics needed for search.       
"""


class Problem:
    """
        The constructor. A new instance of the Problem class should be used
        for every new search that will be executed.
    """
    def __init__(self, map: CityMap, test_case: SearchRequest):
        self.__city_map = map
        self.__current_case = test_case

    def get_city_map(self):
        return self.__city_map

    def get_current_case(self):
        return self.__current_case

    """
        Use the information from map and test case to generate the 
        Initial State for the current search problem.
        
        Hint: current_case has information about both the starting location
              and the search nodes. 
    """
    def get_initial_state(self) -> State | None:
        # get the start location
        start_location = self.__current_case.get_start_location()
        # get the target locations
        targets = self.__current_case.get_targets()
        # stores list with default boolean value false, for each target
        visited_targets = len(targets) * [False]
        # return the initial state    
        return State(start_location, visited_targets)

    """
        Check if the given state object represent a goal state. This can be
        determined using information stored in the state itself and also
        in the current_case object. 
        
        Hint: Keep in mind that the goal state requires the driver to be back 
            at the same location where the trip start, and all targets should 
            have been visited.
    """
    def is_goal_state(self, state: State) -> bool:
        # get the start location
        start_location = self.__current_case.get_start_location()
        # get the current location from state
        current_location = state.get_location()
        # get all the boolean values in visited list from state
        visited_targets = state.get_visited_targets()
        # return true if current location is the start location and all targets are visited
        return current_location == start_location and all(visited_targets)

    """
        Given a state, this function generates a List of the Children states.
        This function basically implements the transition model. First, you 
        need to consider which locations are neighbors of current location 
        (the map class can help with that). Actions here are equal to travel
        to any of these immediate neighbors. For each of these actions, you 
        will have to generate a new state object. Keep in mind that some of 
        these locations might be part of the targets. As such, your state 
        generation process needs to correctly update the vector of visited
        nodes for the child state. 
        
        Hint: This function needs to use the city_map, the state, and the
            search_case. 
        Hint: Mind the alignment between target locations in the Search Request
            object and visited targets in the State object.  
    """
    def generate_children(self, state: State) -> List[State]:
        # get current location from state
        current_location = state.get_location()
        # get neighbors from map
        neighbors = self.__city_map.get_neighbors(current_location)
        # get visited targets from state (Boolean List)
        visited_targets = state.get_visited_targets()
        # get targets from search_request (List)
        targets = self.__current_case.get_targets()

        # new list of child state to return 
        child_state = []

        # for each neighbor create new state
        for neighbor in neighbors:
            # create new visited targets copy 
            child_visited_targets = visited_targets.copy()
            # if any neighbor in targets, for that neighbor(index) mark it as visited
            if neighbor in targets:
                target_index = targets.index(neighbor)
                child_visited_targets[target_index] = True
        
        # create new state with current location and visited targets
        new_child_state = State(neighbor, child_visited_targets)
        # append to the child_state list
        child_state.append(new_child_state)

        return child_state

    """
        Cost-Model: cost of executing the given action on the given state
              cost = Action-Cost( State, Action)
    
        Hint 1: You need to consider the location of current state
        Hint 2: The city map can give you the info you need about costs of
             traveling between neighboring locations.
        Hint 3: Do not confuse this with the straight line traveling distance 
             used to estimate the cost of traveling between non-neighboring 
             locations    
    """
    def get_action_cost(self, state: State, action: str) -> float:
        # get current location from state
        current_location = state.get_location()
        # destination is action
        destination = action
        # calculate get_cost from map
        get_cost = self.__city_map.get_cost(current_location, destination)
        return get_cost

    """
        Cost-Estimation-Model: estimated cost of reaching a goal state from the
        given state. This is the Heuristic function required by A*
    
            estimated-cost = Heuristic( State)
    
        Hint 1: You need to consider the location of current state
        Hint 2: The city map can help you estimate the lower-bound of the real
            cost of traveling certain distances.
        Hint 3: The write-up offers a more detailed explanation of the function
            that you should be implementing here
    
    """
    def estimate_cost_to_solution(self, state: State) -> float:
        # TODO: YOUR CODE HERE
        return 0.0
