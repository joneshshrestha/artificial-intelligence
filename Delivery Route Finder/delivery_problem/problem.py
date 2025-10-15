from .state import State
from .map import CityMap
from .search_request import SearchRequest
from typing import List, Tuple

"""
    The Problem class. This is where most of the problem-specific logic (e.g. 
    the problem formulation) is done. For this delivery routing problem, we need
    two attributes: the city map and the search request (a delivery scenario). The 
    city map contains all the information related to the underlying navigation
    problem, while the Search Request includes all the info related to one
    specific search (e.g. starting point, target delivery locations, etc.). It is the 
    job of the Problem class to put these two together to generate the states,
    actions, transitions, costs and heuristics needed for search.       
"""


class Problem:
    """
    The constructor. A new instance of the Problem class should be used
    for every new search that will be executed.
    """

    def __init__(self, map: CityMap, search_request: SearchRequest):
        self.__city_map = map
        self.__current_case = search_request

    def get_city_map(self):
        return self.__city_map

    def get_current_case(self):
        return self.__current_case

    """
        Generate the initial state for the current search problem using
        information from the map and search request.
        
        Returns:
            State: Initial state with starting location and unvisited targets
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
        Check if the given state represents a goal state.
        
        A goal state is reached when the driver has returned to the starting 
        location after visiting all target delivery locations.
        
        Args:
            state: The state to check
            
        Returns:
            bool: True if this is a goal state, False otherwise
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
        Generate all possible child states from the given state.
        
        This implements the transition model by generating a new state for each
        neighboring location. When a neighbor is a target location, the visited
        status is updated in the child state.
        
        Args:
            state: The current state
            
        Returns:
            List[State]: List of all possible child states
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
        child_states = []

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
            child_states.append(new_child_state)

        return child_states

    """
        Calculate the cost of executing an action from a given state.
        
        Returns the actual travel distance between the current state's location
        and the destination (not the straight-line distance estimate).
        
        Args:
            state: The current state
            action: The destination location (action to take)
            
        Returns:
            float: The cost (distance in miles) to travel from current location to destination
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
        Estimate the cost to reach a goal state from the given state.
        
        This heuristic function uses straight-line distances to compute an 
        admissible lower bound for the A* search algorithm. It considers the
        cost to visit remaining unvisited targets and return to the starting location.
        
        Args:
            state: The current state
            
        Returns:
            float: Estimated cost to complete the delivery route from this state
    """

    def estimate_cost_to_solution(self, state: State) -> float:
        # get current location from state
        current_location = state.get_location()
        # get start/final location from search_request
        start_final_location = self.__current_case.get_start_location()
        # get targets from search_requests
        targets = self.__current_case.get_targets()
        # get visited targets from state
        visited_targets = state.get_visited_targets()

        # pending delivery targets list
        pending_targets = []

        # if there is any target which is not visited append to pending delivery targets list
        for i in range(len(targets)):
            if visited_targets[i] == False:
                pending_targets.append(targets[i])

        # observation 1: no deliveries left
        if len(pending_targets) == 0:
            return self.__city_map.get_straight_line_distance(
                current_location, start_final_location
            )
        # observation 2: 1 or more deliveries left
        else:
            min_dist = float("inf")
            max_dist = 0

            for target in pending_targets:
                sl_dist = self.__city_map.get_straight_line_distance(
                    target, start_final_location
                )

                if sl_dist < min_dist:
                    min_dist = sl_dist
                    l_close = target
                if sl_dist > max_dist:
                    max_dist = sl_dist
                    l_far = target

            l_close_distance = self.__city_map.get_straight_line_distance(
                current_location, l_close
            )
            l_far_distance = self.__city_map.get_straight_line_distance(
                current_location, l_far
            )

            c_beginning = min(l_close_distance, l_far_distance)
            c_middle = self.__city_map.get_straight_line_distance(l_close, l_far)
            c_final = self.__city_map.get_straight_line_distance(
                l_close, start_final_location
            )

            return c_beginning + c_middle + c_final
