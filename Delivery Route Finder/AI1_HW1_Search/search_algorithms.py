
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 380/480
#
#  MODIFIED BY: Jonesh Shrestha
# ===============================================
"""

import time
import heapq

from .search_result import SearchResults
from .search_tree_node import SearchTreeNode

from AI1_HW_Problem.problem import Problem
from AI1_HW_Problem.search_request import SearchRequest

class SearchAlgorithms:
    BreadthFirstSearch = 0
    UniformCostSearch = 1
    AStarSearch = 2

    """
        Implementation of the Breadth-first Search (BFS) algorithm. The only input
        required is the problem, which includes a reference to current CityMap and 
        current Search Request. The function needs to compute and store search 
        results and other statistics inside of a SearchResults object. 
    """
    @staticmethod
    def breadth_first_search(problem: Problem) -> SearchResults:
        # create root node using the initial state
        initial_state = problem.get_initial_state()
        root_node = SearchTreeNode(None, None, initial_state, 0)

        # if initial state is goal state return
        if problem.is_goal_state(initial_state):
            return SearchResults([], 0.0, 1, 0)
        
        # create a FIFO queue for frontier and assign root_node initially
        frontier = [root_node]

        # track visited states with reached
        reached = {initial_state.get_representation()}
        explored = 0

        # while frontier is not empty loop
        while frontier:
            # FIFO implementation: get first node from queue
            current_node = frontier.pop(0)
            current_state = current_node.get_state()

            explored += 1

            # generate all possible child states
            child_states = problem.generate_children(current_state)

            for child_state in child_states:
                action = child_state.get_location()

                state_representation = child_state.get_representation()

                # calculate new path cost
                action_cost = problem.get_action_cost(current_state, action)
                path_cost = current_node.get_path_cost() + action_cost

                # create a search tree node
                child_node = SearchTreeNode(current_node, action, child_state, path_cost)

                # check if we have the goal state
                if problem.is_goal_state(child_state):
                    solution = child_node.path_to_root()
                    return SearchResults(solution, path_cost, len(reached), explored)
                
                # add unvisited states to frontier queue
                if state_representation not in reached:
                    reached.add(state_representation)

                    frontier.append(child_node)

        # no solution
        return SearchResults(None, None, len(reached), explored)

    """
        Implementation of the Uniform Cost Search (UCS) algorithm. The only input
        required is the problem, which includes a reference to current CityMap and 
        current Search Request. The function needs to compute and store search 
        results and other statistics inside of a SearchResults object. 
    """
    @staticmethod
    def uniform_cost_search(problem: Problem) -> SearchResults:
         # create root node using the initial state
        initial_state = problem.get_initial_state()
        root_node = SearchTreeNode(None, None, initial_state, 0)
        
        # create a priority queue for frontier ordered by path cost
        frontier = [(0.0, root_node)]
        heapq.heapify(frontier)

        # state representation as key and cost as value to track visited states with best path cost
        reached = {initial_state.get_representation(): root_node}
        explored = 0

        while frontier:
            # get the lowest path_cost
            current_node = heapq.heappop(frontier)[1]
            current_state = current_node.get_state()

            # check for goal state
            if problem.is_goal_state(current_state):
                solution = current_node.path_to_root()
                return SearchResults(solution, current_node.get_path_cost(), len(reached), explored)
            
            explored += 1
            child_states = problem.generate_children(current_state)

            for child_state in child_states:
                action = child_state.get_location()

                state_representation = child_state.get_representation()

                # calculate new path cost
                action_cost = problem.get_action_cost(current_state, action)
                path_cost = current_node.get_path_cost() + action_cost

                child_node = SearchTreeNode(current_node, action, child_state, path_cost)

                # add new state or update existing one if better path found
                if state_representation not in reached or path_cost < reached[state_representation].get_path_cost():
                    reached[state_representation] = child_node
                    heapq.heappush(frontier, (path_cost, child_node))

        # no solution
        return SearchResults(None, None, len(reached), explored)

            

    """
        Implementation of the A* Search algorithm. The only input
        required is the problem, which includes a reference to current CityMap and 
        current Search Request. The function needs to compute and store search 
        results and other statistics inside of a SearchResults object. 
    """
    @staticmethod
    def A_start_search(problem: Problem) -> SearchResults:
         # create root node using the initial state
        initial_state = problem.get_initial_state()
        root_node = SearchTreeNode(None, None, initial_state, 0)

        # create initial heuristic cost
        initial_heuristic = problem.estimate_cost_to_solution(initial_state)
        
        # create a priority queue for frontier ordered by path cost and heuristic
        frontier = [(initial_heuristic, root_node)]
        heapq.heapify(frontier)

        # state representation as key and cost as value to track visited states with best path cost
        reached = {initial_state.get_representation(): root_node}
        explored = 0

        while frontier:
            # get node with lowest f value from frontier
            current_node = heapq.heappop(frontier)[1]
            current_state = current_node.get_state()

            # check for goal state
            if problem.is_goal_state(current_state):
                solution = current_node.path_to_root()
                return SearchResults(solution, current_node.get_path_cost(), len(reached), explored)
            
            explored += 1
            child_states = problem.generate_children(current_state)
            
            for child_state in child_states:
                action = child_state.get_location()

                state_representation = child_state.get_representation()

                action_cost = problem.get_action_cost(current_state, action)
                path_cost = current_node.get_path_cost() + action_cost

                child_node = SearchTreeNode(current_node, action, child_state, path_cost)

                if state_representation not in reached or path_cost < reached[state_representation].get_path_cost():
                    reached[state_representation] = child_node
                    heuristic_cost = problem.estimate_cost_to_solution(child_state)
                    f = path_cost + heuristic_cost

                    heapq.heappush(frontier, (f, child_node))

        return SearchResults(None, None, len(reached), explored)
        

    """
        Auxiliary function for printing search results 
    """
    @staticmethod
    def print_solution_details(test_case: SearchRequest, search_results: SearchResults, search_time: float):
        if search_results.get_solution() is None:
            print(" --> No Solution was found!")
        else:
            # this part is just for formatting... with highlight for important nodes in the path
            tempo_list = []
            for value in [test_case.get_start_location()] + search_results.get_solution():
                if value == test_case.get_start_location():
                    tempo_list.append(f"[{value}]")
                elif value in test_case.get_targets():
                    tempo_list.append(f"({value})")
                else:
                    tempo_list.append(value)
            full_path = " -> ".join(tempo_list)
            print(f" --> Solution found: {full_path}")
            print(f" --> Solution Cost: {search_results.get_cost()}")
        print(f" --> Nodes Reached: {search_results.reached_nodes_count()}")
        print(f" --> Nodes Explored: {search_results.explored_nodes_count()}")
        print(f" --> Search Time (s): {search_time}")

    """
        Auxiliary Function for running the search algorithm specified, 
        and printing the results and statistics.
    """
    @staticmethod
    def search(problem: Problem, algorithm: int):
        # Note: This code might look awkward, but this is intentional
        # the idea is to NOT count the print operation as part of the search time
        if algorithm == SearchAlgorithms.BreadthFirstSearch:
            print("\n - Running Breadth-First Search")
        elif algorithm == SearchAlgorithms.UniformCostSearch:
            print("\n - Running Uniform Cost Search")
        elif algorithm == SearchAlgorithms.AStarSearch:
            print("\n - Running A Star Search")
        else:
            raise Exception(f"Invalid Search Algorithm: {algorithm}")

        start_time = time.time()
        if algorithm == SearchAlgorithms.BreadthFirstSearch:
            solution = SearchAlgorithms.breadth_first_search(problem)
        elif algorithm == SearchAlgorithms.UniformCostSearch:
            solution = SearchAlgorithms.uniform_cost_search(problem)
        else:
            # by default, assume it is the A* algorithm
            solution = SearchAlgorithms.A_start_search(problem)
        end_time = time.time()
        total_time = end_time - start_time

        SearchAlgorithms.print_solution_details(problem.get_current_case(), solution, total_time)

