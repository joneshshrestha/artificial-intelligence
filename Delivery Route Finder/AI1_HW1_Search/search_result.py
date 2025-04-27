
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 380/480
# DO NOT MODIFY AND/OR REDISTRIBUTE
# ===============================================
"""

from typing import List

"""
    The SearchResults class, used to collect and group all information relevant 
    to the results of any given search.
"""

class SearchResults:
    def __init__(self, solution_path: List[str] | None, solution_cost: float | None, nodes_reached: int, nodes_explored: int):
        self.__solution_path = solution_path
        self.__solution_cost = solution_cost
        self.__nodes_reached = nodes_reached
        self.__nodes_explored = nodes_explored

    def get_solution(self):
        return self.__solution_path

    def get_cost(self):
        return self.__solution_cost

    def explored_nodes_count(self):
        return self.__nodes_explored

    def reached_nodes_count(self):
        return self.__nodes_reached
