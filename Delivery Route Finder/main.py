
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 380/480
#
#  MODIFIED BY: Jonesh Shrestha
# ===============================================
"""

import sys

from AI1_HW_Problem.map import CityMap
from AI1_HW_Problem.search_request import SearchRequest
from AI1_HW_Problem.problem import Problem
from AI1_HW1_Search.search_algorithms import SearchAlgorithms

def main():
    # ... loading the map from JSON file ...
    map = CityMap.FromFile("./tegucigalpa.json")
    # ... loading the test cases from JSON file ...
    test_cases = SearchRequest.FromTestCasesFile("./test_cases.json")

    for test_case in test_cases:
        print("\n\nTest Case info:")
        print(f" - Name: {test_case.get_name()}")
        print(f" - Starting Location: {test_case.get_start_location()}")
        print(f" - Delivery Locations: {test_case.get_targets()}")

        # Create the problem
        problem = Problem(map, test_case)

        # use BFS ....
        SearchAlgorithms.search(problem, SearchAlgorithms.BreadthFirstSearch)

        # use UCS ....
        SearchAlgorithms.search(problem, SearchAlgorithms.UniformCostSearch)

        # use A* ....
        SearchAlgorithms.search(problem, SearchAlgorithms.AStarSearch)


if __name__ == "__main__":
    main()
