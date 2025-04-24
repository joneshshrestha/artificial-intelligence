
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

"""
    This function provides a few examples of how to use the CityMap class
"""
def example_using_map_class(map: CityMap):
    # TODO: feel free to modify and/or remove this code
    # ... Checking the name of the map
    print(f"City Name: {map.get_name()}")
    # ... Checking the locations
    print(f"Locations: {map.get_locations()}")
    # ... This will give you the neighbors of any location in the map
    loc = 'F3'
    print(f"Neighbors of {loc}: {map.get_neighbors(loc)}")
    # ... This will give you the cost of an existing connection
    other = 'F4'
    print(f"Cost of traveling from {loc} to {other}: {map.get_cost(loc, other)}")
    # ... and will return null if there is no connection between the nodes ...
    other = 'A1'
    print(f"Cost of traveling from {loc} to {other}: {map.get_cost(loc, other)}")
    # ... but you can still get an estimation of the cost based on the straight line distance
    #     even when there is no direct connection between these points
    print(f"Straight Line distance from {loc} to {other}: {map.get_straight_line_distance(loc, other)}")


def main():
    # ... loading the map from JSON file ...
    map = CityMap.FromFile("./tegucigalpa.json")
    # ... loading the test cases from JSON file ...
    test_cases = SearchRequest.FromTestCasesFile("./test_cases.json")

    # TODO: Check the code for this function to understand how the map class works
    #       Afterwards, modify  or remove this line of code
    example_using_map_class(map)

    # TODO: Check the code for this function to understand how the test case class works
    #       Afterwards, modify or remove this line of code
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
