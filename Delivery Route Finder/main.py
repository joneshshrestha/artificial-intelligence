"""
# ===============================================
#  Delivery Route Finder
# ===============================================
"""

import sys

from delivery_problem.map import CityMap
from delivery_problem.search_request import SearchRequest
from delivery_problem.problem import Problem
from search.search_algorithms import SearchAlgorithms


def main():
    # Load the city map from JSON file
    map = CityMap.FromFile("./tegucigalpa.json")
    # Load delivery scenarios from JSON file
    delivery_scenarios = SearchRequest.FromTestCasesFile("./test_cases.json")

    for scenario in delivery_scenarios:
        print("\n\nDelivery Scenario:")
        print(f" - Name: {scenario.get_name()}")
        print(f" - Starting Location: {scenario.get_start_location()}")
        print(f" - Delivery Locations: {scenario.get_targets()}")

        # Create the problem instance
        problem = Problem(map, scenario)

        # Run Breadth-First Search
        SearchAlgorithms.search(problem, SearchAlgorithms.BreadthFirstSearch)

        # Run Uniform Cost Search
        SearchAlgorithms.search(problem, SearchAlgorithms.UniformCostSearch)

        # Run A* Search
        SearchAlgorithms.search(problem, SearchAlgorithms.AStarSearch)


if __name__ == "__main__":
    main()
