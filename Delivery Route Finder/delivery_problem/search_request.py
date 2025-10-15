"""
# ===============================================
#  Search Request for delivery routing
# ===============================================
"""

import json
from typing import List

"""
    This class is used to represent a delivery route search request. 
    Each search has a name used to identify individual scenarios.
    Each search begins from a specific location (the start point), it
    requires visiting a list of target delivery locations and a 
    valid solution must end with the driver being back at the starting 
    location.  
"""


class SearchRequest:
    """
    Constructor
    """

    def __init__(self, name: str, start: str, targets: List[str]):
        self.__name = name
        self.__start = start
        self.__targets = targets

    """
        Returns the name of this Search Request (delivery scenario name)
    """

    def get_name(self):
        return self.__name

    """
        Returns the starting (and also the final) location 
    """

    def get_start_location(self):
        return self.__start

    """
        Returns the list of locations to visit. 
    """

    def get_targets(self):
        return self.__targets

    """
        This function reads a JSON file containing multiple Search Requests (delivery scenarios).
        The data for each request is loaded from the file, and a new instance of 
        SearchRequest is created. The function returns a list of all requests found in the file.
    """

    @staticmethod
    def FromTestCasesFile(filename: str):
        with open(filename, "r", encoding="utf-8") as in_file:
            raw_data = json.load(in_file)

        if not "cases" in raw_data:
            raise Exception("Invalid search scenarios file")

        all_search_requests = []
        for case_info in raw_data["cases"]:
            name = case_info["name"]
            start = case_info["start"]
            deliveries = case_info["deliveries"]

            test_case = SearchRequest(name, start, deliveries)
            all_search_requests.append(test_case)

        return all_search_requests
