
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 380/480
#  DO NOT MODIFY / DO NOT REDISTRIBUTE!!
# ===============================================
"""
import json
from typing import List

"""
    This class is used to represent a Search that needs to be executed. 
    Each search has a name used to identify individual test cases.
    Each search begins from a specific location (the start point), it
    requires visiting a list of targets (a list of locations) and a 
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
        Returns the name of this Search Request (a.k.a. the test-case name)
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
        This function reads a JSON file containing multiple Search Requests (test cases).
        The data for each request is loaded from the file, and a new instance of 
        SearchRequest is created. The function returns a list of all request found in the
        file   
    """
    @staticmethod
    def FromTestCasesFile(filename: str):
        with open(filename, "r", encoding="utf-8") as in_file:
            raw_data = json.load(in_file)

        if not "cases" in raw_data:
            raise Exception("Invalid Test Cases file")

        all_search_requests = []
        for case_info in raw_data["cases"]:
            name = case_info["name"]
            start = case_info["start"]
            deliveries = case_info["deliveries"]

            test_case = SearchRequest(name, start, deliveries)
            all_search_requests.append(test_case)

        return all_search_requests
