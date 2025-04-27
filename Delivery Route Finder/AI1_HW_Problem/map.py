
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 380/480
#  DO NOT MODIFY / DO NOT REDISTRIBUTE!!
# ===============================================
"""
import json
import math
from typing import List, Dict, Tuple


"""
    This class is used to represent the Map of a city (part of the underlying problem).
    It includes information about the city name, the locations of interest in the map, 
    with their known 2D position. It also includes connectivity information that can be 
    used to determine if a location can be reached from another, and what is the 
    corresponding cost (distance in miles). The relative scale represents the value that
    was used to convert the raw distances between 2D points in the map to miles.        
"""


class CityMap:
    """
        Constructor for CityMap class. It takes all the info provided in the JSON file
    """
    def __init__(self, name: str, locations: List[str], connections: Dict[str, Dict[str, float]],
                 loc_positions: Dict[str, Tuple[float, float]], map_scale: float):
        self.__name = name
        self.__locations = locations
        self.__connections = connections
        self.__map_positions = loc_positions
        self.__map_scale = map_scale

        # validate ...
        if len(self.__locations) != len(self.__connections):
            raise Exception("Invalid map: The number of locations does not match the number of origins for connections")

        common = set(self.__locations).intersection(self.__connections.keys())
        if len(common) != len(self.__locations):
            raise Exception("Invalid map: Mismatch in locations and origins of connections")

        # Pre-computing pairwise straight line distances
        self.__pairwise_distance_cache = {src: {} for src in self.__locations}
        for src_idx in range(len(self.__locations)):
            src = self.__locations[src_idx]
            # diagonal
            self.__pairwise_distance_cache[src][src] = 0.0
            for dst_idx in range(src_idx + 1, len(self.__locations)):
                dst = self.__locations[dst_idx]
                distance = self.__compute_straight_line_distance(src, dst)
                # note that the lower-bound of distance is symmetric even if real shortest distances aren't
                self.__pairwise_distance_cache[src][dst] = distance
                self.__pairwise_distance_cache[dst][src] = distance

    """
        Returns the name of the city
    """
    def get_name(self):
        return self.__name

    """
        Retrieves the list of map locations
    """
    def get_locations(self):
        return self.__locations

    """
        Returns a list of the neighbors of a given location. 
        It throws an error if the location does not exists on the map.
    """
    def get_neighbors(self, location: str):
        if location in self.__connections:
            return list(self.__connections[location].keys())
        else:
            raise Exception(f"Invalid Location: {location}")

    """
        Get the actual distance (or cost) between two connected locations.
        If the source does not exist, it throws an exception.
        If the source and destination are not directly connected, it returns null.
        Otherwise, it returns the value as provided in the original map file.
    """
    def get_cost(self, src: str, dst: str):
        if src in self.__connections:
            if dst in self.__connections[src]:
                # return the actual cost
                return self.__connections[src][dst]
            else:
                # this means no connection was found ...
                return None
        else:
            raise Exception(f"Invalid Source Location: {src}")

    """
        Compute and return the straight line distance between any two locations
        This function does NOT return the actual distance between two locations, 
        because finding such value requires a search on its own.
        Instead, this function simply returns the Straight Line distance between two points in the map
        which represents a lower-bound for the real distance between the two locations.
        The locations in question do not need to be directly connected.
    """
    def __compute_straight_line_distance(self, src, dst) -> float:
        src_x, src_y = self.__map_positions[src]
        dst_x, dst_y = self.__map_positions[dst]

        raw_distance = math.sqrt(math.pow(src_x - dst_x, 2.0) + math.pow(src_y - dst_y, 2.0))
        scaled_distance = raw_distance / self.__map_scale

        return scaled_distance

    """
        Returns the straight line distance between any two locations in the map.
        To avoid recomputing this info, the constructor will create a cache. 
        This function simply returns the cached values          
    """
    def get_straight_line_distance(self, src, dst) -> float:
        # simply use the cached values
        return self.__pairwise_distance_cache[src][dst]

    """
        Creates a new instance of a CityMap based on a JSON file. 
    """
    @staticmethod
    def FromFile(filename: str):
        with open(filename, "r", encoding="utf-8") as in_file:
            raw_data = json.load(in_file)

        name = raw_data["name"]
        locations = raw_data["nodes"]
        map_positions = raw_data["positions"]
        rel_scale = raw_data["scale"]

        connections = {}
        for src in raw_data["edges"]:
            connections[src] = {dst: cost for dst, cost in raw_data["edges"][src]}

        return CityMap(name, locations, connections, map_positions, rel_scale)
