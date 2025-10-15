"""
# ===============================================
#  State representation for delivery problem
# ===============================================
"""

from typing import List

"""
    The State class. This class is problem-specific.  It is used to represent 
    a Single State (a node) of the State Space.
    
    For the delivery problem, this class includes:
    - The current location of the simulated driver
    - The status of each target that needs to be visited
    
    Note that the Search Request class includes the concrete list of locations 
    to visit. The visited_target attribute includes a list of boolean values 
    which must be aligned with that list of locations. Every value represents if 
    a target location has already been visited (True) or not (False). 
    
    For example, 
       if a search requires visiting locations ['F2', 'A2', 'K3', 'X4', 'D2']
                         and visited_targets = [True, False, False, True, False], 
       that would mean that locations 'F2' and 'X4' have been visited already, 
       and that 'A2', 'K3' and 'D2' are still pending.
    
    Note that the locations in the Search Request object are NOT sorted. These 
    locations simply follow whatever order was specified in the input file. This 
    order should not have any effect on the search process. 
    
    Finally, the atomic attribute keeps an "atomic representation" of the whole
    state, which is required for keeping records of the nodes that have been 
    reached and explored.      
"""


class State:
    def __init__(self, current_loc: str, visited_targets: List[bool]):
        # this is the location of the delivery driver
        self.__current_loc = current_loc
        # this is a list of boolean values indicating if
        self.__visited_targets = visited_targets

        # create the atomic representation ...
        self.__atomic = (
            current_loc
            + "-"
            + "".join([("1" if visited else "0") for visited in visited_targets])
        )

    def get_location(self):
        return self.__current_loc

    def get_visited_targets(self):
        return self.__visited_targets

    def get_representation(self):
        return self.__atomic
