"""
# ===============================================
#  Search Tree Node class
# ===============================================
"""

from typing import Self, List

"""
    The SearchTreeNode class. This class represents the individual nodes of a 
    Search Tree. Every node is linked to its parent. Each node holds the action 
    used at the parent node to reach the current node, and also the total cost 
    of the path from the root to the node. We do not need to connect the nodes 
    to their children because the tree is constructed in a top-down fashion, 
    keeping references to all leaves (e.g., the frontier), and once that a goal
    state is explored, we simply navigate the path from the Node holding the goal
    state back to the root of the tree to generate the final solution 
    (e.g., an action sequence). 
     
    Note that this class is enough for implementing different search algorithms.
    We might create a SearchTree class, but we really do not need it. Also, This
    class is very generic, it is totally independent of the problem. All data 
    related to the problem itself is stored on the linked State object.  
    
    In addition, a "node id" attribute is included. This attribute is just a 
    simple correlative which counts how many nodes have been created so far. 
    This attribute is used as the default key for sorting elements of this 
    class. That is, this is the attribute used to determine if a given node is
    "smaller" than another in the "__lt__" method (the method called when you 
    do "a < b"). However, SearchTreeNodes should be sorted externally using the 
    priority criteria that corresponds to each search algorithm. If after using 
    these criteria, two nodes still have the same priority, then the tie will 
    be broken by node id. In other words, the node that was created first will
    have the highest priority (FIFO) for those ties.     
"""


class SearchTreeNode:
    NodeCount = 0

    """
        Constructor
    """

    def __init__(self, parent: Self | None, action: str | None, state, path_cost):
        self.__parent = parent
        self.__action = action
        self.__state = state
        self.__path_cost = path_cost

        # generate a unique ID for each node
        self.__node_id = SearchTreeNode.NodeCount
        SearchTreeNode.NodeCount += 1

    def get_parent(self):
        return self.__parent

    def get_action(self):
        return self.__action

    def get_state(self):
        return self.__state

    def get_path_cost(self):
        return self.__path_cost

    """
        Default sorting for Search Tree Nodes (older goes first). However, the
        main sorting criteria (the priority) needs to be defined externally
        in accordance to the search algorithm being used.  
    """

    def __lt__(self, other):
        # by default, simply sort nodes by "age", older nodes come first ...
        return self.__node_id < other.__node_id

    """
        Reconstruct the path from the current node back to the root, and
        return it as a list of actions. This represents the sequence of actions
        that need to be executed to reach the current state starting from
        the root node (the initial state). 
    """

    def path_to_root(self) -> List[str]:
        full_path = []
        current = self
        while current.__action is not None:
            full_path.insert(0, current.__action)
            current = current.__parent

        return full_path
