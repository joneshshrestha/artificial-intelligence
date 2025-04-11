
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
#  TODO: Modified by: Jonesh Shrestha
#  TODO: Modified When: 10/04/2024
# =========================================

from .operand import Operand
from .operator import Operator
class OperatorTree:
    def __init__(self, root):
        # this is a "private" attribute of the class
        self.__root = root

    def evaluate(self):
        # TODO: evaluate the expression .. starting from the root
        raise NotImplementedError()

    def post_order_list(self):
        # TODO: create a post-order traversal .. starting from the root
        # HINT: you will need a list to put the results.
        raise NotImplementedError()

    @staticmethod
    def BuildFromJSON(json_data):
        # TODO: This function should create the tree using the provided JSON data
        #       this might need to check the type of the root node because
        #       root could be either an Operator or an Operand
        # TODO: after creating the tree, it should return an instance of this
        #       class (OperatorTree), with the root value properly set up as the
        #       root node of the tree.

        if 'operator_tree' not in json_data:
            raise ValueError("JSON file doesn't contain 'operator_tree'.")
        # access json_data operator_tree and converts the json_data into a dictionary
        root_node = json_data['operator_tree']
        
        # access the value of the key 'type' of the root_node
        root_node_type = root_node.get('type')
        # pass the info to the Operator if type operator is found
        if root_node_type == 'operator':
            root = Operator.BuildFromJSON(root_node)
        # pass the info to the Operand if type number is found
        elif root_node_type == 'number':
            root = Operand.BuildFromJSON(root_node)
        else:
            raise ValueError("Node type not recognized.")
        
        return OperatorTree(root_node)