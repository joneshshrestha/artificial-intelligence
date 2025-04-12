
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
#  Modified by: Jonesh Shrestha
#  Modified When: 08/04/2024
# =========================================

from .operand import Operand
from .operator import Operator
class OperatorTree:
    def __init__(self, root):
        # this is a "private" attribute of the class
        self.__root = root

    def evaluate(self):
        # Evaluate the expression .. starting from the root
        return self.__root.evaluate()

    def post_order_list(self):
        # Create a post-order traversal .. starting from the root
        # HINT: you will need a list to put the results.
        result = []
        # pass the result array as out_list array to methods
        self.__root.post_order_list(result)
        return result

    @staticmethod
    def BuildFromJSON(json_data):
        # Create the tree using the provided JSON data and check the type of the root node because root could be either an Operator or an Operand
        # After creating the tree, it should return an instance of this
        # class (OperatorTree), with the root value properly set up as the
        # root node of the tree.

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
            raise ValueError(f'Node type not recognized. {root_node_type}')
        
        return OperatorTree(root)