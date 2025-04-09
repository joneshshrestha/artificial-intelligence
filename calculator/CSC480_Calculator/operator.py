
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
#  TODO: Modified by: ???
#  TODO: Modified When: ???
# =========================================


from .operator_tree_element import OperatorTreeElement
from .operand import Operand

class Operator(OperatorTreeElement):
    def __init__(self, value, children):
        super().__init__(value)
        # this is a "private" attribute of the class
        self.__children = children

    def evaluate(self):
        # Overrides the evaluate function from parent class.
        # TODO: apply the local operator and return the value
        #       - self._value == "+" ?
        #       - self._value == "*" ?
        #       - self._value == "-" ?
        #       - self._value == "/" ?
        raise NotImplementedError()

    def post_order_list(self, out_list):
        # Overrides the post_order_list function from parent class.
        # TODO: Should add itself and its children ... all in post-order
        # hint: recursion is needed
        raise NotImplementedError()

    @staticmethod
    def BuildFromJSON(json_data):
        # Overrides the BuildFromJSON function from parent class.
        # TODO: Use  JSON data is used to create and return a valid Operator object
        #       which in turn requires recursively creating its children.
        #
        #  This function assumes that json_data contains the info for an Operator Node
        #     and all of its children, and children of its children, etc.

        raise NotImplementedError()

