
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
#  Modified by: Jonesh Shrestha
#  Modified When: 08/04/2024
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
        # Apply the local operator and return the value

        # If the children item is an operand, calls the operand evaluate method
        # Else recursively call the evaluate method of this class 
        left_value = self.__children[0].evaluate()
        right_value = self.__children[1].evaluate()

        if self._value == '+':
            return left_value + right_value
        elif self._value == '-':
            return left_value - right_value
        elif self._value == '*':
            return left_value * right_value
        elif self.__children == '/':
            if right_value != 0:
                return left_value / right_value
            else:
                raise ZeroDivisionError("Cannot divide with 0.")
        else:
            raise ValueError(f'Unknown operator: {self._value}')

    def post_order_list(self, out_list):
        # Overrides the post_order_list function from parent class.
        # Add itself and its children ... all in post-order

        # If the children item is an operand, calls the operand post_order_list method which calls repr and appends the value 
        # Else recursively call the post_order_list method of this class 
        for child in self.__children:
            child.post_order_list(out_list)
        # repr called to inside the list which extracts it's value
        out_list.append(self)

    @staticmethod
    def BuildFromJSON(json_data):
        # Overrides the BuildFromJSON function from parent class.
        # JSON data is used to create and return a valid Operator object
        # which in turn requires recursively creating its children.

        # This function assumes that json_data contains the info for an Operator Node and all of its children, and children of its children, etc.

        value = json_data.get('value')
        children = []

        for child_json in json_data.get('operands'):
            if child_json.get('type') == 'operator':
                child = Operator.BuildFromJSON(child_json)
            elif child_json.get('type') == 'number':
                child = Operand.BuildFromJSON(child_json)
            else:
                raise ValueError("Node type not recognized.")
            
            children.append(child)

        return Operator(value, children)
    

