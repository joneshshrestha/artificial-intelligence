
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
#  Modified by: Jonesh Shrestha
#  Modified When: 08/04/2024
# =========================================


from .operator_tree_element import OperatorTreeElement


class Operand(OperatorTreeElement):
    def __init__(self, value):
        super().__init__(value)

    def evaluate(self):
        # Overrides the evaluate function from parent class.
        # Return it's current value
        return self._value

    def post_order_list(self, out_list):
        # Overrides the post_order_list function from parent class.
        # Add itself to the stack, repr extract the value
        out_list.append(self)

    @staticmethod
    def BuildFromJSON(json_data):
        # Overrides the BuildFromJSON function from parent class.
        # Use JSON data to create a valid Operand Object
        # this function assumes that json_data only contains the info for an Operand Node
        value = json_data.get('value')
        return Operand(value)

