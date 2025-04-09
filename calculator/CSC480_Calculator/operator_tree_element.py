
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
# DO NOT MODIFY
# =========================================
class OperatorTreeElement:
    def __init__(self, value):
        # this is a "protected" attribute of the class
        self._value = value

    def get_value(self):
        return self._value

    def __repr__(self):
        return str(self._value)

    def evaluate(self):
        # by default, children should override this function
        # DO NOT MODIFY
        raise NotImplementedError()

    def post_order_list(self, out_list):
        # this method is meant to be overridden bny the children classes
        # DO NOT MODIFY
        raise NotImplementedError()

    @staticmethod
    def BuildFromJSON(json_data):
        # this method is meant to be overridden bny the children classes
        # DO NOT MODIFY
        raise NotImplementedError()

