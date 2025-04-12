
# =========================================
#  Created by: Kenny Davila Castellanos
#      For: CSC 480 - AI 1
#
#  Modified by: Jonesh Shrestha
#  Modified When: 08/04/2024
# =========================================


import sys
import json

from CSC480_Calculator import OperatorTree, Operator, Operand


def stack_based_evaluation(post_order):
    # This is a simple evaluation algorithm, using a stack
    # it sequentially reads the mathematical expression in post-fix notation
    #        - every time it finds an operand (A)
    #             it will simply put (A) on the stack
    #        - every time it finds an operator (OP)
    #             it will take 2 items from stack, (B) and (A)
    #             it must compute result by applying operator
    #                (RES) = (A) (OP) (B)
    #             it will save (RES) to the stack
    #             mind that the ORDER OF THE OPERANDS might affect the result
    #       it does it until the end.

    # Remember:
    #    If the expression is valid, only one item will be there at the end
    #    this is the solution, and must be returned
    #
    
    # isinstance function is used to check the types of the elements on the post_order list
    # get_value() to access value outside of the package
    stack = []
    for item in post_order:
        if isinstance(item, Operand):
            stack.append(item.get_value())
        elif isinstance(item, Operator):
            right_operand_B = stack.pop()
            left_operand_A = stack.pop()
            if item.get_value() == '+':
                stack.append(left_operand_A + right_operand_B)
            elif item.get_value() == '-':
                stack.append(left_operand_A - right_operand_B)
            elif item.get_value() == '*':
                stack.append(left_operand_A * right_operand_B)
            elif item.get_value() == '/':
                if right_operand_B != 0:
                    stack.append(left_operand_A / right_operand_B)
                else:
                     raise ZeroDivisionError("Cannot divide with 0.")
            else:
                raise ValueError(f'Unknown operator: {item.get_value()}')
    return stack.pop()


def main():
    # handling the command line arguments , given do not change!
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} filename")
        return

    # Step 1
    # Load the JSON data
    #     The actual filename to load should be on sys.argv ...
    #     * position 0 is always the path to the script
    #     * position 1 ... n are your custom command line arguments

    # Load the input File using the JSON library
    try:
        with open(sys.argv[1], 'r') as file_name:
            json_data = json.load(file_name)
    except FileNotFoundError:
        print(f'Cannot find file named {sys.argv[1]}')
        return

    # Step 2
    # Load the expression from file using the OperatorTree.BuildFromJSON function
    expression = OperatorTree.BuildFromJSON(json_data)
    # TODO: You must implement the functions
    #       - OperatorTree.BuildFromJSON
    #       - Operand.BuildFromJSON
    #       - Operator.BuildFromJSON
    # Step 3
    # TODO: Evaluate the expression (using the evaluate function of the OperatorTree class)
    evaluate_tree_expression = expression.evaluate()
    print(f'Operator Tree Evaluation Result: {evaluate_tree_expression}')

    # Step 4
    # TODO: Generate a list of the elements on the Operator Tree in post-order and print it!
    post_order_result_list = expression.post_order_list()
    print(f'Post Order: {post_order_result_list}')

    # Step 5
    # TODO: Evaluate the expression (again) but using the post fix notation and a stack
    #       This must be done by calling stack_based_evaluation
    evaluate_post_order_stack = stack_based_evaluation(post_order_result_list)
    print(f'Post Order Stack Evaluation Result: {evaluate_post_order_stack}')



if __name__ == "__main__":
    main()

