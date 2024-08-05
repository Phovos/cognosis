import math
import operator
import sys

class MathQuine:
    def __init__(self):
        self.result = 0
        self.operations = {
            'add': operator.add,
            'subtract': operator.sub,
            'multiply': operator.mul,
            'divide': operator.truediv,
            'power': operator.pow,
            'sqrt': math.sqrt
        }
        self.load_state()

    def load_state(self):
        # Read the source file and extract the stored result
        with open(__file__, 'r') as file:
            for line in file:
                if line.startswith('result_value ='):
                    self.result = float(line.split('=')[1].strip())
                    break

    def save_state(self):
        # Read the entire file content
        with open(__file__, 'r') as file:
            lines = file.readlines()

        # Update the result value in the file content
        with open(__file__, 'w') as file:
            for line in lines:
                if line.startswith('result_value ='):
                    file.write(f'result_value = {self.result}\n')
                else:
                    file.write(line)

    def perform_operation(self, operation, value=None):
        if operation not in self.operations:
            raise ValueError(f"Unknown operation: {operation}")

        if operation == 'sqrt':
            self.result = self.operations[operation](self.result)
        else:
            self.result = self.operations[operation](self.result, value)

    def run(self, operation, value=None):
        self.perform_operation(operation, value)
        self.save_state()
        print(f"Operation '{operation}' performed. New result: {self.result}")

if __name__ == "__main__":
    quine = MathQuine()
    
    if len(sys.argv) > 1:
        operation = sys.argv[1]
        if operation not in quine.operations:
            print(f"Unknown operation: {operation}")
            print("Available operations: add, subtract, multiply, divide, power, sqrt")
            sys.exit(1)
        
        # Only operations other than 'sqrt' need a value
        if operation != 'sqrt':
            if len(sys.argv) < 3:
                print(f"Operation '{operation}' requires a value.")
                sys.exit(1)
            value = float(sys.argv[2])
        else:
            value = None

        quine.run(operation, value)
    else:
        print("Usage: python script.py <operation> <value>")
        print("Available operations: add, subtract, multiply, divide, power, sqrt")
    
# Result placeholder
result_value = -9.0
