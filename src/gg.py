import ast

class LambdaModifier(ast.NodeTransformer):
    def __init__(self, operation):
        self.operation = operation

    def visit_Lambda(self, node):
        if self.operation == "multiply":
            node.body = ast.BinOp(left=node.body, op=ast.Mult(), right=ast.Constant(value=2))
        elif self.operation == "subtract":
            node.body = ast.BinOp(left=node.body, op=ast.Sub(), right=ast.Constant(value=1))
        elif self.operation == "divide":
            node.body = ast.BinOp(left=node.body, op=ast.Div(), right=ast.Constant(value=2))
        # Add more operations as needed
        return node

def transform_lambda(source_code, operation):
    tree = ast.parse(source_code, mode='eval')
    modifier = LambdaModifier(operation)
    modified_tree = modifier.visit(tree)
    return ast.unparse(modified_tree)

if __name__ == "__main__":
    source_code = "lambda x: x + 2"
    for operation in ["multiply", "subtract", "divide"]:
        modified_code = transform_lambda(source_code, operation)
        print(f"Operation: {operation}, Modified Code: {modified_code}")

source_code = "lambda x: x + 2"
operations = ["multiply", "subtract", "divide"]

for operation in operations:
    modified_code = transform_lambda(source_code, operation)
    print(f"Operation: {operation}, Modified Code: {modified_code}")