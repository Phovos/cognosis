import ast

# Define a lambda function
source_code = "lambda x: x + 2"

# Parse the source code into an AST
tree = ast.parse(source_code, mode='eval')

# Print the AST representation
print(ast.dump(tree, annotate_fields=True))

# Modify the lambda function
class LambdaModifier(ast.NodeTransformer):
    def visit_Lambda(self, node):
        # Change the body of the lambda function
        node.body = ast.BinOp(left=node.body, op=ast.Mult(), right=ast.Constant(value=2))
        return node

# Apply the modification
modifier = LambdaModifier()
modified_tree = modifier.visit(tree)

# Convert the AST back to source code using ast.unparse
modified_code = ast.unparse(modified_tree)
print(modified_code)