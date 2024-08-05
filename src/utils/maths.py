import math
from typing import Union, Tuple, List, Callable
from functools import reduce
import operator

# Constants
SECONDS_IN_MINUTE = 60
MINUTES_IN_HOUR = 60
SECONDS_IN_HOUR = MINUTES_IN_HOUR * SECONDS_IN_MINUTE
PI = math.pi
E = math.e
GOLDEN_RATIO = (1 + 5**0.5) / 2

# Type aliases
Number = Union[int, float]
Vector = List[Number]
Matrix = List[List[Number]]
Point2D = Tuple[Number, Number]
Point3D = Tuple[Number, Number, Number]
Rectangle = Tuple[Number, Number, Number, Number]

# Advanced mathematical functions

def matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    """Multiply two matrices
    
    Args:
    a (Matrix): First matrix
    b (Matrix): Second matrix
    
    Returns:
    Matrix: Result of matrix multiplication
    """
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))]
            for i in range(len(a))]

def matrix_transpose(matrix: Matrix) -> Matrix:
    """Transpose a matrix
    
    Args:
    matrix (Matrix): Input matrix
    
    Returns:
    Matrix: Transposed matrix
    """
    return list(map(list, zip(*matrix)))

def determinant(matrix: Matrix) -> Number:
    """Calculate the determinant of a square matrix
    
    Args:
    matrix (Matrix): Square matrix
    
    Returns:
    Number: Determinant of the matrix
    """
    if len(matrix) == 1:
        return matrix[0][0]
    return sum((-1) ** j * matrix[0][j] * determinant([row[:j] + row[j+1:] for row in matrix[1:]])
               for j in range(len(matrix)))

def cross_product(a: Vector, b: Vector) -> Vector:
    """Calculate the cross product of two 3D vectors
    
    Args:
    a (Vector): First 3D vector
    b (Vector): Second 3D vector
    
    Returns:
    Vector: Cross product
    """
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]

def vector_magnitude(v: Vector) -> Number:
    """Calculate the magnitude of a vector
    
    Args:
    v (Vector): Input vector
    
    Returns:
    Number: Magnitude of the vector
    """
    return math.sqrt(sum(x**2 for x in v))

def normalize_vector(v: Vector) -> Vector:
    """Normalize a vector
    
    Args:
    v (Vector): Input vector
    
    Returns:
    Vector: Normalized vector
    """
    mag = vector_magnitude(v)
    return [x / mag for x in v]

def least_common_multiple(a: int, b: int) -> int:
    """Calculate the least common multiple of two integers
    
    Args:
    a (int): First integer
    b (int): Second integer
    
    Returns:
    int: Least common multiple
    """
    return abs(a * b) // math.gcd(a, b)

def derivative(f: Callable[[Number], Number], h: float = 1e-5) -> Callable[[Number], Number]:
    """Return the derivative of a function
    
    Args:
    f (Callable[[Number], Number]): Input function
    h (float): Small value for approximation (default: 1e-5)
    
    Returns:
    Callable[[Number], Number]: Derivative function
    """
    return lambda x: (f(x + h) - f(x)) / h

def integral(f: Callable[[Number], Number], a: Number, b: Number, n: int = 1000) -> Number:
    """Approximate the definite integral of a function using the trapezoidal rule
    
    Args:
    f (Callable[[Number], Number]): Input function
    a (Number): Lower bound
    b (Number): Upper bound
    n (int): Number of trapezoids (default: 1000)
    
    Returns:
    Number: Approximation of the definite integral
    """
    h = (b - a) / n
    return h * (f(a) / 2 + sum(f(a + i * h) for i in range(1, n)) + f(b) / 2)

def newton_raphson(f: Callable[[Number], Number], df: Callable[[Number], Number], x0: Number, epsilon: float = 1e-6, max_iter: int = 100) -> Number:
    """Find the root of a function using the Newton-Raphson method
    
    Args:
    f (Callable[[Number], Number]): Function to find the root of
    df (Callable[[Number], Number]): Derivative of the function
    x0 (Number): Initial guess
    epsilon (float): Tolerance for convergence (default: 1e-6)
    max_iter (int): Maximum number of iterations (default: 100)
    
    Returns:
    Number: Approximate root of the function
    """
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < epsilon:
            return x
        x = x - fx / df(x)
    raise ValueError("Newton-Raphson method did not converge")

def taylor_series(f: Callable[[Number], Number], a: Number, n: int) -> Callable[[Number], Number]:
    """Generate the Taylor series expansion of a function
    
    Args:
    f (Callable[[Number], Number]): Function to expand
    a (Number): Point around which to expand
    n (int): Number of terms in the series
    
    Returns:
    Callable[[Number], Number]: Taylor series expansion
    """
    def factorial(k):
        return 1 if k == 0 else k * factorial(k - 1)
    
    derivatives = [f]
    for _ in range(1, n):
        derivatives.append(derivative(derivatives[-1]))
    
    return lambda x: sum(derivatives[i](a) * (x - a)**i / factorial(i) for i in range(n))

# Example usage
if __name__ == "__main__":
    # Matrix operations
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[5, 6], [7, 8]]
    print("Matrix multiplication:", matrix_multiply(matrix_a, matrix_b))
    print("Matrix transpose:", matrix_transpose(matrix_a))
    print("Matrix determinant:", determinant(matrix_a))

    # Vector operations
    vector_a = [1, 2, 3]
    vector_b = [4, 5, 6]
    print("Cross product:", cross_product(vector_a, vector_b))
    print("Vector magnitude:", vector_magnitude(vector_a))
    print("Normalized vector:", normalize_vector(vector_a))

    # Number theory
    print("LCM of 12 and 18:", least_common_multiple(12, 18))

    # Calculus
    def f(x): return x**2
    def df(x): return 2*x
    print("Derivative of x^2 at x=3:", derivative(f)(3))
    print("Integral of x^2 from 0 to 1:", integral(f, 0, 1))
    print("Root of x^2 - 5 using Newton-Raphson:", newton_raphson(lambda x: x**2 - 5, lambda x: 2*x, 1))

    # Taylor series
    exp_taylor = taylor_series(math.exp, 0, 5)
    print("Taylor series of e^x around 0 (5 terms):", exp_taylor)
    print("e^1 approximation:", exp_taylor(1))
    print("Actual e^1:", math.exp(1))
