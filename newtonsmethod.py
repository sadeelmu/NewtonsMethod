import numpy as np
from numpy.linalg import solve

"""
Newton's Method to solve a system of non-linear equations:
Step 1. Find the Jacobian matrix at x^(0), where J is an nxn matrix that contains the partial derivatives.
Step 2. Find the G matrix at x^(0), where G is an nx1 matrix.
Step 3. Solve the linear system Jy^(0) = G to get y^(0).
Step 4. Define the next iteration using the iteration we found, i.e., x^(1) = x^(0) - y^(0). Continue solving and applying Newton's Method again.
Step 5. To stop the iterations, check for convergence and continue until the norm ||x^(1) - x^(0)|| < Error. Otherwise, continue iterations.
"""

def jacobian(f, x, h=1e-8):
    """
    Computes the Jacobian matrix of f at x, step 1 in the Method. 

    Returns J : The Jacobian matrix of f at x.
    """
    n = len(x)
    J = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        x1 = np.array(x, dtype=float)
        x1[i] += h
        J[:, i] = (f(x1) - fx) / h
    return J

def newton_method(f, x0, tol=1e-8, max_iter=2):
    """
    Solves a system of nonlinear equations using Newton's method.
    
    Returns x : The solution of the system.
    
    Raises: ValueError, If the method does not converge within the maximum number of iterations.
    """
    x = np.array(x0, dtype=float)
    for iteration in range(max_iter):
        J = jacobian(f, x)
        fx = f(x)
        y = solve(J, -fx)
        x_new = x + y
        print(f"Iteration {iteration}: x = {x_new}, f(x) = {f(x_new)}")
        if np.linalg.norm(x_new - x) < tol: #checks converegence and number of iterations here using norm
            return x_new
        x = x_new
    return x

def system(x):
    """
    Defines the system of nonlinear equations to be solved.
    
    Takes in: x : Input variables for the system of equations.
    
    Returns: Output values of the system of equations.
    """
    return np.array([
        4*x[0]**2 - 20*x[0] + 0.25*x[1]**2 + 8,
        0.5*x[0]*x[1]**2 + 2*x[0] - 5*x[1] + 8
    ])

x0 = [0, 0]

solution = newton_method(system, x0, max_iter=2)

print("Solution: ", solution)
print("Thank you Dr. Samia for everything <3")