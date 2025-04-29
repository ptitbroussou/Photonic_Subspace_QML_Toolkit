from sympy import *

from sympy.solvers.solveset import linsolve


### Define the symbols
# Circuit parameters:
list_alpha = [[symbols('alpha_{}{}'.format(i,j)) for j in range(1,5)] for i in range(1,5)]
list_beta = [[symbols('beta_{}{}'.format(i,j)) for j in range(1,7)] for i in range(1,7)]
# Input state:
list_x = [[symbols('x_{}{}'.format(i,j)) for j in range(1,7)] for i in range(1,4)]


# Define the equations:
list_eq = [Eq(sum(list_alpha[i])*sum(list_beta[j]), list_x[i][j]) for i in range(3) for j in range(6)]


# Output the symbolic solution:
sol = solve(list_eq, [list_x[i][j] for i in range(3) for j in range(6)])
print(sol)