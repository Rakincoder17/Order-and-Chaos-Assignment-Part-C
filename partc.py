# Part-C: Phase Space Analysis
#Rakin Absar
#Roll No. 22PH23006
# i) Parameter calculation
p = 0
q = 6

a = 1 + p / 10
b = 1 + q / 10

print("Parameters calculated:")
print(f"a = {a:.1f}")
print(f"b = {b:.1f}")

### ii) Equilibrium points
import sympy as sp

# Define symbolic variables
x, y = sp.symbols('x y', real=True)

# Define the equations set to 0
eq1 = sp.Eq(a * x - y, 0)
eq2 = sp.Eq(x + b * y - (x**2 + y**2) * y, 0)

# Solve the system analytically
solutions = sp.solve((eq1, eq2), (x, y))

print("Analytical Equilibrium Points (x, y):")
for i, sol in enumerate(solutions, 1):
    # Convert symbolic results to floating point for clean output
    x_val = float(sol[0])
    y_val = float(sol[1])
    print(f"E{i} = ({x_val:.4f}, {y_val:.4f})")

### iii) Phase Space Plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Ensure a and b are defined in case this cell is run independently
a = 1.0
b = 1.6

# Define the domain (-3 to 3 for x and y)
x_dom = np.linspace(-3, 3, 25)
y_dom = np.linspace(-3, 3, 25)
X, Y = np.meshgrid(x_dom, y_dom)

# System of equations (Vector field)
U = a * X - Y
V = X + b * Y - (X**2 + Y**2) * Y

N = np.sqrt(U**2 + V**2)
N[N == 0] = 1 # Avoid division by zero
U_norm = U / N
V_norm = V / N

# Setup figure
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U_norm, V_norm, color='red', alpha=0.5)

# Define the ODE system
def sys_eq(t, Y_vec):
    x_val, y_val = Y_vec
    dx = a * x_val - y_val
    dy = x_val + b * y_val - (x_val**2 + y_val**2) * y_val
    return [dx, dy]

def out_of_bounds(t, Y_vec):
    x_val, y_val = Y_vec
    return 10.0 - np.sqrt(x_val**2 + y_val**2) # Triggers when radius hits 10
out_of_bounds.terminal = True

# Define 5 different initial conditions
init_conds = [
    [0.1, 0.1],   # Near origin
    [-0.1, -0.1], # Near origin
    [2.5, 0],     # Outer region right
    [-2.5, 0],    # Outer region left
    [0, 2.5]      # Outer region top
]

t_span = (0, 15)

# Plot trajectories
for ic in init_conds:
    sol = solve_ivp(sys_eq, t_span, ic, method='Radau', dense_output=True, events=out_of_bounds)
    
    # Generate smooth lines from the solution
    t_plot = np.linspace(sol.t[0], sol.t[-1], 500)
    z_plot = sol.sol(t_plot)
    
    plt.plot(z_plot[0], z_plot[1], 'b-', linewidth=1.5)
    # Mark the starting point with a green circle
    plt.plot(ic[0], ic[1], 'go', markersize=6)

# Formatting the plot
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Phase Portrait (a = {a:.1f}, b = {b:.1f})')
plt.grid(True)
plt.show()
