import numpy as np
import matplotlib.pyplot as plt

# Define the Step function
def step(z):
    return 1 if z >= 0 else 0

# Vectorized step function for grid calculation
v_step = np.vectorize(step)

# Network 1 Parameters (Left Diagram)
# h1 weights: x1->1, x2->1, bias=0.5 (subtracted)
# h2 weights: x1->1, x2->1, bias=1.5 (subtracted)
# out weights: h1->1, h2->-1, bias=0.5 (subtracted)

def predict_network1(x1, x2):
    # Hidden Layer 1 (OR-like)
    z1 = (1.0 * x1) + (1.0 * x2) - 0.5
    h1 = v_step(z1)
    
    # Hidden Layer 2 (AND-like)
    z2 = (1.0 * x1) + (1.0 * x2) - 1.5
    h2 = v_step(z2)
    
    # Output Layer (Subtraction)
    z_out = (1.0 * h1) + (-1.0 * h2) - 0.5
    y = v_step(z_out)
    
    return y

# 1. Create a grid of points to classify
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 500), np.linspace(-0.5, 1.5, 500))
Z = predict_network1(xx, yy)

# 2. Setup the plot
plt.figure(figsize=(8, 8))

# 3. Plot the Decision Regions (Contour)
# Levels: 0 (Blue) and 1 (Red)
# Using slightly different colors to distinguish 'off' regions
plt.contourf(xx, yy, Z, levels=[-0.1, 0.5, 1.1], colors=['#d1e7dd', '#f8d7da'], alpha=0.6)

# 4. Plot the Decision Boundaries (Lines where net_input = 0)
x_vals = np.linspace(-0.5, 1.5, 100)

# Line for h1: x1 + x2 - 0.5 = 0  =>  x2 = -x1 + 0.5
y_h1 = -x_vals + 0.5
plt.plot(x_vals, y_h1, 'k--', linewidth=2, label=r'$h_1$ Boundary ($x_1+x_2=0.5$)')

# Line for h2: x1 + x2 - 1.5 = 0  =>  x2 = -x1 + 1.5
y_h2 = -x_vals + 1.5
plt.plot(x_vals, y_h2, 'k-', linewidth=2, label=r'$h_2$ Boundary ($x_1+x_2=1.5$)')

# 5. Plot the XOR Points
points = [
    (0, 0, 0), # x1, x2, Target
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0)
]

for x1, x2, target in points:
    output = predict_network1(x1, x2)
    color = 'red' if output == 1 else 'green' # Green if 0, Red if 1
    edge = 'black'
    
    # Mark the point
    plt.scatter(x1, x2, s=200, c=color, edgecolors=edge, zorder=10)
    
    # Annotate
    plt.text(x1 + 0.05, x2, f"({x1},{x2})\nOut:{output}", fontsize=12, fontweight='bold')

# Formatting
plt.title("Network 1 Decision Regions (Solves XOR)", fontsize=14)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)

plt.show()