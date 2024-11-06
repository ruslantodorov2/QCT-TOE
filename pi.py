import numpy as np
import matplotlib.pyplot as plt

# Constants
num_samples = 1000000  # Number of random samples
radius = 1.0           # Radius of the circle

# Generate random points in the unit square [0, 1] x [0, 1]
x = np.random.uniform(-radius, radius, num_samples)
y = np.random.uniform(-radius, radius, num_samples)

# Count how many points fall inside the quarter circle
inside_circle = np.sum(x**2 + y**2 <= radius**2)

# Estimate pi using the ratio of points inside the circle to total points
pi_estimate = (inside_circle / num_samples) * 4
print(f"Estimated value of pi with {num_samples} samples: {pi_estimate}")

# Optional: Visualize the points
def plot_simulation(x, y):
    plt.figure(figsize=(8, 8))
    plt.scatter(x[x**2 + y**2 <= radius**2], y[x**2 + y**2 <= radius**2], color='blue', s=1)  # Points inside the circle
    plt.scatter(x[x**2 + y**2 > radius**2], y[x**2 + y**2 > radius**2], color='red', s=1)      # Points outside the circle
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title('Monte Carlo Simulation of Pi')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.grid()
    plt.show()

# Uncomment the line below to visualize the simulation
# plot_simulation(x, y)

