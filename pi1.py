import numpy as np
import matplotlib.pyplot as plt

# Constants
num_particles = 1000000  # Increase the number of particles for better accuracy
radius = 1.0             # Radius of the circle

# Generate random points in the unit square [-1, 1] x [-1, 1]
x_positions = np.random.uniform(-radius, radius, num_particles)
y_positions = np.random.uniform(-radius, radius, num_particles)

# Function to estimate pi based on particle positions
def estimate_pi_from_particles(x_positions, y_positions):
    inside_circle = np.sum(x_positions**2 + y_positions**2 <= radius**2)
    total_particles = len(x_positions)
    pi_estimate = (inside_circle / total_particles) * 4  # Area ratio method
    return pi_estimate

# Estimate pi
pi_estimate = estimate_pi_from_particles(x_positions, y_positions)
print(f"Estimated value of pi based on particle positions: {pi_estimate}")

# Optional: Visualize the particles
def plot_particles(x_positions, y_positions):
    plt.figure(figsize=(8, 8))
    plt.scatter(x_positions[x_positions**2 + y_positions**2 <= radius**2], 
                y_positions[x_positions**2 + y_positions**2 <= radius**2], 
                color='blue', s=1)  # Points inside the circle
    plt.scatter(x_positions[x_positions**2 + y_positions**2 > radius**2], 
                y_positions[x_positions**2 + y_positions**2 > radius**2], 
                color='red', s=1)   # Points outside the circle
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title('Monte Carlo Simulation of Pi')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.grid()
    plt.show()

# Uncomment the line below to visualize the particles
# plot_particles(x_positions, y_positions)

