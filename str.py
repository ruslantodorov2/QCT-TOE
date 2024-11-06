import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 299792458  # Speed of light in m/s
hbar = 1.0545718e-34  # Reduced Planck constant in J·s
alpha = 1.0  # Fine-structure constant

# Define string properties
string_length = 1e-35  # Length of the string in meters
string_tension = 1e30  # Tension of the string in N

# Define initial conditions
initial_position = 0.0  # Initial position of the string
initial_velocity = 1e6  # Initial velocity of the string

# Simulation parameters
time_step = 5.39e-44 * 1e3  # Time step for the simulation
total_time = 1e-40  # Extended total simulation time

# Initialize arrays to store position and velocity data
num_steps = int(total_time / time_step) + 1
position = np.zeros(num_steps)
velocity = np.zeros(num_steps)

# Set initial conditions
position[0] = initial_position
velocity[0] = initial_velocity

# Main simulation loop
for i in range(1, num_steps):
    # Calculate acceleration using string equation of motion
    acceleration = -(string_tension / hbar) * position[i - 1]
    
    # Update velocity and position
    velocity[i] = velocity[i - 1] + acceleration * time_step
    position[i] = position[i - 1] + velocity[i] * time_step

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_steps) * time_step, position, label="String Position")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("String Evolution in Time")
plt.grid(True)
plt.legend()
plt.show()

