import numpy as np
import pandas as pd
import json
import os

# Constants
c = 299792458  # Speed of light in m/s
E_mc2 = c**2  # Mass-energy equivalence in J/kg
TSR = E_mc2 / (1.38e-23)  # Temperature to Speed Ratio in K/m/s
alpha = 1.0  # Proportional constant for TSR
Q = 2 ** (1 / 12)  # Fractal structure parameter
dark_energy_density = 5.96e-27  # Density of dark energy in kg/m^3
dark_matter_density = 2.25e-27  # Density of dark matter in kg/m^3
collision_distance = 1e-10  # Distance for collision detection
Hubble_constant = 70.0  # km/s/Mpc (approximation)
Hubble_constant_SI = (
    Hubble_constant * 1000 / 3.086e22
)  # Hubble constant in SI units (s^-1)

# Initial conditions
temperature_initial = 1.42e32  # Planck temperature in K
particle_density_initial = 5.16e96  # Planck density in kg/m^3
particle_speed_initial = c  # Initially at the speed of light

# Simulation time
t_planck = 5.39e-44  # Planck time in s
t_simulation = t_planck * 1e5  # Shorter timescale for simulation

# Quark masses (in GeV) - used for initial mass values and comparison
quark_masses = {
    "up": 2.3e-3,
    "down": 4.8e-3,
    "charm": 1.28,
    "strange": 0.095,
    "top": 173.0,
    "bottom": 4.18,
    "electron": 5.11e-4,
    "muon": 1.05e-1,
    "tau": 1.78,
    "photon": 0,
}

# Conversion factor from GeV to J
GeV_to_J = 1.60217662e-10

# Simulation setup
num_steps = int(t_simulation / t_planck)

# Tunneling probabilities to investigate
tunneling_probabilities = np.arange(0.01, 2.5, 0.1)  # Adjust range as needed

# Create a directory to store the data
data_dir = "big_bang_simulation_data"
os.makedirs(data_dir, exist_ok=True)

# Functions to incorporate relativistic effects and collisions
def relativistic_energy(particle_speed, particle_mass):
    if particle_speed >= c:
        return np.inf
    return particle_mass * c**2 / np.sqrt(max(1e-10, 1 - (particle_speed / c) ** 2))

def relativistic_momentum(particle_speed, particle_mass):
    if particle_speed >= c:
        return np.inf
    return (
        particle_mass
        * particle_speed
        / np.sqrt(max(1e-10, 1 - (particle_speed / c) ** 2))
    )

def update_speed(current_speed, current_temperature, particle_mass):
    rel_momentum = relativistic_momentum(current_speed, particle_mass)
    return c * np.sqrt(
        max(1e-10, 1 - (rel_momentum / (rel_momentum + dark_energy_density)) ** 2)
    )

def check_collision(particle_speeds, collision_distance):
    # Assuming 1D for simplicity. Expand for 3D if needed.
    for j in range(len(particle_speeds)):
        for k in range(j + 1, len(particle_speeds)):
            if np.abs(particle_speeds[j] - particle_speeds[k]) < collision_distance:
                return True, j, k
    return False, -1, -1

def handle_collision(particle_speeds, idx1, idx2):
    # Exchange momentum for a simplified collision response
    p1 = relativistic_momentum(particle_speeds[idx1], particle_masses[idx1])
    p2 = relativistic_momentum(particle_speeds[idx2], particle_masses[idx2])
    
    # Simplified exchange
    particle_speeds[idx1], particle_speeds[idx2] = p2 / particle_masses
[idx1], p1 / particle_masses[idx2]

# Simulate the Big Bang with Dark Energy, Dark Matter, Tunneling, and Relativistic Effects
for tunneling_probability in tunneling_probabilities:
    print(f"Simulating for tunneling probability: {tunneling_probability}")

    # Initialize arrays for simulation
    particle_speeds = np.zeros((len(quark_masses), num_steps))  # 2D array for speeds
    particle_temperatures = np.zeros((len(quark_masses), num_steps))  # 2D array for temperatures
    particle_masses_evolution = np.zeros((len(quark_masses), num_steps))  # 2D array for mass evolution
    tunneling_steps = np.zeros((len(quark_masses), num_steps), dtype=bool)  # 2D array for tunneling steps

    # Create an array of masses for each quark
    particle_masses = np.array([mass * GeV_to_J for mass in quark_masses.values()])

    for j, (quark, mass) in enumerate(quark_masses.items()):
        # Initialize particle speeds and temperatures
        particle_speeds[j, 0] = particle_speed_initial
        particle_temperatures[j, 0] = temperature_initial
        particle_masses_evolution[j, 0] = mass * GeV_to_J  # Convert to Joules

    # Time evolution loop
    for step in range(1, num_steps):
        for j in range(len(quark_masses)):
            # Update temperature based on some model (placeholder)
            particle_temperatures[j, step] = particle_temperatures[j, step - 1] * 0.99  # Cooling down

            # Update speed based on temperature and mass
            particle_speeds[j, step] = update_speed(
                particle_speeds[j, step - 1], 
                particle_temperatures[j, step], 
                particle_masses[j]
            )

            # Check for collisions
            collision_detected, idx1, idx2 = check_collision(particle_speeds[:, step], collision_distance)
            if collision_detected:
                handle_collision(particle_speeds[:, step], idx1, idx2)

            # Tunneling effect (placeholder for actual physics)
            if np.random.rand() < tunneling_probability:
                tunneling_steps[j, step] = True
                # Modify mass or speed based on tunneling (placeholder)
                particle_masses[j] *= 1.1  # Increase mass as an example

        # Store mass evolution
        particle_masses_evolution[:, step] = particle_masses

    # Save the simulation data for this tunneling probability
    simulation_data = {
        "particle_speeds": particle_speeds.tolist(),  # Convert to list for JSON serialization
        "particle_temperatures": particle_temperatures.tolist(),  # Convert to list for JSON serialization
        "particle_masses_evolution": particle_masses_evolution.tolist(),  # Convert to list for JSON serialization
        "tunneling_steps": tunneling_steps.tolist(),  # Convert to list for JSON serialization
    }

    with open(os.path.join(data_dir, f"simulation_tunneling_{tunneling_probability:.2f}.json"), "w") as f:
        json.dump(simulation_data, f)

print("Simulation completed and data saved.")

