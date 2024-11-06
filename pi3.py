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
Hubble_constant_SI = Hubble_constant * 1000 / 3.086e22  # Convert to SI units (s^-1)

# Initial conditions
temperature_initial = 1.0  # Planck temperature in K
particle_density_initial = 5.16e96  # Planck density in kg/m^3
particle_speed_initial = TSR * temperature_initial  # Initial speed based on TSR

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
tunneling_probabilities = np.arange(0.00000000001, 0.00000000010, 0.00000000001 )  # Exclude 1.0

# Create a directory to store the data
data_dir = "big_bang_simulation_data"
os.makedirs(data_dir, exist_ok=True)

# Functions to incorporate relativistic effects and collisions
def relativistic_energy(particle_speed, particle_mass):
    epsilon = 1e-15  # A small value to avoid division by zero
    return particle_mass * c**2 / np.sqrt(max(1e-15, 1 - (particle_speed / c) ** 2 + epsilon))

def relativistic_momentum(particle_speed, particle_mass):
    epsilon = 1e-15  # A small value to avoid division by zero
    return particle_mass * particle_speed / np.sqrt(max(1e-15, 1 - (particle_speed / c) ** 2 + epsilon))

def update_speed(current_speed, current_temperature, particle_mass):
    """Update the speed of a particle based on temperature and mass."""
    return TSR * current_temperature  # Update speed using TSR

def check_collision(particle_speeds, collision_distance):
    epsilon = 1e-15  # A small value to avoid invalid subtraction
    for j in range(len(particle_speeds)):
        for k in range(j+1, len(particle_speeds)):
            if np.abs(particle_speeds[j] - particle_speeds[k]) < collision_distance + epsilon:
                return True, j, k
    return False, -1, -1

def handle_collision(particle_speeds, particle_masses, idx1, idx2):
    """Handle a collision between two particles."""
    p1 = relativistic_momentum(particle_speeds[idx1], particle_masses[idx1])
    p2 = relativistic_momentum(particle_speeds[idx2], particle_masses[idx2])
    
    # Calculate velocities after collision using conservation of momentum
    total_momentum = p1 + p2
    total_mass = particle_masses[idx1] + particle_masses[idx2]
    v1_new = (total_momentum / total_mass) * (particle_masses[idx1] / total_mass)
    v2_new = (total_momentum / total_mass) * (particle_masses[idx2] / total_mass)
    
    particle_speeds[idx1], particle_speeds[idx2] = v1_new, v2_new

#...

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
        particle_masses_evolution[j, 0] = particle_masses[j]  # Initialize mass
        particle_speeds[j, 0] = particle_speed_initial  # Initialize speed
        particle_temperatures[j, 0] = temperature_initial  # Initialize temperature
        
        for i in range(1, num_steps):
            # Update temperature based on expansion of the universe
            particle_temperatures[j, i] = particle_temperatures[j, i-1] * (1 - Hubble_constant_SI * t_planck)
            
            # Update speed using TSR
            particle_speeds[j, i] = update_speed(particle_speeds[j, i-1], particle_temperatures[j, i], particle_masses[j])
            
            # Apply tunneling effect
            if np.random.rand() < tunneling_probability:
                particle_speeds[j, i] = particle_speeds[j, 0]
                tunneling_steps[j, i] = True
            
	    # Calculate entropy using von Neumann entropy formula
            if particle_masses[j] == 0:
    	        entropy = 0
            else:
    	        entropy = -particle_masses[j] * np.log1p(particle_masses[j])
            
            # Update mass based on entropy
            particle_masses_evolution[j, i] = particle_masses_evolution[j, i-1] + entropy / c**2
            
            # Check for entanglement
            if particle_speeds[j, i] == c and particle_temperatures[j, i] == 0:
                print(f"Entanglement detected for particle {quark} at step {i}")
                
    # Print calculated masses at the end of the simulation
    print(f"Calculated masses at the end of the simulation using the von Neumann entropy (Tunneling Probability: {tunneling_probability}):")
    for j, quark in enumerate(quark_masses.keys()):
        print(f"{quark}: {particle_masses_evolution[j, -1] / GeV_to_J:.4e} GeV")

    # Save data to JSON file
    data_filename = os.path.join(data_dir, f"big_bang_simulation_data_{tunneling_probability:.2f}.json")
    data = {
        "tunneling_probability": tunneling_probability,
        "particle_masses_evolution": particle_masses_evolution.tolist(),
        "particle_speeds": particle_speeds.tolist(),
        "particle_temperatures": particle_temperatures.tolist(),
        "tunneling_steps": tunneling_steps.tolist()
    }
    with open(data_filename, "w") as f:
        json.dump(data, f)
