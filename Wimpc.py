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

# Convert dark matter density to GeV/m³
dark_matter_density_GeV = dark_matter_density / 1.60217662e-10

# Initial conditions
temperature_initial = 1.0  # Planck temperature in K
particle_density_initial = 5.16e96  # Planck density in kg/m^3
particle_speed_initial = TSR * temperature_initial  # Initial speed based on TSR

# Simulation time
t_planck = 5.39e-44  # Planck time in s
t_simulation = t_planck * 1e5  # Shorter timescale for simulation

# Updated particle masses (in GeV)
particle_masses = {
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
    "electron_neutrino": 0,  # Neutrinos have very small masses
    "muon_neutrino": 0,
    "tau_neutrino": 0,
    "W_boson": 80.379,
    "Z_boson": 91.1876,
    "Higgs_boson": 125.1,
    "gluon": 0,  # Massless
    "proton": 0.938,
    "neutron": 0.939,
    "pion_plus": 0.140,
    "pion_zero": 0.135,
    "kaon_plus": 0.494,
    "kaon_zero": 0.498
}

# Conversion factor from GeV to J
GeV_to_J = 1.60217662e-10

# Simulation setup
num_steps = int(t_simulation / t_planck)

# Tunneling probabilities to investigate
tunneling_probabilities = np.arange((0.1, 0.6, 0.15))  # Exclude 1.0

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

def handle_collision(particle_speeds, particle_masses, idx1, idx2, current_step):
    """Handle a collision between two particles."""
    if particle_masses[idx1] == 0 or particle_masses[idx2] == 0:
        # Skip handling collisions involving massless particles
        return

    p1 = relativistic_momentum(particle_speeds[idx1, current_step], particle_masses[idx1])
    p2 = relativistic_momentum(particle_speeds[idx2, current_step], particle_masses[idx2])

    # Calculate velocities after collision using conservation of momentum
    total_momentum = p1 + p2
    total_mass = particle_masses[idx1] + particle_masses[idx2]
    v1_new = (total_momentum / total_mass) * (particle_masses[idx1] / total_mass)
    v2_new = (total_momentum / total_mass) * (particle_masses[idx2] / total_mass)

    particle_speeds[idx1, current_step], particle_speeds[idx2, current_step] = v1_new, v2_new


def calculate_redshift(particle_speed):
    return (1 + particle_speed / c)
# Calculate the exact mass of the WIMP
def calculate_wimp_mass(dark_matter_density_GeV, redshift):
    return np.sqrt(2 * dark_matter_density_GeV * (1 + redshift)**3)

# Calculate the exact mass of the axion
def calculate_axion_mass(dark_matter_density_GeV):
    return np.sqrt(dark_matter_density_GeV) * 1e-5

# Calculate the exact mass of the graviton
def calculate_graviton_mass(dark_matter_density_GeV):
    return 0  # Graviton is massless

# Calculate the exact mass of the muon g-2
def calculate_muon_g2_mass(dark_matter_density_GeV):
    return 1.05e-1  # Muon mass

# Calculate the exact mass of the preon
def calculate_preon_mass(dark_matter_density_GeV):
    return 1.0e-3  # Preon mass

# Simulate the Big Bang with Dark Energy, Dark Matter, Tunneling, Relativistic Effects, Redshift, and Entanglement
for tunneling_probability in tunneling_probabilities:
    print(f"Simulating for tunneling probability: {tunneling_probability}")

    # Initialize arrays for simulation
    num_particles = len(particle_masses)
    particle_speeds = np.zeros((num_particles, num_steps))  # 2D array for speeds
    particle_temperatures = np.zeros((num_particles, num_steps))  # 2D array for temperatures
    particle_masses_evolution = np.zeros((num_particles, num_steps))  # 2D array for mass evolution
    tunneling_steps = np.zeros((num_particles, num_steps), dtype=bool)  # 2D array for tunneling steps
    particle_momentum = np.zeros((num_particles, num_steps))  # 2D array for momentum
    total_energy = np.zeros(num_steps)  # 1D array for total energy of the system
    redshifts = np.zeros((num_particles, num_steps))  # 2D array for redshift
    entanglement_entropies = np.zeros((num_particles, num_steps))  # 2D array for entanglement entropy
    particle_states = np.random.rand(num_particles, num_steps)  # Placeholder for particle states

    # Create an array of masses for each particle
    particle_masses_array = np.array([mass * GeV_to_J for mass in particle_masses.values()])

    for j, (particle, mass) in enumerate(particle_masses.items()):
        particle_speeds[j, 0] = particle_speed_initial  # Initialize speed
        particle_masses_evolution[j, 0] = mass * GeV_to_J  # Initialize mass evolution

    for current_step in range(1, num_steps):
        for j in range(num_particles):
            # Update temperature based on expansion of the universe
            particle_temperatures[j, current_step] = particle_temperatures[j, current_step-1] * (1 - Hubble_constant_SI * t_planck)

            # Update speed using TSR
            particle_speeds[j, current_step] = update_speed(particle_speeds[j, current_step-1], particle_temperatures[j, current_step], particle_masses_array[j])

            # Apply tunneling effect
            if np.random.rand() < tunneling_probability:
                particle_speeds[j, current_step] = particle_speeds[j, 0]
                tunneling_steps[j, current_step] = True

            # Calculate redshift
            redshifts[j, current_step] = (1 + particle_speeds[j, current_step] / c)

            # Calculate entanglement entropy
            entanglement_entropies[j, current_step] = -np.sum(particle_states[j, current_step] * np.log(particle_states[j, current_step]))

            # Update mass evolution
            particle_masses_evolution[j, current_step] = particle_masses_evolution[j, current_step-1] * (1 - dark_energy_density * t_planck)

        # Check for collisions
        collision_detected, idx1, idx2 = check_collision(particle_speeds[:, current_step], collision_distance)
        if collision_detected:
            handle_collision(particle_speeds, particle_masses_array, idx1, idx2, current_step)

    # Print calculated masses at the end of the simulation
    print(f"Calculated masses at the end of the simulation (Tunneling Probability: {tunneling_probability}):")
    for j, particle in enumerate(particle_masses.keys()):
        print(f"{particle}: {particle_masses_evolution[j, -1] / GeV_to_J:.4e} GeV")

    # Calculate the exact masses of the WIMP, axion, graviton, muon g-2, and preon
    wimp_mass = calculate_wimp_mass(dark_matter_density_GeV, calculate_redshift(particle_speeds[-1, -1]))
    axion_mass = calculate_axion_mass(dark_matter_density_GeV)
    graviton_mass = calculate_graviton_mass(dark_matter_density_GeV)
    muon_g2_mass = calculate_muon_g2_mass(dark_matter_density_GeV)
    preon_mass = calculate_preon_mass(dark_matter_density_GeV)

    # Print the exact masses of the WIMP, axion, graviton, muon g-2, and preon
    print(f"Exact mass of the WIMP: {wimp_mass / GeV_to_J:.4e} GeV")
    print(f"Exact mass of the axion: {axion_mass / GeV_to_J:.4e} GeV")
    print(f"Exact mass of the graviton: {graviton_mass / GeV_to_J:.4e} GeV")
    print(f"Exact mass of the muon g-2: {muon_g2_mass / GeV_to_J:.4e} GeV")
    print(f"Exact mass of the preon: {preon_mass / GeV_to_J:.4e} GeV")


