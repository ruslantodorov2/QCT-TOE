import numpy as np
import pandas as pd
import json
import os

# Constants
c = 299792458  # Speed of light in m/s
E_mc2 = c**2  # Mass-energy equivalence in J/kg
TSR = E_mc2 / (1.38e-23)  # Temperature to Speed Ratio in K/m/s
alpha = 1.1056e-52  # Proportional constant for TSR
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
tunneling_probabilities = np.arange(0.1, 4.1, 0.1)  # Exclude 1.0

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
        for k in range(j+1, len(particle_speeds)):
            if np.abs(particle_speeds[j] - particle_speeds[k]) < collision_distance:
                return True, j, k
    return False, -1, -1

# Simulate the Big Bang with Dark Energy, Dark Matter, Tunneling, and Relativistic Effects
for tunneling_probability in tunneling_probabilities:
    print(f"Simulating for tunneling probability: {tunneling_probability}")

    # Initialize arrays for simulation
    particle_speeds = np.zeros((len(quark_masses), num_steps))  # 2D array for speeds
    particle_temperatures = np.zeros(
        (len(quark_masses), num_steps)
    )  # 2D array for temperatures
    particle_masses_evolution = np.zeros(
        (len(quark_masses), num_steps)
    )  # 2D array for mass evolution
    tunneling_steps = np.zeros(
        (len(quark_masses), num_steps), dtype=bool
    )  # 2D array for tunneling steps

    # Create an array of masses for each quark
    particle_masses = np.array([mass * GeV_to_J for mass in quark_masses.values()])

    for j, (quark, mass) in enumerate(quark_masses.items()):
        particle_masses_evolution[j, 0] = particle_masses[j]  # Initialize mass

        for i in range(1, num_steps):
            particle_speeds[j, i] = update_speed(
                particle_speeds[j, i - 1],
                particle_temperatures[j, i - 1],
                particle_masses[j],
            )

            value = (
                1
                - (particle_speeds[j, i] / (TSR * temperature_initial))
                + dark_matter_density
            )

            if np.random.rand() < tunneling_probability:
                particle_speeds[j, i] = particle_speeds[j, 0]  # Tunneling effect
                tunneling_steps[j, i] = True  # Mark tunneling step

            if value < 0:
                value = 0

            particle_temperatures[j, i] = (
                alpha * particle_speeds[j, i] ** 2
            )  # Apply TSR equation

            # Update mass based on energy conversion
            speed_squared_diff = (
                particle_speeds[j, i] ** 2 - particle_speeds[j, i - 1] ** 2
            )

            # Avoid division by zero (if speed doesn't change, mass doesn't change)
            if speed_squared_diff == 0:
                particle_masses_evolution[j, i] = particle_masses_evolution[j, i - 1]
            else:
                # Calculate the change in relativistic energy
                energy_diff = relativistic_energy(
                    particle_speeds[j, i], particle_masses[j]
                ) - relativistic_energy(particle_speeds[j, i - 1], particle_masses[j])

                # Avoid NaN by checking if energy_diff is practically zero
                if abs(energy_diff) < 1e-15:  # Adjust the tolerance as needed
                    particle_masses_evolution[j, i] = particle_masses_evolution[
                        j, i - 1
                    ]
                else:
                    # Update mass based on energy difference
                    new_mass = (
                        particle_masses_evolution[j, i - 1] + energy_diff / c**2
                    )
                    if np.isfinite(new_mass):  # Check if the new mass is finite
                        particle_masses_evolution[j, i] = new_mass
                    else:
                        particle_masses_evolution[j, i] = particle_masses_evolution[
                            j, i - 1
                        ]

            # Apply expansion of the universe (redshift)
            particle_speeds[j, i] *= 1 - Hubble_constant_SI * t_planck

            # Apply expansion of the universe (cooling)
            particle_temperatures[j, i] *= 1 - Hubble_constant_SI * t_planck

            # Check for collisions and handle them
            collision, idx1, idx2 = check_collision(particle_speeds[:, i], collision_distance)
            if collision:
                # Simplified collision response: reverse speeds
                particle_speeds[idx1, i], particle_speeds[idx2, i] = -particle_speeds[idx1, i], -particle_speeds[idx2, i]

    # Print calculated masses at the end of the simulation
    print(
        f"Calculated masses at the end of the simulation (Tunneling Probability: {tunneling_probability}):"
    )
    for j, quark in enumerate(quark_masses.keys()):
        print(
            f"{quark}: {particle_masses_evolution[j, -1] / GeV_to_J:.4e} GeV"
        )

    # Save data to JSON file
    data_filename = os.path.join(
        data_dir, f"big_bang_simulation_data_{tunneling_probability:.2f}.json"
    )
    data = {
        "tunneling_probability": tunneling_probability,
        "particle_masses_evolution": particle_masses_evolution.tolist(),
        "particle_speeds": particle_speeds.tolist(),
        "particle_temperatures": particle_temperatures.tolist(),
        "tunneling_steps": tunneling_steps.tolist()
    }
    with open(data_filename, "w") as f:
        json.dump(data, f)

