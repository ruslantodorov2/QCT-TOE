import numpy as np
import numba
from numba import cuda
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
import time
from scipy.stats import norm

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
t_simulation = t_planck * 1e3  # Shorter timescale for simulation

# Quark masses (in GeV) - used for initial mass values and comparison
quark_masses = {
    "up": 2.3e-3,
    "down": 4.8e-3,
    "charm": 1.28,
    "strange": 0.095,
    "top": 173.0,
    "bottom": 4.18,
}

# **ASK FOR NUMBER OF PARTICLES**
while True:
    try:
        num_particles = int(input("Enter the number of particles (integer): "))
        if num_particles <= 0:
            print("Please enter a positive integer.")
        else:
            break
    except ValueError:
        print("Invalid input. Please enter an integer.")

# **ASK FOR TUNNELING PROBABILITY**
while True:
    try:
        tunneling_probability = float(input("Enter the tunneling probability (float, 0-1): "))
        if 0 <= tunneling_probability <= 1:
            break
        else:
            print("Please enter a value between 0 and 1.")
    except ValueError:
        print("Invalid input. Please enter a float.")

# Generate additional particles based on user input
additional_particles = {
    f"new_quark_{i}": np.random.uniform(1e-3, 1e-1) for i in range(num_particles - len(quark_masses))
}

all_particles = {**quark_masses, **additional_particles}

# Conversion factor from GeV to J
GeV_to_J = 1.60217662e-10

# Simulation setup
num_steps = int(t_simulation / t_planck)

# CUDA kernel for simulation step
@cuda.jit
def simulation_step(particle_speeds, particle_temperatures, particle_masses, step, tunneling_probability):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw

    if i < num_particles:
        # Update speed
        particle_speeds[i] = update_speed(
            particle_speeds[i], particle_temperatures[i], particle_masses[i]
        )

        # Apply tunneling probability
        if np.random.rand() < tunneling_probability:
            particle_speeds[i] = particle_speed_initial

        # Update temperature
        particle_temperatures[i] = alpha * particle_speeds[i] ** 2

        # Simple collision detection (for demonstration; enhance as needed)
        for j in range(num_particles):
            if i != j:
                # Collision logic here (omitted for brevity)
                pass  # Placeholder for collision logic

# CPU function for updating speed (example; optimize as necessary)
def update_speed(current_speed, current_temperature, particle_mass):
    rel_momentum = relativistic_momentum(current_speed, particle_mass)
    return c * np.sqrt(
        max(1e-10, 1 - (rel_momentum / (rel_momentum + dark_energy_density)) ** 2)
    )

# CPU function for relativistic momentum (example; optimize as necessary)
def relativistic_momentum(particle_speed, particle_mass):
    if particle_speed >= c:
        return np.inf
    return (
        particle_mass
        * particle_speed
        / np.sqrt(max(1e-10, 1 - (particle_speed / c) ** 2))
    )

# Generate additional particles based on user input
additional_particles = {
    f"new_quark_{i}": np.random.uniform(1e-3, 1e-1) for i in range(num_particles - len(quark_masses))
}

# Ensure that the number of particles is at least the number of quark masses
if num_particles < len(quark_masses):
    print(f"Warning: Reducing the number of particles to {len(quark_masses)} to match quark masses.")
    num_particles = len(quark_masses)

all_particles = {**quark_masses, **additional_particles}

# Initialize particle properties
initial_speeds = np.full(num_particles, particle_speed_initial, dtype=np.float64)
initial_temperatures = np.full(num_particles, temperature_initial, dtype=np.float64)

# Create an array of masses based on the number of particles
initial_masses = np.zeros(num_particles, dtype=np.float64)

# Fill initial_masses with quark masses and additional particles
for i, (key, mass) in enumerate(all_particles.items()):
    if i < num_particles:
        initial_masses[i] = mass

# Main simulation loop
def main_simulation(tunneling_probability):
    # Memory allocation for simulation arrays
    d_particle_speeds = cuda.device_array(num_particles, dtype=np.float64)
    d_particle_temperatures = cuda.device_array(num_particles, dtype=np.float64)
    d_particle_masses = cuda.device_array(num_particles, dtype=np.float64)

    # Copy initial values to device
    d_particle_speeds.copy_to_device(initial_speeds)
    d_particle_temperatures.copy_to_device(initial_temperatures)
    d_particle_masses.copy_to_device(initial_masses)

    # Simulation loop
    for step in range(num_steps):
        simulation_step[1, num_particles](d_particle_speeds, d_particle_temperatures, d_particle_masses, step, tunneling_probability)

    # Copy results back to host
    h_particle_speeds = d_particle_speeds.copy_to_host()
    h_particle_temperatures = d_particle_temperatures.copy_to_host()
    h_particle_masses = d_particle_masses.copy_to_host()

    return h_particle_speeds, h_particle_temperatures, h_particle_masses

if __name__ == "__main__":
    start_time = time.time()
    with ProgressBar():
        task = delayed(main_simulation)(tunneling_probability)
        result = task.compute()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time} seconds")

    # Process and visualize results as needed
    particle_speeds, particle_temperatures, particle_masses = result
    print("Final Particle Speeds:", particle_speeds)
    print("Final Particle Temperatures:", particle_temperatures)
    print("Final Particle Masses:", particle_masses)
