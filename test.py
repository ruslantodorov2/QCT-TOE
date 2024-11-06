import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cupy as cp
from tqdm import tqdm
import plotly.graph_objects as go
import streamlit as st

x = st.slider('Select a value')
st.write(x, 'squared is', x * x)

# Define the twelfth root of two
Q = 2 ** (1/12)

# Define the wave function of the universe with variants (using CuPy)
def wave_function_cupy(x, t, scale=1.0, phase_shift=0.0):
    denominator = 2 * (t**2 + 1e-10)  # Add a small value to avoid division by zero
    return scale * Q * cp.exp(-x**2 / denominator) * cp.exp(-1j * (t + phase_shift))

# Simulation parameters
x = np.linspace(-10, 10, 100)
t = np.linspace(0, 10, 100)
X, T = np.meshgrid(x, t)

# Convert numpy arrays to CuPy arrays
X_cupy = cp.asarray(X)
T_cupy = cp.asarray(T)

# Variants parameters
scales = [0.5, 1.0, 1.5]  # Different scaling factors
phase_shifts = [0, np.pi/4, np.pi/2]  # Different phase shifts

# Initialize a 3D array to store results
wave_functions_3d = np.zeros((len(scales), len(phase_shifts), len(x), len(t)), dtype=complex)

# Simulate with variants and store results in the 3D array
for i, scale in enumerate(scales):
    for j, phase_shift in enumerate(phase_shifts):
        wave_functions_3d[i, j, :, :] = cp.asnumpy(wave_function_cupy(X_cupy, T_cupy, scale, phase_shift))

# --- Plotly Interactive Visualization ---

# Create the figure
fig = go.Figure(data=[
    go.Surface(x=x, y=t, z=np.abs(wave_functions_3d[0, 0, :, :])**2)
])

fig.update_layout(
    title="Wave Function of the Universe",
    scene=dict(
        xaxis_title="x",
        yaxis_title="t",
        zaxis_title="|ψ(x,t)|^2"
    ),
)

# Add Scale Slider
fig.update_layout(
    sliders=[
        dict(
            active=True,
            currentvalue=dict(
                prefix="Scale: ",
                font=dict(size=12)
            ),
            steps=[
                dict(
                    method="update",
                    args=[
                        {"z": [np.abs(wave_functions_3d[i, 0, :, :])**2]}  # Update z data
                    ],
                    label=f"Scale: {scales[i]:.2f}"  # Label for step values
                ) for i in range(len(scales))
            ],
            pad=dict(t=50),
            len=0.9,  # Length of the slider
            x=0.1,    # X position of the slider
            y=0.1,    # Y position of the slider
        ),
        dict(
            active=True,
            currentvalue=dict(
                prefix="Phase Shift: ",
                font=dict(size=12)
            ),
            steps=[
                dict(
                    method="update",
                    args=[
                        {"z": [np.abs(wave_functions_3d[0, j, :, :])**2]}  # Update z data
                    ],
                    label=f"Phase Shift: {phase_shifts[j]:.2f}"  # Label for step values
                ) for j in range(len(phase_shifts))
            ],
            pad=dict(t=50),
            len=0.9,  # Length of the slider
            x=0.1,    # X position of the slider
            y=0.3,    # Y position of the slider
        )
    ]
)

fig.show()

# --- End of Plotly ---

# --- Matplotlib Animation ---

# Animation
fig, ax = plt.subplots()
im = ax.imshow(np.abs(wave_functions_3d[0, 0, :, :]) ** 2, extent=[-10, 10, 0, 10], aspect='auto', cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title('Wave Function of the Universe')
cbar = fig.colorbar(im, ax=ax, label='|ψ(x,t)|^2')

def update(frame):
    i, j = divmod(frame, len(phase_shifts))  # Get the index for the 3D array
    im.set_array(np.abs(wave_functions_3d[i, j, :, :]) ** 2)  # Update with the correct frame
    ax.set_title(f'Wave Function at Scale: {scales[i]}, Phase Shift: {phase_shifts[j]:.2f}')
    return im,

ani = FuncAnimation(fig, update, frames=len(scales) * len(phase_shifts), blit=True)
ani.save('wave_function_animation.gif', writer='pillow')
plt.show()
