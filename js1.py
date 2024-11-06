import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load JSON data
def load_json_data(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)

# Directory containing the JSON files
data_dir = '.\\Documents\\big_bang_simulation_data\\'

# List all JSON files in the directory
data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]

# Load multiple JSON files into a DataFrame
data_list = [load_json_data(f) for f in data_files]

# Extract relevant data into a DataFrame
df = pd.DataFrame([
    {
        'tunneling_probability': data['tunneling_probability'],
        'particle_mass_up': data['particle_masses_evolution'][0][-1],
        'particle_mass_down': data['particle_masses_evolution'][1][-1],
        'particle_mass_charm': data['particle_masses_evolution'][2][-1],
        'particle_mass_strange': data['particle_masses_evolution'][3][-1],
        'particle_mass_top': data['particle_masses_evolution'][4][-1],
        'particle_mass_bottom': data['particle_masses_evolution'][5][-1],
        'particle_mass_electron': data['particle_masses_evolution'][6][-1],
        'particle_mass_muon': data['particle_masses_evolution'][7][-1],
        'particle_mass_tau': data['particle_masses_evolution'][8][-1],
        'particle_mass_photon': data['particle_masses_evolution'][9][-1],
        'particle_speed': data['particle_speeds'][0][-1],
        'particle_temperature': data['particle_temperatures'][0][-1],
    }
    for data in data_list
])

# Compute correlations
correlation_matrix = df.corr()

# Adjust figure size for better visibility
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

