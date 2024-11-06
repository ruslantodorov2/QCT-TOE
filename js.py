import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load JSON data
def load_json_data(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)

# Load multiple JSON files into a DataFrame
data_files = [
    '.\\Downloads\\big_bang_simulation_data\\big_bang_simulation_data_0.01.json',
    '.\\Downloads\\big_bang_simulation_data\\big_bang_simulation_data_0.02.json',
    '.\\Downloads\\big_bang_simulation_data\\big_bang_simulation_data_0.03.json',
    # Add more files as needed
]

data_list = [load_json_data(f) for f in data_files]

# Extract relevant data into a DataFrame
df = pd.DataFrame([
    {
        'tunneling_probability': data['tunneling_probability'],
        'particle_mass_up': data['particle_masses_evolution'][0][-1],
        'particle_mass_down': data['particle_masses_evolution'][1][-1],
        # Add more particles as needed
        'particle_speed': data['particle_speeds'][0][-1],  # Example particle speed
        'particle_temperature': data['particle_temperatures'][0][-1],  # Example particle temperature
    }
    for data in data_list
])

# Assume df is your DataFrame with simulation results
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
