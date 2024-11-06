import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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

# Scatter Plot: Tunneling Probability vs Up Quark Mass
plt.figure(figsize=(8, 6))
sns.scatterplot(x='tunneling_probability', y='particle_mass_up', data=df)
plt.title('Tunneling Probability vs Up Quark Mass')
plt.xlabel('Tunneling Probability')
plt.ylabel('Up Quark Mass (GeV)')
plt.show()

# Scatter Plot: Particle Temperature vs Speed
plt.figure(figsize=(8, 6))
sns.scatterplot(x='particle_temperature', y='particle_speed', data=df)
plt.title('Particle Temperature vs Speed')
plt.xlabel('Temperature (K)')
plt.ylabel('Speed (m/s)')
plt.show()

# Line Graph: Evolution of Up Quark Mass Over Time
time_steps = range(len(data_list[0]['particle_masses_evolution'][0]))
plt.figure(figsize=(10, 6))
for data in data_list:
    plt.plot(time_steps, data['particle_masses_evolution'][0], label=f"Tunneling Probability: {data['tunneling_probability']:.2f}")
plt.title('Evolution of Up Quark Mass Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Up Quark Mass (GeV)')
plt.legend()
plt.show()

# Dimensionality Reduction: PCA
features = ['particle_mass_up', 'particle_mass_down', 'particle_mass_charm', 'particle_mass_strange', 'particle_mass_top', 'particle_mass_bottom', 'particle_mass_electron', 'particle_mass_muon', 'particle_mass_tau', 'particle_mass_photon', 'particle_speed', 'particle_temperature']
X = df[features]
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# PCA Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue=df['tunneling_probability'], palette='viridis')
plt.title('PCA of Particle Properties')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Investigate correlations in detail
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Detailed Correlation Matrix')
plt.show()

# Identify highly correlated pairs
correlated_pairs = correlation_matrix.unstack().sort_values(kind="quicksort")
highly_correlated_pairs = correlated_pairs[(abs(correlated_pairs) > 0.8) & (abs(correlated_pairs) < 1)]

# Print highly correlated pairs
print("Highly Correlated Pairs:")
print(highly_correlated_pairs)

# Considering additional variables if available
df['particle_momentum'] = [...]  # Add momentum data if available
df['particle_energy'] = [...]    # Add energy data if available

# Re-run PCA with additional variables
features = ['particle_mass_up', 'particle_mass_down', 'particle_mass_charm', 'particle_mass_strange', 'particle_mass_top', 'particle_mass_bottom', 'particle_mass_electron', 'particle_mass_muon', 'particle_mass_tau', 'particle_mass_photon', 'particle_speed', 'particle_temperature', 'particle_momentum', 'particle_energy']
X = df[features]
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Plot updated PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue=df['tunneling_probability'], palette='viridis')
plt.title('PCA of Particle Properties with Additional Variables')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

