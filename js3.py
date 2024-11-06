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
data_list = []
for f in data_files:
    data = load_json_data(f)
    data_list.append({
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
    })

df = pd.DataFrame(data_list)

# Scatter Plot: Tunneling Probability vs Various Particle Masses
plt.figure(figsize=(10, 8))
sns.scatterplot(x='tunneling_probability', y='particle_mass_up', data=df, label='Up Quark')
sns.scatterplot(x='tunneling_probability', y='particle_mass_down', data=df, label='Down Quark')
sns.scatterplot(x='tunneling_probability', y='particle_mass_charm', data=df, label='Charm Quark')
sns.scatterplot(x='tunneling_probability', y='particle_mass_strange', data=df, label='Strange Quark')
sns.scatterplot(x='tunneling_probability', y='particle_mass_top', data=df, label='Top Quark')
sns.scatterplot(x='tunneling_probability', y='particle_mass_bottom', data=df, label='Bottom Quark')
sns.scatterplot(x='tunneling_probability', y='particle_mass_electron', data=df, label='Electron')
sns.scatterplot(x='tunneling_probability', y='particle_mass_muon', data=df, label='Muon')
sns.scatterplot(x='tunneling_probability', y='particle_mass_tau', data=df, label='Tau')
sns.scatterplot(x='tunneling_probability', y='particle_mass_photon', data=df, label='Photon')
plt.title('Tunneling Probability vs Particle Masses')
plt.xlabel('Tunneling Probability')
plt.ylabel('Particle Mass (GeV)')
plt.legend()
plt.show()

# Heatmap: Correlation between Particle Masses
corr_matrix = df[['particle_mass_up', 'particle_mass_down', 'particle_mass_charm', 'particle_mass_strange', 'particle_mass_top', 'particle_mass_bottom', 'particle_mass_electron', 'particle_mass_muon', 'particle_mass_tau', 'particle_mass_photon']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation between Particle Masses')
plt.show()

# PCA
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
