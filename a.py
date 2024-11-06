# Initialize output dictionaries
outputs = {
    "tunneling_probabilities": tunneling_probabilities.tolist(),
    "speed_ranges": speed_ranges,
    "results": []
}

for speed_range in speed_ranges:
    for tunneling_probability in tunneling_probabilities:
        print(f"Simulating for tunneling probability: {tunneling_probability}, Speed range: {speed_range}")

        #... (rest of the simulation code remains the same until the WIMP mass calculation)

        wimp_mass = calculate_wimp_mass(dark_matter_density_GeV, calculate_redshift(particle_speeds[-1, -1]))
        print(f"Exact mass of the WIMP: {wimp_mass / GeV_to_J:.4e} GeV")

        outputs["results"].append({
            "tunneling_probability": tunneling_probability,
            "speed_range": speed_range,
            "wimp_mass_GeV": wimp_mass / GeV_to_J,
            "wimp_mass_J": wimp_mass,
            "particle_speeds": particle_speeds.tolist()
        })

# Save outputs to JSON file
with open("comprehensive_outputs.json", "w") as f:
    json.dump(outputs, f, indent=4)

# Generate correlation plots and statistics
for speed_range in speed_ranges:
    speed_range_results = [result for result in outputs["results"] if result["speed_range"] == speed_range]
    tunneling_probabilities_speed_range = [result["tunneling_probability"] for result in speed_range_results]
    wimp_masses_speed_range = [result["wimp_mass_GeV"] for result in speed_range_results]

    plt.figure(figsize=(8, 6))
    plt.scatter(tunneling_probabilities_speed_range, wimp_masses_speed_range)
    plt.xlabel("Tunneling Probability")
    plt.ylabel("Exact Mass of WIMP (GeV)")
    plt.title(f"Correlation between WIMP Mass and Tunneling Probability (Speed range: {speed_range})")
    plt.savefig(f"wimp_mass_correlation_speed_range_{speed_range}.png")
    plt.show()

    pearson_corr, _ = pearsonr(tunneling_probabilities_speed_range, wimp_masses_speed_range)
    print(f"Pearson correlation coefficient (Speed range: {speed_range}): {pearson_corr:.4f}")
		
