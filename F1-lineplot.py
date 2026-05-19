import matplotlib.pyplot as plt

# Data from the user
window_sizes = [20, 50, 100, 200]
f1_normal = [0.912875, 0.919133, 0.898581, 0.780522]
f1_anomaly = [0.713521, 0.832831, 0.868901, 0.835840]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot F1 Normal
plt.plot(window_sizes, f1_normal, marker='o', linestyle='-', linewidth=2, label='F1-score Normal', color='#2ecc71')

# Plot F1 Anomaly
plt.plot(window_sizes, f1_anomaly, marker='s', linestyle='-', linewidth=2, label='F1-score Anomaly', color='#e74c3c')

# Add labels and title
plt.xlabel('Window Size (Number of Samples)', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.title('F1-Score Performance vs. Window Size', fontsize=14, fontweight='bold')

# Set X-axis ticks to match the window sizes
plt.xticks(window_sizes)

# Add grid for readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(loc='best')

# Add annotations to the points for clarity
for i, txt in enumerate(f1_normal):
    plt.annotate(f"{txt:.3f}", (window_sizes[i], f1_normal[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

for i, txt in enumerate(f1_anomaly):
    plt.annotate(f"{txt:.3f}", (window_sizes[i], f1_anomaly[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)

# Save the plot
output_path = 'f1_window_performance.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to {output_path}")

# Close to free up memory
plt.close()
