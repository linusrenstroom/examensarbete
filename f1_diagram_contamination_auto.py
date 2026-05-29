import matplotlib.pyplot as plt
import numpy as np

# Updated Data for Step Size 25
window_sizes = [20, 50, 100, 200]
x = np.arange(len(window_sizes))

# Data extracted from user values for Step Size 25
f1_anomali = [0.713521, 0.832831, 0.868901, 0.835840]
f1_normal = [0.912875, 0.919133, 0.898581, 0.780522]

# Colors matching the F1-diagram class style
# ws20: red, ws50: blue, ws100: green, ws200: orange
colors = ['#E24B4A', '#378ADD', '#1D9E75', '#EF9F27']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.patch.set_facecolor('white')

for ax in (ax1, ax2):
    ax.set_facecolor('white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'ws={ws}' for ws in window_sizes], fontsize=11)
    ax.set_xlabel('Fönsterstorlek (Step Size = 25)', fontsize=12)
    ax.set_ylim(0.65, 0.96)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(axis='y', color='#e0e0e0', linewidth=0.7)
    ax.grid(axis='x', color='#f0f0f0', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Axis 1: Anomaly
ax1.set_ylabel('F1-score', fontsize=12)
ax1.set_title('F1 anomali', fontsize=13, fontweight='bold', pad=10)

# Axis 2: Normal
ax2.set_title('F1 normal', fontsize=13, fontweight='bold', pad=10)

# Plotting the data
# To match the style, we plot one line per class, but color the points/segments 
# Or we can plot them as individual colored bars/points as requested for "exactly like the class"
# However, since the class plots vs Step Size, and here we have 1 Step Size vs multiple Windows,
# we will plot a single line connecting the windows to show the trend.

ax1.plot(x, f1_anomali, color='#555555', linewidth=1.5, linestyle=':', alpha=0.5, zorder=1)
ax2.plot(x, f1_normal, color='#555555', linewidth=1.5, linestyle=':', alpha=0.5, zorder=1)

for i in range(len(window_sizes)):
    ax1.plot(x[i], f1_anomali[i], marker='o', markersize=9, color=colors[i], 
             label=f'ws = {window_sizes[i]}', linestyle='None', zorder=3)
    ax2.plot(x[i], f1_normal[i], marker='s', markersize=9, color=colors[i], 
             label=f'ws = {window_sizes[i]}', linestyle='None', zorder=3)

# Legends removed as requested


plt.tight_layout()
plt.savefig('f1_diagram_step25.png', dpi=200, bbox_inches='tight')
print("Plot saved to f1_diagram_step25.png")
plt.show()
