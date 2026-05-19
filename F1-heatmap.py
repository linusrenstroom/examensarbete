import matplotlib.pyplot as plt
import numpy as np

step_sizes = [1, 5, 10, 25]
window_sizes = [20, 50, 100, 200]

anomali = np.array([
    [0.712106, 0.712269, 0.709963, 0.710602],
    [0.876728, 0.875871, 0.880585, 0.868207],
    [0.905744, 0.921172, 0.906724, 0.909273],
    [np.nan,   0.916065, 0.914860, 0.919212],
])

normal = np.array([
    [0.901831, 0.901937, 0.900815, 0.902051],
    [0.929998, 0.929852, 0.931928, 0.926406],
    [0.916065, 0.927718, 0.916591, 0.918705],
    [np.nan,   0.859274, 0.857711, 0.863555],
])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
fig.patch.set_facecolor('white')

def draw_heatmap(ax, data, title, cmap):
    masked = np.ma.masked_invalid(data)
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(step_sizes)))
    ax.set_xticklabels(step_sizes, fontsize=11)
    ax.set_yticks(range(len(window_sizes)))
    ax.set_yticklabels([f'ws = {w}' for w in window_sizes], fontsize=11)
    ax.set_xlabel('Steglängd', fontsize=12)
    ax.set_ylabel('Fönsterstorlek', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    mid = (vmin + vmax) / 2
    for i in range(len(window_sizes)):
        for j in range(len(step_sizes)):
            val = data[i, j]
            if not np.isnan(val):
                # Mörk text på ljusa celler, ljus text på mörka celler
                text_color = 'white' if val > mid else '#222222'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=9, color=text_color)

    plt.colorbar(im, ax=ax, format='%.3f', shrink=0.85)

draw_heatmap(ax1, anomali, 'F1 anomali', 'Blues')
draw_heatmap(ax2, normal,  'F1 normal',  'Greens')

plt.tight_layout()
plt.savefig('f1_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()