import matplotlib.pyplot as plt
import numpy as np

step_labels = ['1', '5', '10', '25']
x = np.arange(len(step_labels))

data = {
    'ws20_anomali':  [0.712106, 0.712269, 0.709963, 0.710602],
    'ws50_anomali':  [0.876728, 0.875871, 0.880585, 0.868207],
    'ws100_anomali': [0.905744, 0.921172, 0.906724, 0.909273],
    'ws200_anomali': [None,     0.916065, 0.914860, 0.919212],
    'ws20_normal':   [0.901831, 0.901937, 0.900815, 0.902051],
    'ws50_normal':   [0.929998, 0.929852, 0.931928, 0.926406],
    'ws100_normal':  [0.916065, 0.927718, 0.916591, 0.918705],
    'ws200_normal':  [None,     0.859274, 0.857711, 0.863555],
}

colors = ['#E24B4A', '#378ADD', '#1D9E75', '#EF9F27']
ws_labels = ['ws = 20', 'ws = 50', 'ws = 100', 'ws = 200']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.patch.set_facecolor('white')

for ax in (ax1, ax2):
    ax.set_facecolor('white')
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, fontsize=11)
    ax.set_xlabel('Steglängd', fontsize=12)
    ax.set_ylim(0.65, 0.96)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(axis='y', color='#e0e0e0', linewidth=0.7)
    ax.grid(axis='x', color='#f0f0f0', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax1.set_ylabel('F1-score', fontsize=12)
ax1.set_title('F1 anomali', fontsize=13, fontweight='bold', pad=10)
ax2.set_title('F1 normal', fontsize=13, fontweight='bold', pad=10)

for i, ws in enumerate([20, 50, 100, 200]):
    y_anom = data[f'ws{ws}_anomali']
    y_norm = data[f'ws{ws}_normal']

    x_anom = [x[j] for j, v in enumerate(y_anom) if v is not None]
    v_anom = [v for v in y_anom if v is not None]
    x_norm = [x[j] for j, v in enumerate(y_norm) if v is not None]
    v_norm = [v for v in y_norm if v is not None]

    ax1.plot(x_anom, v_anom, color=colors[i], linewidth=2,
             marker='o', markersize=7, label=ws_labels[i])
    ax2.plot(x_norm, v_norm, color=colors[i], linewidth=2,
             marker='o', markersize=7, linestyle='--', label=ws_labels[i])

ax1.legend(title='Fönsterstorlek', fontsize=9, title_fontsize=9,
           framealpha=0.9, edgecolor='#cccccc', loc='lower left')
ax2.legend(title='Fönsterstorlek', fontsize=9, title_fontsize=9,
           framealpha=0.9, edgecolor='#cccccc', loc='lower left')

plt.tight_layout()
plt.savefig('f1_diagram.png', dpi=200, bbox_inches='tight')
plt.show()