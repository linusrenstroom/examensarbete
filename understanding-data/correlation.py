import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../datasets/sidstyrning-februari.txt", sep=';', decimal=',', skiprows=[0, 2])

corr = df.corr().round(3)
print(corr.to_string())

# Eller som heatmap om du vill ha en figur istället för tabell


sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.tight_layout()
plt.savefig("korrelationsmatris.png", dpi=150)