import pandas as pd

df = pd.read_csv("datasets/sidstyrning-februari.txt", sep=';', decimal=',', skiprows=[0, 2])
desc = df.describe().T[["mean", "std", "min", "50%", "max"]]
desc.columns = ["Medelvärde", "Standardavvikelse", "Min", "Median", "Max"]
desc["IQR"] = df.quantile(0.75) - df.quantile(0.25)
desc = desc[["Medelvärde", "Standardavvikelse", "Min", "Median", "IQR", "Max"]]
desc = desc.round(3)

print(desc.to_string())