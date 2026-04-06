import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("data/processed/clean_data.csv")

# Feature Engineering
df["risk_index"] = df["rainfall"] * df["population_density"]

low_th = df["risk_index"].quantile(0.33)
high_th = df["risk_index"].quantile(0.66)

def label_risk(value):
    if value < low_th:
        return 0
    elif value < high_th:
        return 1
    else:
        return 2

df["risk_label"] = df["risk_index"].apply(label_risk)

os.makedirs("outputs", exist_ok=True)

numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/heatmap.png")   # save BEFORE show
plt.show()

# Scatter plot
plt.figure()
plt.scatter(df["rainfall"], df["past_disasters"])
plt.xlabel("Rainfall")
plt.ylabel("Past Disasters")
plt.title("Rainfall vs Disasters")
plt.savefig("outputs/rainfall_vs_disasters.png")  # save BEFORE show
plt.show()

# Save dataset
df.to_csv("data/processed/final_data.csv", index=False)

