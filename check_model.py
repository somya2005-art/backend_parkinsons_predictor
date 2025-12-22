# test_italian_pd.py
import pandas as pd
import numpy as np

df = pd.read_csv("parkinsons.csv")

# Get a strong PD sample (high jitter, high shimmer, low HNR)
pd_samples = df[df["status"] == 1].copy()
pd_samples["pd_score"] = (
    pd_samples["MDVP:Jitter(%)"] * 100 + 
    pd_samples["MDVP:Shimmer"] * 100 - 
    pd_samples["HNR"]
)
worst_pd = pd_samples.nlargest(1, "pd_score")

print("🔴 Strongest PD sample in dataset:")
print(f"Name: {worst_pd['name'].values[0]}")
print(f"Jitter: {worst_pd['MDVP:Jitter(%)'].values[0]:.6f}")
print(f"Shimmer: {worst_pd['MDVP:Shimmer'].values[0]:.6f}")
print(f"HNR: {worst_pd['HNR'].values[0]:.2f}")

