import pandas as pd
import random
from pathlib import Path

BASE = Path(r"C:\\Users\\pavan\\demo_fyp\\processed")

cd = pd.read_csv(BASE / "chem_disease.csv")

# Positive pairs
pos = cd[["ChemicalID", "DiseaseID"]].drop_duplicates()
pos["label"] = 1

# Candidate nodes
drugs = pos["ChemicalID"].unique().tolist()
diseases = pos["DiseaseID"].unique().tolist()

positive_set = set(zip(pos.ChemicalID, pos.DiseaseID))

# Negative sampling
neg = []
while len(neg) < len(pos):
    d = random.choice(drugs)
    dis = random.choice(diseases)
    if (d, dis) not in positive_set:
        neg.append((d, dis))

neg = pd.DataFrame(neg, columns=["ChemicalID", "DiseaseID"])
neg["label"] = 0

# Combine
samples = pd.concat([pos, neg], ignore_index=True)
samples = samples.sample(frac=1, random_state=42).reset_index(drop=True)

samples.to_csv(BASE / "link_samples.csv", index=False)

print("Samples created:", len(samples))
print(samples["label"].value_counts())