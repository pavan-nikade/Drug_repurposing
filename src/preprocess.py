import pandas as pd
from pathlib import Path

RAW = Path(r"C:\\Users\\pavan\\demo_fyp\\row_data")
OUT = Path(r"C:\\Users\\pavan\\demo_fyp\\processed")
OUT.mkdir(exist_ok=True)

# ---------- 1. Chemical-Disease ----------
cd = pd.read_csv(
     RAW / "CTD_curated_chemicals_diseases.csv",
    comment="#",
    header=None,
    low_memory=False
)

cd.columns = [
    "ChemicalName",
    "ChemicalID",
    "CasRN",
    "DiseaseName",
    "DiseaseID",
    "DirectEvidence",
    "PubMedIDs"
]

# Keep only therapeutic evidence
cd = cd[cd["DirectEvidence"].notna()]
cd = cd[cd["DirectEvidence"].str.contains("therapeutic", case=False, na=False)]

# ---------- 2. Chemical-Gene ----------
cg = pd.read_csv(
    RAW / 'CTD_chem_gene_ixns.csv',
    comment='#',
    header=None
)

print(cg.shape)
print(cg.head(2))

cg.columns = [
    'ChemicalName',      # 0
    'ChemicalID',        # 1
    'CasRN',             # 2
    'GeneSymbol',        # 3
    'GeneID',            # 4
    'GeneForms',         # 5
    'Organism',          # 6
    'OrganismID',        # 7
    'Interaction',       # 8
    'InteractionActions',# 9
    'PubMedIDs'          # 10
]

print(cg[['Organism','OrganismID']].head())

# Keep only human genes
cg = cg[cg['OrganismID'].astype(str) == '9606']

# ---------- 3. Gene-Disease ----------
gd = pd.read_csv(
    RAW / "CTD_curated_genes_diseases.csv",
    comment="#",
    header=None,
    low_memory=False
)

gd.columns = [
    "GeneSymbol","GeneID",
    "DiseaseName","DiseaseID",
    "DirectEvidence","InferenceScore",
    "PubMedIDs"
]

# Keep curated direct evidence
gd = gd[gd["DirectEvidence"].notna()]

# ---------- 4. Gene-Pathway ----------
gp = pd.read_csv(
    RAW / 'CTD_genes_pathways.csv',
    comment='#',
    header=None,
    low_memory=False
)

print(gp.shape)

# If it prints (rows, 4), use:

gp.columns = [
    'GeneSymbol',
    'GeneID',
    'PathwayName',
    'PathwayID'
]

# Disease–Pathway

dp = pd.read_csv(
    RAW / 'CTD_diseases_pathways.csv',
    comment='#',
    header=None,
    low_memory=False
)

print(dp.shape)

# If it prints (rows, 4), use:

dp.columns = [
    'DiseaseName',
    'DiseaseID',
    'PathwayName',
    'PathwayID',
    'InferenceGeneSymbol'
]

# ---------- Save ----------
cd.to_csv(OUT / "chem_disease.csv", index=False)
cg.to_csv(OUT / "chem_gene.csv", index=False)
gd.to_csv(OUT / "gene_disease.csv", index=False)
gp.to_csv(OUT / "gene_pathway.csv", index=False)
dp.to_csv(OUT / "disease_pathway.csv", index=False)

print("Saved processed CTD files to:", OUT)
print("chem_disease (therapeutic):", len(cd))
print("chem_gene (human):", len(cg))
print("gene_disease (curated):", len(gd))
print("gene_pathway:", len(gp))
print("disease_pathway:", len(dp))
