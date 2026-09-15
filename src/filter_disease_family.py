"""Filter the processed CTD graph to a named disease family."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE = PROJECT_DIR / "processed"
OUT = PROJECT_DIR / "filtered"
TARGET_KEYWORDS = ("muscular dystrophy", "duchenne", "becker")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pattern = "|".join(TARGET_KEYWORDS)

    chem_disease = pd.read_csv(BASE / "chem_disease.csv")
    gene_disease = pd.read_csv(BASE / "gene_disease.csv")
    gene_pathway = pd.read_csv(BASE / "gene_pathway.csv")
    disease_pathway = pd.read_csv(BASE / "disease_pathway.csv")

    # Find IDs from every table that retains disease names.  Filtering by ID
    # alone cannot work because IDs such as MESH:D009136 do not contain words.
    target_disease_ids = set()
    for table in (chem_disease, gene_disease, disease_pathway):
        matches = table[table["DiseaseName"].str.contains(pattern, case=False, na=False)]
        target_disease_ids.update(matches["DiseaseID"].dropna().astype(str))

    target_chem_disease = chem_disease[chem_disease["DiseaseID"].astype(str).isin(target_disease_ids)].copy()
    target_gene_disease = gene_disease[gene_disease["DiseaseID"].astype(str).isin(target_disease_ids)].copy()
    target_disease_pathway = disease_pathway[
        disease_pathway["DiseaseID"].astype(str).isin(target_disease_ids)
    ].copy()

    selected_gene_ids = set(target_gene_disease["GeneID"].dropna().astype(str))
    target_gene_pathway = gene_pathway[gene_pathway["GeneID"].astype(str).isin(selected_gene_ids)].copy()

    # chem_gene is large: write matched human rows incrementally.
    output_chem_gene = OUT / "chem_gene.csv"
    wrote_rows = False
    for chunk in pd.read_csv(BASE / "chem_gene.csv", chunksize=100_000, low_memory=False):
        matches = chunk[chunk["GeneID"].astype(str).isin(selected_gene_ids)]
        if not matches.empty:
            matches.to_csv(output_chem_gene, mode="w" if not wrote_rows else "a", header=not wrote_rows, index=False)
            wrote_rows = True
    if not wrote_rows:
        pd.read_csv(BASE / "chem_gene.csv", nrows=0).to_csv(output_chem_gene, index=False)

    target_chem_disease.to_csv(OUT / "chem_disease.csv", index=False)
    target_gene_disease.to_csv(OUT / "gene_disease.csv", index=False)
    target_gene_pathway.to_csv(OUT / "gene_pathway.csv", index=False)
    target_disease_pathway.to_csv(OUT / "disease_pathway.csv", index=False)

    print("Filtered graph saved to:", OUT)
    print("Diseases:", len(target_disease_ids))
    print("Genes:", len(selected_gene_ids))
    print("Chemical-Disease:", len(target_chem_disease))
    print("Gene-Disease:", len(target_gene_disease))
    print("Gene-Pathway:", len(target_gene_pathway))
    print("Disease-Pathway:", len(target_disease_pathway))


if __name__ == "__main__":
    main()
