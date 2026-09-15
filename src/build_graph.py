from collections import Counter
import pickle  
import networkx as nx
import pandas as pd
from pathlib import Path

# ---------- Configuration ----------
BASE = Path(r"C:\Users\pavan\demo_fyp\processed")

# ---------- Load processed files ----------
cd = pd.read_csv(BASE / "chem_disease.csv")
cg = pd.read_csv(BASE / "chem_gene.csv")
gd = pd.read_csv(BASE / "gene_disease.csv")
gp = pd.read_csv(BASE / "gene_pathway.csv")
dp = pd.read_csv(BASE / "disease_pathway.csv")

# ---------- Data Cleaning & Validation ----------
for df in [cd, cg, gd, gp, dp]:
    if "ChemicalID" in df.columns:
        df = df[df["ChemicalID"] != "ChemicalID"]
    if "GeneID" in df.columns:
        df = df[df["GeneID"] != "GeneID"]
    df.dropna(subset=[col for col in df.columns if "ID" in col], inplace=True)

# ---------- FIX: Map and Extract All Unique Disease Identity Strings ----------
# Creates a super-fast lookup set to intercept diseases masquerading as pathways
all_disease_names = set(dp['DiseaseName'].astype(str).str.strip().str.lower())

# ---------- Build Graph Structures ----------
G = nx.Graph()

# 1. Drug-Disease Relationships
for _, r in cd.iterrows():
    d = f"drug:{r.ChemicalID}"
    dis = f"disease:{r.DiseaseID}"
    G.add_node(d, type="drug", name=r.ChemicalName)
    G.add_node(dis, type="disease", name=r.DiseaseName)
    G.add_edge(d, dis, relation="treats")

# 2. Drug-Gene Relationships
for _, r in cg.iterrows():
    d = f"drug:{r.ChemicalID}"
    g = f"gene:{r.GeneID}"
    G.add_node(d, type="drug", name=r.ChemicalName)
    G.add_node(g, type="gene", name=r.GeneSymbol)
    G.add_edge(d, g, relation="interacts_with")

# 3. Gene-Disease Relationships
for _, r in gd.iterrows():
    g = f"gene:{r.GeneID}"
    dis = f"disease:{r.DiseaseID}"
    G.add_node(g, type="gene", name=r.GeneSymbol)
    G.add_node(dis, type="disease", name=r.DiseaseName)
    G.add_edge(g, dis, relation="associated_with")

# 4. Gene-Pathway Connections (Sampled for balance)
gp_sample = gp.sample(min(50000, len(gp)), random_state=42)
for _, r in gp_sample.iterrows():
    # EXPLICIT GUARDRAIL: Block pathway instantiation if label overlaps with disease registry
    if str(r.PathwayName).strip().lower() in all_disease_names:
        continue
        
    g = f"gene:{r.GeneID}"
    p = f"pathway:{r.PathwayID}"
    G.add_node(g, type="gene", name=r.GeneSymbol)
    G.add_node(p, type="pathway", name=r.PathwayName)
    G.add_edge(g, p, relation="participates_in")

# 5. Disease-Pathway Connections (Sampled for balance)
dp_sample = dp.sample(min(50000, len(dp)), random_state=42)
for _, r in dp_sample.iterrows():
    # EXPLICIT GUARDRAIL: Verify pathway node integrity
    if str(r.PathwayName).strip().lower() in all_disease_names:
        continue
        
    dis = f"disease:{r.DiseaseID}"
    p = f"pathway:{r.PathwayID}"
    G.add_node(dis, type="disease", name=r.DiseaseName)
    G.add_node(p, type="pathway", name=r.PathwayName)
    G.add_edge(dis, p, relation="involves_pathway")

# ---------- Post-Processing Graph Optimization ----------
# Extract the largest connected topology component
largest = max(nx.connected_components(G), key=len)
G = G.subgraph(largest).copy()

# ---------- Save the Re-Built Graph ----------
output_file = BASE / "biomedical_graph.gpickle"
with open(output_file, "wb") as f:
    pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

print("Biomedical Graph completely rebuilt successfully with type integrity fixes!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

types = Counter(nx.get_node_attributes(G, "type").values())
print("Cleaned Node type array breakdown:", types)
