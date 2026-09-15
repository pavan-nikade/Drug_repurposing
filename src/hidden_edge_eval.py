import random
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec

# =========================================================================
# 1. Configuration
# =========================================================================
BASE = Path(r"C:\Users\pavan\demo_fyp\processed")

HIDE_RATIO = 0.20  
EMB_DIM = 64
WALK_LENGTH = 20
NUM_WALKS = 20

random.seed(42)
np.random.seed(42)

# =========================================================================
# 2. Load Therapeutic Data Matrices
# =========================================================================
dd = pd.read_csv(BASE / "chem_disease.csv")
edges = list(zip(dd["ChemicalID"], dd["DiseaseID"]))
print(f"Total therapeutic edges inside registry: {len(edges)}")

# Split Hidden (Test Set) and Visible (Graph Base Structure)
visible_edges, hidden_edges = train_test_split(
    edges, test_size=HIDE_RATIO, random_state=42
)
print(f"Visible structure base edges : {len(visible_edges)}")
print(f"Hidden validation target edges: {len(hidden_edges)}")

# Split Visible Edges to Isolate Training Data (Prevents Model Leakage)
train_pos_edges, val_pos_edges = train_test_split(
    visible_edges, test_size=0.25, random_state=42
)

# =========================================================================
# 3. Build Reduced Graph Structure
# =========================================================================
G = nx.Graph()

# Add visible edges to map fundamental base graph topology
for c, d in visible_edges:
    G.add_node(f"drug:{c}", type="drug")
    G.add_node(f"disease:{d}", type="disease")
    G.add_edge(f"drug:{c}", f"disease:{d}")

# Incorporate chem-gene multi-modal relations
cg = pd.read_csv(BASE / "chem_gene.csv")
for _, r in cg.iterrows():
    G.add_node(f"drug:{r.ChemicalID}", type="drug")
    G.add_node(f"gene:{r.GeneSymbol}", type="gene")
    G.add_edge(f"drug:{r.ChemicalID}", f"gene:{r.GeneSymbol}")

# Incorporate gene-disease multi-modal relations
gd = pd.read_csv(BASE / "gene_disease.csv")
for _, r in gd.iterrows():
    G.add_node(f"gene:{r.GeneSymbol}", type="gene")
    G.add_node(f"disease:{r.DiseaseID}", type="disease")
    G.add_edge(f"gene:{r.GeneSymbol}", f"disease:{r.DiseaseID}")

print(f"Reduced graph compiled: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# =========================================================================
# 4. Random Walk Generation
# =========================================================================
def random_walk(start, length):
    walk = [start]
    current = start
    for _ in range(length - 1):
        nbrs = list(G.neighbors(current))
        if not nbrs:
            break
        current = random.choice(nbrs)
        walk.append(current)
    return walk

walks = []
nodes = list(G.nodes())
for _ in range(NUM_WALKS):
    random.shuffle(nodes)
    for n in nodes:
        walks.append(random_walk(n, WALK_LENGTH))
print(f"Generated {len(walks)} random walk sequence arrays.")

# =========================================================================
# 5. Train Node Embeddings (Node2Vec-style)
# =========================================================================
w2v = Word2Vec(
    sentences=walks,
    vector_size=EMB_DIM,
    window=5,
    min_count=1,
    sg=1,
    workers=4,
    epochs=5
)
wv = w2v.wv
print("Node embedding representation optimization complete.")

# =========================================================================
# 6. Prepare Stratified Unbiased Training Datasets
# =========================================================================
all_drugs = list(dd["ChemicalID"].unique())
all_diseases = list(dd["DiseaseID"].unique())
total_edges_set = set(edges)

# Generate Non-Overlapping Negative Samples for Training
train_neg = []
while len(train_neg) < len(train_pos_edges):
    c = random.choice(all_drugs)
    d = random.choice(all_diseases)
    if (c, d) not in total_edges_set:
        train_neg.append((c, d))

# Generate Non-Overlapping Negative Samples for Validation
val_neg = []
val_neg_set = set(train_neg)
while len(val_neg) < len(val_pos_edges):
    c = random.choice(all_drugs)
    d = random.choice(all_diseases)
    if (c, d) not in total_edges_set and (c, d) not in val_neg_set:
        val_neg.append((c, d))

def edge_vector(c, d):
    u = f"drug:{c}"
    v = f"disease:{d}"
    if u not in wv or v not in wv:
        return None
    return np.concatenate([wv[u], wv[v]])

# Build Isolated Training Matrix Subspaces
X_train, y_train = [], []
X_val, y_val = [], []

for c, d in train_pos_edges:
    vec = edge_vector(c, d)
    if vec is not None:
        X_train.append(vec)
        y_train.append(1)

for c, d in train_neg:
    vec = edge_vector(c, d)
    if vec is not None:
        X_train.append(vec)
        y_train.append(0)

for c, d in val_pos_edges:
    vec = edge_vector(c, d)
    if vec is not None:
        X_val.append(vec)
        y_val.append(1)

for c, d in val_neg:
    vec = edge_vector(c, d)
    if vec is not None:
        X_val.append(vec)
        y_val.append(0)

X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)

print(f"Train subspace samples     : {len(y_train)}")
print(f"Validation subspace samples: {len(y_val)}")

# =========================================================================
# 7. Train Classifier Architecture (CRITICAL: 'clf' is defined right here)
# =========================================================================
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Calculate generalization performance metrics on unseen val slice
val_probs = clf.predict_proba(X_val)[:, 1]
generalization_auc = roc_auc_score(y_val, val_probs)

# =========================================================================
# 8. Ultra-Fast Vectorized Matrix Ranking Evaluation
# =========================================================================
hidden_by_drug = {}
for c, d in hidden_edges:
    hidden_by_drug.setdefault(c, set()).add(d)

recall10 = []
recall50 = []
rrs = []

print("\nCompiling disease matrix for fast matrix evaluation...")
valid_diseases = [d for d in all_diseases if f"disease:{d}" in wv]
disease_matrix = np.array([wv[f"disease:{d}"] for d in valid_diseases], dtype=np.float32)

print("Running multi-core matrix inference across hidden links...")
for drug in hidden_by_drug.keys():
    drug_node = f"drug:{drug}"
    if drug_node not in wv:
        continue

    true_diseases = hidden_by_drug[drug]
    drug_vec = wv[drug_node]
    
    # Vectorized Matrix Concatenation: Stacks the drug with ALL disease rows instantly
    repeated_drug_matrix = np.repeat(drug_vec[np.newaxis, :], len(valid_diseases), axis=0)
    eval_matrix = np.concatenate([repeated_drug_matrix, disease_matrix], axis=1)

    # Multi-core prediction sweep
    probs = clf.predict_proba(eval_matrix)[:, 1]

    # Map scores back to diseases and sort
    scores = list(zip(valid_diseases, probs))
    scores.sort(key=lambda x: x[1], reverse=True)
    ranked = [d for d, _ in scores]

    # Calculate Information Retrieval Performance Metrics
    top10 = set(ranked[:10])
    top50 = set(ranked[:50])

    r10 = len(true_diseases & top10) / len(true_diseases)
    r50 = len(true_diseases & top50) / len(true_diseases)

    recall10.append(r10)
    recall50.append(r50)

    # Compute Mean Reciprocal Rank (MRR)
    rr = 0.0
    for rank, disease in enumerate(ranked, start=1):
        if disease in true_diseases:
            rr = 1.0 / rank
            break
    rrs.append(rr)

# =========================================================================
# 9. Finalized Analytical Summary
# =========================================================================
print("\n" + "=" * 60)
print("UNBIASED VECTORIZED HIDDEN-EDGE EVALUATION RESULTS")
print("=" * 60)
print(f"Generalization ROC-AUC : {generalization_auc:.4f}")
print(f"Mean Recall@10 Score   : {np.mean(recall10):.4f}")
print(f"Mean Recall@50 Score   : {np.mean(recall50):.4f}")
print(f"Mean MRR Performance   : {np.mean(rrs):.4f}")
print("=" * 60)
