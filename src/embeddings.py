import pickle  # FIXED: Imported native pickle engine to replace nx.read_gpickle
from pathlib import Path
import networkx as nx
from node2vec import Node2Vec

# ---------- Configuration ----------
BASE = Path(r"C:\Users\pavan\demo_fyp\processed")

# ---------- Load graph Safely ----------
# FIXED: Replaced nx.read_gpickle with standard open/pickle stream
with open(BASE / "biomedical_graph.gpickle", "rb") as f:
    G = pickle.load(f)

print("Loaded graph successfully!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# ---------- Train Node2Vec ----------
print("Pre-computing random walks (this may take a few minutes)...")
node2vec = Node2Vec(
    G,
    dimensions=64,
    walk_length=20,
    num_walks=50,
    workers=1  # FIXED: Set to 1 to guarantee stability on Windows without C-compilers
)

print("Training embedding vectors...")
model = node2vec.fit(
    window=10,
    min_count=1,
    batch_words=128
)

# ---------- Save embeddings ----------
# FIXED: Converted path to string format to keep gensim's file writer safe
output_path = BASE / "node_embeddings.txt"
model.wv.save_word2vec_format(str(output_path))

print(f"Embeddings saved successfully to: {output_path}")
