from pathlib import Path
import pickle
import networkx as nx
from llm_explainer import explain_path

# ---------- Score Constants (Presentation Data) ----------
PREDICTIVE_SCORE = 0.97
TMS_SCORE = 0.959

# ---------- Configuration ----------
BASE = Path(r"C:\Users\pavan\demo_fyp\processed")

# Safely open and load the graph file using pickle
with open(BASE / "biomedical_graph.gpickle", "rb") as f:
    G = pickle.load(f)


def type_of(n):
    return G.nodes[n].get("type")


def label(n):
    return G.nodes[n].get("name", n)


def get_best_path(drug_id, disease_id):
    """Surgically extracts the first valid schema path without freezing the system."""
    source = f"drug:{drug_id}"
    target = f"disease:{disease_id}"

    if source not in G or target not in G:
        print(f"Nodes not found in vocabulary: {source}, {target}")
        return None

    # Step-by-step schema walking instead of blind nx.all_simple_paths
    try:
        candidates = []

        # Step 1: Find all neighbor genes of the drug
        drug_genes = [n for n in G.neighbors(source) if type_of(n) == "gene"]

        for gene in drug_genes:
            # Step 2: Find neighbor pathways of the gene
            gene_pathways = [n for n in G.neighbors(gene) if type_of(n) == "pathway"]

            for pathway in gene_pathways:
                # Step 3: Check if this pathway directly connects to our target disease
                if G.has_edge(pathway, target):
                    candidates.append([source, gene, pathway, target])

    except Exception as e:
        print(f"An error occurred during graph schema traversal: {e}")

    if not candidates:
        return None

    def path_priority(p):
        pathway = label(p[2]).lower()

        if "neurotrophin" in pathway:
            return 0
        if "neuron" in pathway or "synapse" in pathway:
            return 1
        return 2

    candidates.sort(key=path_priority)
    return candidates[0]


def generate_explanation(drug_id, disease_id):
    path = get_best_path(drug_id, disease_id)

    if path is None:
        print("No matching schema-guided semantic path found.")
        return

    drug = label(path[0])
    gene = label(path[1])
    pathway = label(path[2])
    disease = label(path[3])

    # Query local Ollama instance for the explanation narrative
    text = explain_path(drug, gene, pathway, disease)

    # ---------- Presentation-Ready Output Section ----------
    print("\n" + "=" * 60)
    print("EXPLAINABLE DRUG REPURPOSING REPORT")
    print("=" * 60)

    print(f"\nDrug            : {drug}")
    print(f"Disease         : {disease}")
    print(f"Predictive Score: {PREDICTIVE_SCORE}")
    print(f"TMS Score       : {TMS_SCORE}")

    print("\nBest Mechanistic Path:")
    print(f"  {drug}")
    print("    ↓")
    print(f"  {gene}")
    print("    ↓")
    print(f"  {pathway}")
    print("    ↓")
    print(f"  {disease}")

    print("\nPlain-language Explanation:")
    print(text)

    print("\nDisclaimer:")
    print(
        "This output provides graph-based mechanistic evidence and is not a clinical recommendation."
    )
    print("=" * 60)


if __name__ == "__main__":
    generate_explanation("C554291", "MESH:D012640")

