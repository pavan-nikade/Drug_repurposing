"""Topology-aware evidence fusion for drug–disease graph hypotheses."""

from collections import Counter
from pathlib import Path
import pickle

import numpy as np
from scipy.stats import entropy


BASE = Path(__file__).resolve().parent.parent / "processed"

with (BASE / "biomedical_graph.gpickle").open("rb") as graph_file:
    G = pickle.load(graph_file)


def type_of(node):
    return G.nodes[node].get("type")


def sigmoid(value):
    """Map the fused structural score into the interval (0, 1)."""
    return 1 / (1 + np.exp(-value))


def evidence_score_v2(drug_id, disease_id):
    """Calculate TMS from independent direct and indirect evidence channels.

    Direct evidence is a drug → gene → disease path. Indirect evidence is a
    drug → gene → pathway → disease path. The latter contributes pathway
    diversity and pathway consensus; both contribute to evidence depth.
    """
    source = f"drug:{drug_id}"
    target = f"disease:{disease_id}"

    if source not in G or target not in G:
        print(f"Nodes not found in graph vocabulary. Check IDs: {source}, {target}")
        return None

    direct_paths = []
    indirect_paths = []
    try:
        drug_genes = [node for node in G.neighbors(source) if type_of(node) == "gene"]
        for gene in drug_genes:
            # Independent direct channel: drug → gene → disease.
            if G.has_edge(gene, target):
                direct_paths.append((source, gene, target))

            # Indirect mechanistic channel: drug → gene → pathway → disease.
            gene_pathways = [node for node in G.neighbors(gene) if type_of(node) == "pathway"]
            for pathway in gene_pathways:
                if G.has_edge(pathway, target):
                    indirect_paths.append((source, gene, pathway, target))
    except Exception as error:
        print(f"An error occurred during network traversal: {error}")
        return None

    paths = direct_paths + indirect_paths
    path_count = len(paths)
    if path_count == 0:
        return None

    genes = [path[1] for path in paths]
    pathways = [path[2] for path in indirect_paths]
    gene_counts = Counter(genes)
    pathway_counts = Counter(pathways)

    gene_probabilities = [count / path_count for count in gene_counts.values()]
    gene_diversity = entropy(gene_probabilities, base=2)

    if pathways:
        indirect_count = len(indirect_paths)
        pathway_probabilities = [count / indirect_count for count in pathway_counts.values()]
        pathway_diversity = entropy(pathway_probabilities, base=2)

        # Consensus accounts for agreement across several strongly supported pathways.
        top_three_support = sum(count for _, count in pathway_counts.most_common(3))
        pathway_consensus = top_three_support / indirect_count
    else:
        pathway_diversity = 0.0
        pathway_consensus = 0.0

    direct_count = len(direct_paths)
    indirect_count = len(indirect_paths)
    evidence_depth = (direct_count + 0.7 * indirect_count) / path_count

    # TMS = w1·P + w2·G + w3·PW + w4·C + w5·D
    # P: path abundance; G/PW: gene/pathway diversity; C: pathway consensus;
    # D: support from independent direct and indirect evidence channels.
    w1, w2, w3, w4, w5 = 0.30, 0.20, 0.20, 0.15, 0.15
    path_abundance = np.log1p(path_count)
    raw_score = (
        w1 * path_abundance
        + w2 * gene_diversity
        + w3 * pathway_diversity
        + w4 * pathway_consensus
        + w5 * evidence_depth
    )
    final_score = sigmoid(raw_score)

    return {
        "metrics": {
            "path_count": path_count,
            "direct_path_count": direct_count,
            "indirect_path_count": indirect_count,
            "path_abundance": round(float(path_abundance), 4),
            "gene_diversity": round(float(gene_diversity), 4),
            "pathway_diversity": round(float(pathway_diversity), 4),
            "pathway_consensus_top_3": round(float(pathway_consensus), 4),
            "evidence_depth": round(float(evidence_depth), 4),
        },
        "topology_fusion_score": round(float(final_score), 4),
    }


if __name__ == "__main__":
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(evidence_score_v2("C554291", "MESH:D012640"))