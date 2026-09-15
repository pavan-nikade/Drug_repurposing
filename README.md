# Explainable Drug Discovery

A research prototype for exploring drug–disease hypotheses from a biomedical knowledge graph. It combines graph paths, node embeddings, a Random Forest association model, and topology-aware evidence scoring to produce explainable candidate results.

> This software generates graph-based research hypotheses. It is not a clinical decision-support tool and does not establish causality, safety, or treatment effectiveness.

## What the application does

1. Select a drug from the Streamlit interface.
2. Find semantic graph paths of either form:
   - `Drug → Gene → Disease` (direct evidence)
   - `Drug → Gene → Pathway → Disease` (indirect mechanistic evidence)
3. Score disease candidates using a trained predictive model and Topology-aware Mechanistic Score (TMS).
4. Display the highest-ranked candidates, an explanation, supporting pathways, and a mechanism diagram.

## Project layout

```text
app.py                 Streamlit discovery interface
discover.py            Candidate discovery, predictive scoring, TMS fusion
network_view.py        Mechanism-path visualization
final_explainer.py     CLI graph-path explainer for a chosen drug/disease pair
llm_explainer.py       Optional local Ollama narrative generator

row_data/              Source CTD datasets
processed/             Cleaned data, graph, embeddings, and trained model
src/preprocess.py      Cleans the raw CTD files
src/build_graph.py     Builds biomedical_graph.gpickle
src/create_samples.py  Creates positive/negative model training pairs
src/embeddings.py      Trains Node2Vec embeddings
src/train_model.py     Trains the Random Forest classifier
src/evidance_score.py  Standalone TMS/evidence-fusion calculation
```

## Data inputs

The data originates from CTD-derived CSV files in `row_data/`. Preprocessing creates the following files in `processed/`.

| File | Main fields | Purpose |
| --- | --- | --- |
| `chem_disease.csv` | Chemical name/ID, disease name/ID, direct evidence | Therapeutic chemical–disease associations |
| `chem_gene.csv` | Chemical ID, gene symbol/ID, interaction | Drug–gene interactions |
| `gene_disease.csv` | Gene symbol/ID, disease name/ID, evidence | Gene–disease associations |
| `gene_pathway.csv` | Gene symbol/ID, pathway name/ID | Gene–pathway links |
| `disease_pathway.csv` | Disease name/ID, pathway name/ID | Disease–pathway links |
| `link_samples.csv` | Chemical ID, disease ID, label | Training data for association prediction |

`src/build_graph.py` converts these tables into an undirected NetworkX graph. Nodes use stable prefixed IDs:

```text
drug:<ChemicalID>
gene:<GeneID>
pathway:<PathwayID>
disease:<DiseaseID>
```

## Outputs

For each candidate disease, the application reports:

| Output | Meaning |
| --- | --- |
| Final score | Weighted ranking score used to order candidate hypotheses |
| Predictive | Random Forest probability for the drug–disease pair |
| TMS | Topology-aware Mechanistic Score from graph evidence |
| Paths | Number of valid direct and indirect semantic paths |
| Mechanism graph | Selected Drug → Gene → Pathway → Disease explanation path |
| Supporting pathways | The three most frequent pathway labels among indirect paths |

Results are labelled **High confidence** only when they have at least 10 valid paths, predictive score ≥ 0.75, and TMS ≥ 0.60. Other graph-supported candidates are shown as **Exploratory evidence** rather than being hidden.

## TMS evidence fusion

TMS combines five structural and evidence channels:

```text
TMS = sigmoid(w1·P + w2·G + w3·PW + w4·C + w5·D)
```

Where:

- `P`: path abundance, using `log(1 + total paths)`
- `G`: Shannon diversity of the supporting genes
- `PW`: Shannon diversity of indirect-path pathways
- `C`: pathway consensus using the support share of the top three pathways
- `D`: evidence depth, calculated as:

```text
(direct paths + 0.7 × indirect paths) / total paths
```

Direct paths represent independent `Drug → Gene → Disease` support. Indirect paths represent `Drug → Gene → Pathway → Disease` mechanistic support.

## Run the application

### Prerequisites

Use Python 3.10 or later. Install the packages used by the project:

```bash
pip install streamlit pandas numpy scipy networkx matplotlib joblib scikit-learn gensim node2vec requests
```

The repository must contain the prebuilt artifacts in `processed/`:

- `biomedical_graph.gpickle`
- `node_embeddings.txt`
- `rf_model.pkl`

### Start Streamlit

```bash
streamlit run app.py
```

Open the local URL Streamlit displays. Choose a drug in **Select a drug**; the dropdown can be searched by chemical name or CTD chemical ID. Then select **Run Discovery**.

## Rebuilding the pipeline

Run these scripts from the project root when raw data changes:

```bash
python src/preprocess.py
python src/build_graph.py
python src/create_samples.py
python src/embeddings.py
python src/train_model.py
```

Graph building samples gene–pathway and disease–pathway relations to keep the demonstration graph manageable. Rebuilding can therefore change downstream paths, embeddings, scores, and rankings.

## Optional local LLM explanation

`final_explainer.py` and `llm_explainer.py` can request a local Ollama server at `http://localhost:11434` using the `qwen2.5:3b-instruct` model. This is optional for the Streamlit discovery interface. Ensure Ollama is running and that the model is available before using the CLI explainer.

```bash
python final_explainer.py
```

## Limitations

- The graph represents associations from the bundled source data, not experimental confirmation.
- Scores should be compared within this prototype; they are not calibrated clinical probabilities.
- Pathway sampling in graph construction can omit otherwise relevant pathways.
- Text explanations are interpretive summaries and must be reviewed against primary biomedical evidence.
