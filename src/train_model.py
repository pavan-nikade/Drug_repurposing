import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from pathlib import Path
import joblib

BASE = Path(r"C:\\Users\\pavan\\demo_fyp\\processed")

# Load embeddings
wv = KeyedVectors.load_word2vec_format(
    BASE / "node_embeddings.txt"
)

# Load samples
samples = pd.read_csv(BASE / "link_samples.csv")

X = []
y = []

for _, r in samples.iterrows():
    drug = f"drug:{r.ChemicalID}"
    disease = f"disease:{r.DiseaseID}"

    if drug in wv and disease in wv:
        vec = np.concatenate([wv[drug], wv[disease]])
        X.append(vec)
        y.append(r.label)

X = np.array(X)
y = np.array(y)

print("Usable samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:,1]

print("AUC:", roc_auc_score(y_test, prob))
print(classification_report(y_test, pred))

joblib.dump(clf, BASE / "rf_model.pkl")

print("Model saved to rf_model.pkl")