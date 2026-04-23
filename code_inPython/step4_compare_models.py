import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
import seaborn as sns

# Paths
FORMANT_CSV = "formant_index.csv"
MFCC_CSV = "mfcc_index.csv"
OUT_DIR = "outputs_step4"
os.makedirs(OUT_DIR, exist_ok=True)

# Load formant features (F1, F2)
def load_formants():
    df = pd.read_csv(FORMANT_CSV)
    X = df[["F1", "F2"]].values
    y = df["label"].astype(str).values
    return X, y

# Load MFCC features
def load_mfcc():
    df = pd.read_csv(MFCC_CSV)
    X, y = [], []
    for _, row in df.iterrows():
        if os.path.exists(row["npy_path"]):
            M = np.load(row["npy_path"])
            vec = np.concatenate([M.mean(axis=1), M.std(axis=1)])
            X.append(vec)
            y.append(str(row["label"]))
    return np.array(X), np.array(y)

# Plot confusion matrix
def plot_cm(cm, classes, title, filename):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close()

# Evaluate classifier
def evaluate(X, y, model, title):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, cms = [], []
    labels = np.unique(y)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        cms.append(confusion_matrix(y_test, y_pred, labels=labels))

    cm_mean = np.mean(cms, axis=0)
    plot_cm(cm_mean, labels, title, f"cm_{title.replace(' ', '_')}.png")
    print(f"{title} Accuracy: {np.mean(accs):.3f}")

def main():
    Xf, yf = load_formants()
    Xm, ym = load_mfcc()

    evaluate(Xf, yf, NearestCentroid(), "Formants (No ML)")
    evaluate(Xf, yf, SVC(kernel='rbf', C=10, gamma='scale'), "Formants SVM")
    evaluate(Xf, yf, RandomForestClassifier(n_estimators=100), "Formants RF")

    evaluate(Xm, ym, SVC(kernel='rbf', C=10, gamma='scale'), "MFCC SVM")
    evaluate(Xm, ym, RandomForestClassifier(n_estimators=100), "MFCC RF")

if __name__ == "__main__":
    main()