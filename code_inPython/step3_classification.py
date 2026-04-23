import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

MFCC_CSV = "mfcc_index.csv"
FORMANT_CSV = "formant_index.csv"

def load_features(csv_path):
    df = pd.read_csv(csv_path)
    if "npy_path" in df.columns:
        feats, labels = [], []
        for _, row in df.iterrows():
            npy = row["npy_path"]
            if not os.path.exists(npy):
                continue
            f = np.load(npy)
            feats.append(f.mean(axis=1))
            labels.append(str(row["label"]))
        return np.array(feats), np.array(labels)
    else:
        label_col = "label"
        feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].values
        y = df[label_col].astype(str).values
        return X, y

def evaluate_supervised(X, y, model, model_name, dataset_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, precs, recs, f1s, cms = [], [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        cms.append(confusion_matrix(y_test, y_pred, labels=np.unique(y)))
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        precs.append(p)
        recs.append(r)
        f1s.append(f)

    mean_cm = np.mean(cms, axis=0).astype(int)
    print(f"\n=== {dataset_name} with {model_name} ===")
    print(f"Acc: {np.mean(accs):.3f}  Prec: {np.mean(precs):.3f}  Rec: {np.mean(recs):.3f}  F1: {np.mean(f1s):.3f}")
    plot_confusion_matrix(mean_cm, np.unique(y), f"{dataset_name} - {model_name}", f"cm_{dataset_name}_{model_name}.png")

def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    Xf, yf = load_features(FORMANT_CSV)
    Xm, ym = load_features(MFCC_CSV)

    # formant features with SVM and RF
    evaluate_supervised(Xf, yf, SVC(kernel='rbf', C=10, gamma='scale'), "SVM", "Formants")
    evaluate_supervised(Xf, yf, RandomForestClassifier(n_estimators=100, random_state=42), "RandomForest", "Formants")

    # MFCC features with SVM and RF
    evaluate_supervised(Xm, ym, SVC(kernel='rbf', C=10, gamma='scale'), "SVM", "MFCC")
    evaluate_supervised(Xm, ym, RandomForestClassifier(n_estimators=100, random_state=42), "RandomForest", "MFCC")

if __name__ == "__main__":
    main()