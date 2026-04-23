# Automatic Vowel Classification in English Speech

A machine learning pipeline for classifying English vowels using two acoustic feature sets: **Formants (F1, F2)** and **MFCCs**. The project compares classifier performance across feature types using SVM and Random Forest with 5-fold cross-validation, and produces a full set of visualizations including vowel space plots, MFCC heatmaps, and confusion matrices.


---

## Table of contents

- [Project overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Feature extraction](#feature-extraction)
- [Classification](#classification)
- [Results](#results)
- [Output files](#output-files)

---

## Project overview

Vowels are distinguished by their formant frequencies — particularly the first formant (F1) and second formant (F2) — which reflect tongue height and backness. This project builds a vowel classifier that:
1. Preprocesses raw speech audio (normalization, segmentation, pre-emphasis)
2. Extracts formant features (F1, F2) using Praat via `parselmouth`
3. Extracts MFCC features (13 coefficients + mean/std) using `librosa`
4. Trains and evaluates SVM and Random Forest classifiers on each feature set
5. Compares formants vs MFCCs and discusses which features work better and why

---

## Dataset

Uses the **Vowel Dataset** from the University of Texas at Dallas:  
`https://personal.utdallas.edu/~assmann/KIDVOW/`

- Contains audio recordings of 12 English vowels produced by adult males, adult females, and children (ages 3–7)
- Vowels are labeled using the carrier word format: **heed, hid, hayed, head, had, hod, hawed, hoed, hood, whod, hud, herd**
- Useful for exploring how F1 and F2 vary across speaker age and gender

---

## Pipeline

```
download_dataset.py
        ↓
step1_preprocessing.py     # Resample to 16 kHz, pre-emphasis, windowing → preproc_index.csv
        ↓
step2a_formant_extraction.py   # Praat LPC formant tracking → F1, F2 → formant_index.csv
step2b_mfcc_extraction.py      # librosa 13-MFCC extraction → .npy files → mfcc_index.csv
        ↓
step3_classification.py        # SVM + RF on each feature set → confusion matrices
        ↓
step4_compare_models.py        # Full comparison: Formants (no ML) vs Formants SVM/RF vs MFCC SVM/RF
```

---

## Feature extraction

### Formant extraction (`step2a_formant_extraction.py`)
Uses `parselmouth` (Python bindings for Praat) with the **Burg LPC algorithm**:

- Detects speaker type from pitch estimate to set `max_formant` adaptively:
  - F0 > 260 Hz (children) → max formant = 6000 Hz
  - F0 > 180 Hz (female) → max formant = 5500 Hz
  - F0 ≤ 180 Hz (male) → max formant = 5000 Hz
- Samples F1 and F2 at 5 time points around the vowel midpoint
- Uses median across time points to reduce noise from transient formant tracking errors
- Validity filter: `90 < F1 < 1500`, `300 < F2 < 4000`, `F2 > F1`
- Outputs: `formant_index.csv` with columns `[label, F1, F2]`

### MFCC extraction (`step2b_mfcc_extraction.py`)
Uses `librosa.feature.mfcc()`:
- 13 MFCC coefficients per frame
- Feature vector = `[mean(mfcc_1..13), std(mfcc_1..13)]` → 26-dimensional vector
- Outputs: per-file `.npy` arrays + `mfcc_index.csv` index

---

## Classification

### Models (`step3_classification.py`)

| Model | Configuration |
|---|---|
| SVM | RBF kernel · C=10 · gamma='scale' |
| Random Forest | 100 estimators · random_state=42 |

Both models use `StandardScaler` inside a 5-fold `StratifiedKFold` cross-validation loop. Metrics reported per fold and averaged:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score

### Comparison (`step4_compare_models.py`)
Evaluates 5 configurations to isolate the effect of features vs classifiers:

| Configuration | Features | Model |
|---|---|---|
| Formants (no ML) | F1, F2 | Nearest Centroid |
| Formants SVM | F1, F2 | SVM (RBF) |
| Formants RF | F1, F2 | Random Forest |
| MFCC SVM | 26-dim MFCC | SVM (RBF) |
| MFCC RF | 26-dim MFCC | Random Forest |

---

## Results

Four confusion matrices are generated — one per feature/model combination:

| Configuration | Confusion matrix |
|---|---|
| Formants + SVM | `cm_Formants_SVM.png` |
| Formants + RF | `cm_Formants_RandomForest.png` |
| MFCC + SVM | `cm_MFCC_SVM.png` |
| MFCC + RF | `cm_MFCC_RandomForest.png` |

**Key findings:**
- MFCC features outperform Formants-only across both classifiers — MFCCs capture spectral envelope shape beyond just two resonance peaks
- Formants without ML (Nearest Centroid) perform worst — the F1/F2 vowel space has significant overlap especially across speaker types (male/female/children)
- SVM with MFCC achieves the best classification accuracy
- Vowels most commonly confused: pairs with similar tongue positions (e.g. hid/head, hood/whod)

---

## Output files

```
Confusion_matrices/
├── cm_Formants_SVM.png
├── cm_Formants_RandomForest.png
├── cm_MFCC_SVM.png
└── cm_MFCC_RandomForest.png

plots_mfcc/
├── heed_mfcc.png
├── hid_mfcc.png
├── hayed_mfcc.png
├── head_mfcc.png
├── had_mfcc.png
├── hod_mfcc.png
├── hawed_mfcc.png
├── hoed_mfcc.png
├── hood_mfcc.png
├── whod_mfcc.png
├── hud_mfcc.png
└── herd_mfcc.png

F1_Vs_F2_Plot.png              # Vowel space scatter plot (F2 on x-axis, F1 on y-axis)
formant_index.csv              # Extracted F1, F2 per sample
mfcc_index.csv                 # MFCC .npy file paths and labels
preproc_index.csv              # Preprocessed audio paths and labels
```
