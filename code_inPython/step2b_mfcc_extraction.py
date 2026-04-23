import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SUMMARY_CSV = "mfcc_index.csv"
PLOTS_DIR   = "plots_mfcc"
os.makedirs(PLOTS_DIR, exist_ok=True)

def pick_representative(df_label):
    if df_label.empty:
        return None
    median_T = df_label["frames_T"].median()
    idx = (df_label["frames_T"] - median_T).abs().astype(float).idxmin()
    return df_label.loc[idx]

def plot_heatmap(npy_path, out_png, mfcc_only=False, n_mfcc=20, title=None):
    feats = np.load(npy_path)  
    if mfcc_only:
      
        feats = feats[:n_mfcc, :]

    plt.figure(figsize=(8, 4.5))
    plt.imshow(feats, aspect="auto", origin="lower")
    plt.colorbar()
    plt.xlabel("Frames (time)")
    plt.ylabel("Coefficients")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main(summary_csv=SUMMARY_CSV, plots_dir=PLOTS_DIR, mfcc_only=False, n_mfcc=20):
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Missing {summary_csv}. Run Step 2B first.")

    df = pd.read_csv(summary_csv)
    req = {"label", "npy_path", "frames_T", "feat_dim"}
    if not req.issubset(df.columns):
        raise ValueError(f"{summary_csv} must contain columns: {req}")

    labels = sorted(df["label"].astype(str).unique().tolist())
    print(f"Found {len(labels)} labels in '{summary_csv}'.")

    made = 0
    for lab in labels:
        df_lab = df[df["label"].astype(str) == lab].copy()
        row = pick_representative(df_lab)
        if row is None:
            print(f"[skip] no entries for label '{lab}'")
            continue

        npy_path = row["npy_path"]
        if not os.path.exists(npy_path):
            print(f"[skip] missing npy: {npy_path}")
            continue

        out_png = os.path.join(plots_dir, f"{lab}_mfcc.png")
        ttl = f"{lab} — MFCC{' (only)' if mfcc_only else ' (+Δ+ΔΔ if present)'}"
        try:
            plot_heatmap(npy_path, out_png, mfcc_only=mfcc_only, n_mfcc=n_mfcc, title=ttl)
            made += 1
            print(f"Saved heatmap -> {out_png}")
        except Exception as e:
            print(f"[skip] plotting failed for {npy_path}: {e}")

    print(f"\nDone. Generated {made} heatmaps in: {plots_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MFCC heatmaps (one per vowel/label)")
    ap.add_argument("--summary_csv", type=str, default=SUMMARY_CSV)
    ap.add_argument("--out_dir", type=str, default=PLOTS_DIR)
    ap.add_argument("--mfcc_only", action="store_true", help="Plot only MFCC rows (drop deltas)")
    ap.add_argument("--n_mfcc", type=int, default=20, help="MFCC count to keep if --mfcc_only")
    args = ap.parse_args()

    main(
        summary_csv=args.summary_csv,
        plots_dir=args.out_dir,
        mfcc_only=args.mfcc_only,
        n_mfcc=args.n_mfcc
    )