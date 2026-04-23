import os, random, math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import parselmouth
from parselmouth.praat import call

INDEX_CSV = "preproc_index.csv"  
OUT_DIR   = "features_formant"
os.makedirs(OUT_DIR, exist_ok=True)

def choose_max_formant(sound):
    try:
        pitch = call(sound, "To Pitch (ac)", 0.0, 75, 600)  
        f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    except Exception:
        f0 = 150.0
    if f0 and f0 > 260:
        return 6000  
    elif f0 and f0 > 180:
        return 5500 
    else:
        return 5000 

def robust_f1f2_with_praat(sound, lpc_n_formants=5, win=0.025, step=0.01):
    dur = sound.get_total_duration()
    if dur <= 0.04:
        t_grid = [dur / 2.0]
    else:
        mid = dur / 2.0
        t_grid = [max(0.001, min(dur - 0.001, mid + d)) for d in (-0.02, -0.01, 0, 0.01, 0.02)]

    max_formant = choose_max_formant(sound)
    formant = call(sound, "To Formant (burg)", step, lpc_n_formants, max_formant, win, 50)

    f1_vals, f2_vals = [], []
    for t in t_grid:
        try:
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            if (f1 is not None and f2 is not None and
                90 < f1 < 1500 and 300 < f2 < 4000 and f2 > f1):
                f1_vals.append(f1); f2_vals.append(f2)
        except Exception:
            continue

    if len(f1_vals) == 0:
        return np.nan, np.nan
    return float(np.median(f1_vals)), float(np.median(f2_vals))


if not os.path.exists(INDEX_CSV):
    raise FileNotFoundError(
        f"Could not find {INDEX_CSV}. Run Step 1 to generate it, or update the path."
    )

df = pd.read_csv(INDEX_CSV)
required_cols = {"label", "wav_path"}
if not required_cols.issubset(set(df.columns)):
    raise ValueError(f"{INDEX_CSV} is missing required columns: {required_cols}")

labels, F1s, F2s = [], [], []
items = list(zip(df["label"].astype(str), df["wav_path"].astype(str)))
print(f"Found {len(items)} entries in '{INDEX_CSV}'.")

for label, wav_path in items:
    if not os.path.exists(wav_path):
        print(f"skip (missing file): {wav_path}")
        continue
    try:
        snd = parselmouth.Sound(wav_path)  
        f1, f2 = robust_f1f2_with_praat(snd, lpc_n_formants=5, win=0.025, step=0.01)
        if not (math.isnan(f1) or math.isnan(f2)):
            labels.append(label); F1s.append(f1); F2s.append(f2)
            out_lab = os.path.join(OUT_DIR, label); os.makedirs(out_lab, exist_ok=True)
            np.save(
                os.path.join(out_lab, os.path.basename(wav_path).replace(".wav", ".npy")),
                np.array([f1, f2], dtype=np.float32)
            )
    except Exception as e:
        print("skip", wav_path, "->", e)

print(f"\nExtracted {len(F1s)} valid (F1, F2) pairs.")


out_csv = "formant_index.csv"
df_out = pd.DataFrame({
    "label": labels,
    "F1": F1s,
    "F2": F2s
})
df_out.to_csv(out_csv, index=False)
print(f"Saved -> {out_csv}")


print("\n Random sample (one per label):")
for lab in sorted(set(labels)):
    idxs = [i for i, L in enumerate(labels) if L == lab]
    if idxs:
        i = random.choice(idxs)
        print(f"Label={labels[i]:<6} | F1={F1s[i]:6.1f} Hz | F2={F2s[i]:6.1f} Hz")



plt.figure(figsize=(9,7))
unique_labels = sorted(set(labels))
cmap = plt.colormaps.get_cmap('tab20')  
label_color = {lab: cmap(i / len(unique_labels)) for i, lab in enumerate(unique_labels)}

for lab in unique_labels:
    idxs = [i for i, L in enumerate(labels) if L == lab]
    f1_vals = [F1s[i] for i in idxs]
    f2_vals = [F2s[i] for i in idxs]
    plt.scatter(
        f2_vals, f1_vals,
        s=28, alpha=0.7,
        color=label_color[lab],
        label=lab
    )

ax = plt.gca()
ax.invert_xaxis()  
plt.xlabel("F2 (Hz)")
plt.ylabel("F1 (Hz)")
plt.title("Vowel Space (F2 vs F1) — Praat (parselmouth)")
plt.grid(True, alpha=0.25)
plt.legend(title="Vowels", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize=9)
plt.tight_layout()
plt.savefig("vowel_space_from_index.png", dpi=200)
plt.show()
print("Saved plot -> vowel_space_from_index.png")
