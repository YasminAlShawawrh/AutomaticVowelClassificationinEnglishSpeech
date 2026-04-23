import os, argparse, csv
import numpy as np
import librosa
from scipy.signal.windows import hamming

RAW_DIR_DEFAULT = "vowel_data_raw"
OUT_DIR_DEFAULT = "preproc_framed"

def list_wavs(root):
    items = []
    for label in sorted(os.listdir(root)):
        p = os.path.join(root, label)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.lower().endswith(".wav"):
                items.append((label, os.path.join(p, f)))
    return items

def pre_emphasis(x, alpha=0.97):
    # y[n] = x[n] - a*x[n-1]
    if x.size == 0:
        return x
    return np.append(x[0], x[1:] - alpha * x[:-1])

def frame_and_window(y, sr, frame_ms=25.0, hop_ms=10.0):
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len   = int(sr * hop_ms / 1000.0)
    if len(y) < frame_len:
        return np.empty((0, frame_len), dtype=np.float32)
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len).T
    win = hamming(frame_len, sym=False).astype(np.float32)
    return (frames * win).astype(np.float32)

def process_file(wav_path, out_path, sr=16000, alpha=0.97, frame_ms=25.0, hop_ms=10.0):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    mx = np.max(np.abs(y)) if y.size else 0.0
    if mx > 0:
        y = 0.99 * (y / mx)
    y = pre_emphasis(y, alpha=alpha)
    framed = frame_and_window(y, sr=sr, frame_ms=frame_ms, hop_ms=hop_ms)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, framed)
    return framed

def main(raw_dir, out_dir, sr, alpha, frame_ms, hop_ms, index_csv):
    os.makedirs(out_dir, exist_ok=True)
    items = list_wavs(raw_dir)
    saved, skipped = 0, 0

    with open(index_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["label","wav_path","n_frames","frame_len","hop_len","duration_sec","npy_path"])

        for label, wav_path in items:
            rel_name = os.path.splitext(os.path.basename(wav_path))[0] + ".npy"
            out_path = os.path.join(out_dir, label, rel_name)
            if os.path.exists(out_path):
                try:
                    arr = np.load(out_path, mmap_mode="r")
                    n_frames = arr.shape[0] if arr.ndim == 2 else 0
                    duration = librosa.get_duration(path=wav_path)
                    writer.writerow([label, wav_path, n_frames,
                                     int(sr*frame_ms/1000.0), int(sr*hop_ms/1000.0),
                                     round(duration,3), out_path])
                except Exception:
                    pass
                continue

            try:
                arr = process_file(
                    wav_path, out_path, sr=sr, alpha=alpha,
                    frame_ms=frame_ms, hop_ms=hop_ms
                )
                n_frames = arr.shape[0]
                duration = librosa.get_duration(path=wav_path)
                writer.writerow([label, wav_path, n_frames,
                                 int(sr*frame_ms/1000.0), int(sr*hop_ms/1000.0),
                                 round(duration,3), out_path])
                saved += 1
                print(f" {label}/{os.path.basename(out_path)}  -> frames={n_frames}")
            except Exception as e:
                skipped += 1
                print(f" skip {wav_path}: {e}")

    print(f"\nDone. Saved: {saved} | Skipped: {skipped}")
    print(f"Output dir: {out_dir}")
    print(f"Index CSV : {index_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Preprocessing (resample, pre-emphasis, framing, Hamming)")
    parser.add_argument("--raw", type=str, default=RAW_DIR_DEFAULT, help="Root of raw wavs (by label)")
    parser.add_argument("--out", type=str, default=OUT_DIR_DEFAULT, help="Output root for framed .npy")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--alpha", type=float, default=0.97, help="Pre-emphasis factor")
    parser.add_argument("--frame_ms", type=float, default=25.0, help="Frame length in ms")
    parser.add_argument("--hop_ms", type=float, default=10.0, help="Hop length in ms")
    parser.add_argument("--index", type=str, default="preproc_index.csv", help="CSV index path")
    args = parser.parse_args()

    main(args.raw, args.out, args.sr, args.alpha, args.frame_ms, args.hop_ms, args.index)
