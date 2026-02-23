import argparse, json, os
import numpy as np
import hashlib

NUM_DENSE = 13
NUM_SPARSE = 26

def stable_hash(s: str) -> int:
    # stable across runs/machines (unlike Python's hash())
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

def parse_dense(x: str) -> float:
    if x == "" or x is None:
        return 0.0
    try:
        return float(x)
    except ValueError:
        return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hash_size", type=int, default=131072)
    ap.add_argument("--train_frac", type=float, default=0.80)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--max_rows", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # First pass: count rows (unless max_rows set and small)
    total = 0
    with open(args.in_txt, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            total += 1
            if args.max_rows and total >= args.max_rows:
                break

    n_train = int(total * args.train_frac)
    n_val = int(total * args.val_frac)
    n_test = total - n_train - n_val

    # Allocate arrays
    y = np.zeros((total,), dtype=np.uint8)
    dense = np.zeros((total, NUM_DENSE), dtype=np.float32)
    sparse = np.zeros((total, NUM_SPARSE), dtype=np.int32)

    # Load
    i = 0
    with open(args.in_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if args.max_rows and i >= args.max_rows:
                break
            parts = line.rstrip("\n").split("\t")
            # Expected 1 + 13 + 26 = 40 columns
            if len(parts) < 1 + NUM_DENSE + NUM_SPARSE:
                continue

            y[i] = 1 if parts[0] == "1" else 0

            # dense
            for j in range(NUM_DENSE):
                dense[i, j] = parse_dense(parts[1 + j])

            # sparse
            base = 1 + NUM_DENSE
            for j in range(NUM_SPARSE):
                v = parts[base + j]
                if v == "" or v is None:
                    sparse[i, j] = 0
                else:
                    # include field index to avoid collisions across fields
                    h = stable_hash(f"{j}:{v}") % args.hash_size
                    sparse[i, j] = int(h) + 1  # reserve 0 for missing
            i += 1

    # If we skipped malformed lines, trim
    y = y[:i]
    dense = dense[:i]
    sparse = sparse[:i]

    # Recompute splits in case i != total
    total = i
    n_train = int(total * args.train_frac)
    n_val = int(total * args.val_frac)
    n_test = total - n_train - n_val

    # Compute train mean/std for dense and normalize all
    d_train = dense[:n_train]
    mean = d_train.mean(axis=0)
    std = d_train.std(axis=0)
    std = np.maximum(std, 1e-6)

    dense = (dense - mean) / std

    # Split views
    splits = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, total),
    }

    for name, sl in splits.items():
        out_path = os.path.join(args.out_dir, f"{name}.npz")
        np.savez_compressed(out_path, y=y[sl], dense=dense[sl], sparse=sparse[sl])

    stats = {
        "rows_total": int(total),
        "rows_train": int(n_train),
        "rows_val": int(n_val),
        "rows_test": int(n_test),
        "hash_size": int(args.hash_size),
        "dense_mean": mean.tolist(),
        "dense_std": std.tolist(),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("Wrote:", args.out_dir)
    print(stats)

if __name__ == "__main__":
    main()