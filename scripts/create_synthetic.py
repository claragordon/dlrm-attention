import argparse
import json
import os

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_base_case(n_rows=200_000, n_users=50_000, n_items=100_000, emb_dim=16, seed=42):
    rng = np.random.default_rng(seed)

    user_emb = rng.normal(0, 1, size=(n_users, emb_dim)).astype(np.float32)
    item_emb = rng.normal(0, 1, size=(n_items, emb_dim)).astype(np.float32)
    w_dense = rng.normal(0, 0.2, size=(13,)).astype(np.float32)

    y = np.zeros((n_rows,), dtype=np.uint8)
    dense_arr = np.zeros((n_rows, 13), dtype=np.float32)
    sparse_arr = np.zeros((n_rows, 26), dtype=np.int32)

    for idx in range(n_rows):
        # Select random user, item, and item history
        u = rng.integers(0, n_users)
        i = rng.integers(0, n_items)
        hist = rng.integers(0, n_items, size=(8,))

        dense = rng.normal(0, 1, size=(13,)).astype(np.float32)

        s_user_item = float(np.dot(user_emb[u], item_emb[i]) / np.sqrt(emb_dim))
        s_dense = float(np.dot(dense, w_dense))
        noise = float(rng.normal(0, 0.2))

        bias = -3.3  # sigmoid(-3.3) ~= 0.035 when other terms average near zero
        score = bias + 1.0 * s_user_item + 0.3 * s_dense + noise
        p = sigmoid(score)
        y[idx] = 1 if rng.random() < p else 0

        sparse = [0] * 26
        sparse[0] = u + 1
        sparse[1] = i + 1
        for j in range(8):
            sparse[2 + j] = int(hist[j]) + 1
        for j in range(10, 26):
            sparse[j] = 1
        dense_arr[idx] = dense
        sparse_arr[idx] = sparse

    return y, dense_arr, sparse_arr


def write_npz_splits(y, dense, sparse, out_dir, train_frac=0.8, val_frac=0.1):
    total = y.shape[0]
    n_train = int(total * train_frac)
    n_val = int(total * val_frac)
    n_test = total - n_train - n_val
    splits = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, total),
    }
    for name, sl in splits.items():
        np.savez_compressed(
            os.path.join(out_dir, f"{name}.npz"),
            y=y[sl],
            dense=dense[sl],
            sparse=sparse[sl],
        )
    stats = {
        "rows_total": int(total),
        "rows_train": int(n_train),
        "rows_val": int(n_val),
        "rows_test": int(n_test),
        "ctr_total": float(y.mean()) if total else 0.0,
        "ctr_train": float(y[splits["train"]].mean()) if n_train else 0.0,
        "ctr_val": float(y[splits["val"]].mean()) if n_val else 0.0,
        "ctr_test": float(y[splits["test"]].mean()) if n_test else 0.0,
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def write_tsv(y, dense, sparse, out_path):
    with open(out_path, "w") as f:
        for idx in range(y.shape[0]):
            row = [str(int(y[idx]))]
            row.extend(f"{x:.6f}" for x in dense[idx])
            row.extend(format(int(x), "x") for x in sparse[idx])
            f.write("\t".join(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--users", type=int, default=50_000)
    ap.add_argument("--n_items", type=int, default=100_000)
    ap.add_argument("--emb_dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.80)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--write_tsv", action="store_true")

    args = ap.parse_args()
    if args.train_frac <= 0 or args.val_frac <= 0 or args.train_frac + args.val_frac >= 1:
        raise ValueError("train_frac and val_frac must be >0 and sum to <1.")
    os.makedirs(args.out_dir, exist_ok=True)

    y, dense, sparse = generate_base_case(
        n_rows=args.rows,
        n_users=args.users,
        n_items=args.n_items,
        emb_dim=args.emb_dim,
        seed=args.seed,
    )
    write_npz_splits(
        y=y,
        dense=dense,
        sparse=sparse,
        out_dir=args.out_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    if args.write_tsv:
        write_tsv(y, dense, sparse, os.path.join(args.out_dir, "base.tsv"))
    print(f"Wrote synthetic splits to {args.out_dir}")


if __name__ == "__main__":
    main()