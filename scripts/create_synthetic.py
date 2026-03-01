import argparse
import os

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_base_case(n_rows=200_000, n_users=50_000, n_items=100_000, emb_dim=16, seed=42):
    rng = np.random.default_rng(seed)
    
    U = rng.normal(0, 1, size=(n_users, emb_dim)).astype(np.float32)
    I = rng.normal(0, 1, size=(n_items, emb_dim)).astype(np.float32)
    w_dense = rng.normal(0, 0.2, size=(13,)).astype(np.float32)
    
    rows = []
    for _ in range(n_rows):
        # Select random user, item, and item history
        u = rng.integers(0, n_users)
        i = rng.integers(0, n_items)
        hist = rng.integers(0, n_items, size=(8,))

        dense = rng.normal(0, 1, size = (13,)).astype(np.float32)

        s_user_item = float(np.dot(U[u], I[i])/np.sqrt(emb_dim))
        s_hist_item = float(np.mean([np.dot(I[i], I[h]) / np.sqrt(emb_dim) for h in hist]))
        s_dense = float(np.dot(dense, w_dense))
        noise = float(rng.normal(0, 0.2))

        bias = -3.3  # sigmoid(-3.3) ~= 0.035 when other terms average near zero
        score = bias + 1.0 * s_user_item + 0.8 * s_hist_item + 0.3 * s_dense + noise
        p = sigmoid(score)
        y = 1 if rng.random() < p else 0

        sparse = [0] * 26
        sparse[0] = u + 1 
        sparse[1] = i + 1 
        for j in range(8):
            sparse[2 + j] = int(hist[j]) + 1
        for j in range(10, 26):
            sparse[j] = int(rng.integers(1, 20000))

        row = [str(y)] + [f"{x:.6f}" for x in dense] + [format(x, "x") for x in sparse]
        rows.append("\t".join(row))
        
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--users", type=int, default=50_000)
    ap.add_argument("--n_items", type=int, default=100_000)
    ap.add_argument("--emb_dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base_rows = generate_base_case(
        n_rows=args.rows,
        n_users=args.users,
        n_items=args.n_items,
        emb_dim=args.emb_dim,
        seed=args.seed,
    )
    with open(os.path.join(args.out_dir, "base.tsv"), "w") as f:
        f.write("\n".join(base_rows) + "\n")

if __name__ == "__main__": 
    main()