import argparse
import json
import os

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_base_case(
    n_rows=200_000,
    n_users=50_000,
    n_items=100_000,
    emb_dim=16,
    seed=42,
    noise_std=0.2,
    w_user_item=1.0,
    w_dense=0.3,
    bias=-3.3,
    deterministic_labels=False,
    include_history_noise=False,
    unused_sparse_id=0,
    dataset_mode="base",
    history_len=8,
    w_seq=1.0,
    recency_alpha=2.0,
    zero_dense=False,
    n_anchor_fields=0,
    anchor_cardinality=128,
):
    rng = np.random.default_rng(seed)
    if dataset_mode not in {"base", "seq_easy", "seq_hard"}:
        raise ValueError("dataset_mode must be one of: base, seq_easy, seq_hard")
    if history_len < 1 or history_len > 8:
        raise ValueError("history_len must be in [1, 8] to fit sparse[2:10].")
    if anchor_cardinality < 2:
        raise ValueError("anchor_cardinality must be >= 2.")
    max_anchor_fields = 26 - (2 + history_len)
    if n_anchor_fields < 0 or n_anchor_fields > max_anchor_fields:
        raise ValueError(f"n_anchor_fields must be in [0, {max_anchor_fields}] for history_len={history_len}.")

    user_emb = rng.normal(0, 1, size=(n_users, emb_dim)).astype(np.float32)
    item_emb = rng.normal(0, 1, size=(n_items, emb_dim)).astype(np.float32)
    anchor_embs = [
        rng.normal(0, 1, size=(anchor_cardinality, emb_dim)).astype(np.float32)
        for _ in range(n_anchor_fields)
    ]
    w_dense_vec = rng.normal(0, 0.2, size=(13,)).astype(np.float32)
    recency_weights = np.linspace(1.0, recency_alpha, history_len).astype(np.float32)
    recency_weights /= recency_weights.sum()

    y = np.zeros((n_rows,), dtype=np.uint8)
    dense_arr = np.zeros((n_rows, 13), dtype=np.float32)
    sparse_arr = np.zeros((n_rows, 26), dtype=np.int32)

    for idx in range(n_rows):
        # Select random user, item, and item history
        u = rng.integers(0, n_users)
        i = rng.integers(0, n_items)
        hist = rng.integers(0, n_items, size=(history_len,))

        if zero_dense:
            dense = np.zeros((13,), dtype=np.float32)
        else:
            dense = rng.normal(0, 1, size=(13,)).astype(np.float32)

        s_user_item = float(np.dot(user_emb[u], item_emb[i]) / np.sqrt(emb_dim))
        s_dense = float(np.dot(dense, w_dense_vec))
        if dataset_mode == "base":
            s_seq = 0.0
        else:
            sims = np.array(
                [np.dot(item_emb[i], item_emb[h]) / np.sqrt(emb_dim) for h in hist],
                dtype=np.float32,
            )
            if dataset_mode == "seq_easy":
                # Mild sequential dependency: weighted average similarity with recency.
                s_seq = float(np.dot(recency_weights, sims))
            else:
                # Stronger sequential dependency: best matching recent signal.
                s_seq = float(np.max(recency_weights * sims))
        s_anchor = 0.0
        anchor_ids = []
        for k in range(n_anchor_fields):
            # Deterministic anchor IDs tied to user/item create additional non-sequence signal.
            aid = int((31 * int(u) + 17 * int(i) + 13 * k) % anchor_cardinality)
            anchor_ids.append(aid + 1)
            s_anchor += float(np.dot(item_emb[i], anchor_embs[k][aid]) / np.sqrt(emb_dim))
        noise = float(rng.normal(0, noise_std))

        s_nonseq = s_user_item + s_anchor / np.sqrt(n_anchor_fields)
        score = bias + w_user_item * s_nonseq + w_seq * s_seq + w_dense * s_dense + noise
        p = sigmoid(score)
        if deterministic_labels:
            y[idx] = 1 if score > 0.0 else 0
        else:
            y[idx] = 1 if rng.random() < p else 0

        sparse = [0] * 26
        sparse[0] = u + 1
        sparse[1] = i + 1
        if dataset_mode != "base" or include_history_noise:
            for j in range(history_len):
                sparse[2 + j] = int(hist[j]) + 1
        anchor_start = 2 + history_len
        for k, aid in enumerate(anchor_ids):
            sparse[anchor_start + k] = aid
        for j in range(2 + history_len, 26):
            if sparse[j] == 0:
                sparse[j] = int(unused_sparse_id)
        dense_arr[idx] = dense
        sparse_arr[idx] = sparse

    return y, dense_arr, sparse_arr


def write_npz_splits(
    y,
    dense,
    sparse,
    out_dir,
    train_frac=0.8,
    val_frac=0.1,
    generation_config=None,
):
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
    if generation_config is not None:
        stats["generation_config"] = generation_config
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
    ap.add_argument("--noise_std", type=float, default=0.2)
    ap.add_argument("--w_user_item", type=float, default=1.0)
    ap.add_argument("--w_seq", type=float, default=1.0)
    ap.add_argument("--w_dense", type=float, default=0.3)
    ap.add_argument("--bias", type=float, default=-3.3)
    ap.add_argument("--dataset_mode", choices=["base", "seq_easy", "seq_hard"], default="base")
    ap.add_argument("--history_len", type=int, default=8)
    ap.add_argument("--recency_alpha", type=float, default=2.0)
    ap.add_argument("--zero_dense", action="store_true")
    ap.add_argument("--n_anchor_fields", type=int, default=0)
    ap.add_argument("--anchor_cardinality", type=int, default=128)
    ap.add_argument("--deterministic_labels", action="store_true")
    ap.add_argument("--include_history_noise", action="store_true")
    ap.add_argument("--unused_sparse_id", type=int, default=0)
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
        noise_std=args.noise_std,
        w_user_item=args.w_user_item,
        w_seq=args.w_seq,
        w_dense=args.w_dense,
        bias=args.bias,
        dataset_mode=args.dataset_mode,
        history_len=args.history_len,
        recency_alpha=args.recency_alpha,
        zero_dense=args.zero_dense,
        n_anchor_fields=args.n_anchor_fields,
        anchor_cardinality=args.anchor_cardinality,
        deterministic_labels=args.deterministic_labels,
        include_history_noise=args.include_history_noise,
        unused_sparse_id=args.unused_sparse_id,
    )
    generation_config = {
        "rows": int(args.rows),
        "users": int(args.users),
        "n_items": int(args.n_items),
        "emb_dim": int(args.emb_dim),
        "seed": int(args.seed),
        "dataset_mode": args.dataset_mode,
        "history_len": int(args.history_len),
        "recency_alpha": float(args.recency_alpha),
        "zero_dense": bool(args.zero_dense),
        "noise_std": float(args.noise_std),
        "w_user_item": float(args.w_user_item),
        "w_seq": float(args.w_seq),
        "w_dense": float(args.w_dense),
        "bias": float(args.bias),
        "n_anchor_fields": int(args.n_anchor_fields),
        "anchor_cardinality": int(args.anchor_cardinality),
        "deterministic_labels": bool(args.deterministic_labels),
        "include_history_noise": bool(args.include_history_noise),
        "unused_sparse_id": int(args.unused_sparse_id),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
    }
    write_npz_splits(
        y=y,
        dense=dense,
        sparse=sparse,
        out_dir=args.out_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        generation_config=generation_config,
    )
    if args.write_tsv:
        write_tsv(y, dense, sparse, os.path.join(args.out_dir, "base.tsv"))
    print(f"Wrote synthetic splits to {args.out_dir}")


if __name__ == "__main__":
    main()