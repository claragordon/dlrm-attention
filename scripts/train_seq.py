import argparse
import json
import os
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data import SeqNPZ
from src.metrics import compute_metrics
from src.model_seq import SeqCTRModel


def run_benchmark(
    model,
    loader,
    device,
    loss_fn,
    opt,
    warmup_steps,
    benchmark_steps,
    seq_len,
    log_every=0,
):
    model.train()
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    # Warmup
    for step in range(warmup_steps):
        seq_ids, y = next_batch()
        seq_ids = seq_ids.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logit = model(seq_ids)
        loss = loss_fn(logit, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every > 0 and (step + 1) % log_every == 0:
            print(f"[bench] warmup step {step + 1}/{warmup_steps}", flush=True)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    step_times = []
    loss_sum = 0.0
    token_count = 0
    for step in range(benchmark_steps):
        seq_ids, y = next_batch()
        seq_ids = seq_ids.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bsz = int(seq_ids.size(0))
        t0 = time.perf_counter()
        logit = model(seq_ids)
        loss = loss_fn(logit, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        loss_sum += float(loss.item())
        token_count += bsz * seq_len
        if log_every > 0 and (step + 1) % log_every == 0:
            print(f"[bench] timed step {step + 1}/{benchmark_steps}", flush=True)

    total_time = float(sum(step_times))
    result = {
        "warmup_steps": int(warmup_steps),
        "benchmark_steps": int(benchmark_steps),
        "avg_step_time_s": total_time / max(1, benchmark_steps),
        "tokens_per_sec": token_count / max(total_time, 1e-12),
        "examples_per_sec": (token_count / max(seq_len, 1)) / max(total_time, 1e-12),
        "avg_train_loss": loss_sum / max(1, benchmark_steps),
    }
    if device == "cuda":
        result["max_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    return result


def eval_loop(model, loader, device):
    model.eval()
    ys, logits = [], []
    with torch.no_grad():
        for seq_ids, y in loader:
            seq_ids = seq_ids.to(device, non_blocking=True)
            logit = model(seq_ids)
            ys.append(y.numpy())
            logits.append(logit.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    logits_arr = np.concatenate(logits)
    return compute_metrics(y_true, logits_arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seq_key", default="seq")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_dir", default="runs/seq_baseline")
    ap.add_argument("--early_stop_patience", type=int, default=0)

    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--ffn_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_flex_attention", action="store_true")
    ap.add_argument("--recency_bias", type=float, default=0.0)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--benchmark_only", action="store_true")
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--benchmark_steps", type=int, default=500)
    ap.add_argument("--benchmark_log_every", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = SeqNPZ(os.path.join(args.data_dir, "train.npz"), seq_key=args.seq_key)
    val_ds = SeqNPZ(os.path.join(args.data_dir, "val.npz"), seq_key=args.seq_key)
    test_ds = SeqNPZ(os.path.join(args.data_dir, "test.npz"), seq_key=args.seq_key)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    max_id = int(train_ds.seq.max()) if len(train_ds) else 0
    vocab_size = max_id
    model = SeqCTRModel(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        use_flex_attention=args.use_flex_attention,
        recency_bias=args.recency_bias,
        causal=args.causal,
    ).to(device)

    run_config = {
        "args": vars(args),
        "device": device,
        "rows_train": len(train_ds),
        "rows_val": len(val_ds),
        "rows_test": len(test_ds),
        "vocab_size": vocab_size,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    if args.benchmark_only:
        benchmark = run_benchmark(
            model=model,
            loader=train_loader,
            device=device,
            loss_fn=loss_fn,
            opt=opt,
            warmup_steps=args.warmup_steps,
            benchmark_steps=args.benchmark_steps,
            seq_len=args.seq_len,
            log_every=args.benchmark_log_every,
        )
        with open(os.path.join(args.out_dir, "benchmark.json"), "w") as f:
            json.dump(benchmark, f, indent=2)
        print("BENCHMARK:", benchmark)
        return

    step = 0
    best_auc = -float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = os.path.join(args.out_dir, "model_best.pt")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for seq_ids, y in train_loader:
            seq_ids = seq_ids.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logit = model(seq_ids)
            loss = loss_fn(logit, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 200 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1

        print(f"Epoch {epoch} done in {time.time() - t0:.1f}s")

        val_metrics = eval_loop(model, val_loader, device)
        print("VAL:", val_metrics)
        with open(os.path.join(args.out_dir, f"val_epoch{epoch}.json"), "w") as f:
            json.dump(val_metrics, f, indent=2)

        val_auc = float(val_metrics["auc"])
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            with open(os.path.join(args.out_dir, "best_val.json"), "w") as f:
                json.dump({"epoch": best_epoch, **val_metrics}, f, indent=2)
            print(f"New best AUC {best_auc:.6f} at epoch {best_epoch}; saved {best_path}")
        else:
            bad_epochs += 1
            print(
                f"No val AUC improvement at epoch {epoch} "
                f"(best {best_auc:.6f} @ epoch {best_epoch}); bad_epochs={bad_epochs}"
            )
            if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
                print(f"Early stopping triggered (patience={args.early_stop_patience}).")
                break

    if best_epoch >= 0 and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best checkpoint from epoch {best_epoch} for test evaluation.")

    test_metrics = eval_loop(model, test_loader, device)
    print("TEST:", test_metrics)
    with open(os.path.join(args.out_dir, "test.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    print("Saved model to", args.out_dir)


if __name__ == "__main__":
    main()
