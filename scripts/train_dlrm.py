import argparse
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.data import CriteoNPZ
from src.model_dlrm import DLRM
from src.metrics import compute_metrics

def eval_loop(model, loader, device):
    model.eval()
    ys, logits = [], []
    with torch.no_grad():
        for dense, sparse, y in loader:
            dense = dense.to(device)
            sparse = sparse.to(device)
            logit = model(dense, sparse)
            ys.append(y.numpy())
            logits.append(logit.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    logits_arr = np.concatenate(logits)
    return compute_metrics(y_true, logits_arr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--hash_size", type=int, default=131072)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--use_attention", action="store_false")
    ap.add_argument("--attention_heads", type=int, default=4)
    ap.add_argument("--out_dir", default="runs/baseline")
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop if val AUC does not improve for this many epochs (0 disables).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = CriteoNPZ(os.path.join(args.data_dir, "train.npz"))
    val_ds   = CriteoNPZ(os.path.join(args.data_dir, "val.npz"))
    test_ds  = CriteoNPZ(os.path.join(args.data_dir, "test.npz"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = DLRM(
        hash_size=args.hash_size,
        emb_dim=args.emb_dim,
        use_attention=args.use_attention,
        attention_heads=args.attention_heads,
    ).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    step = 0
    best_auc = -float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = os.path.join(args.out_dir, "model_best.pt")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for dense, sparse, y in train_loader:
            dense = dense.to(device, non_blocking=True)
            sparse = sparse.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logit = model(dense, sparse)
            loss = loss_fn(logit, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 200 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1

        print(f"Epoch {epoch} done in {time.time()-t0:.1f}s")

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