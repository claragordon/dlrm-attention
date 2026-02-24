python -m scripts.train_dlrm \
  --data_dir data/processed \
  --hash_size 131072 \
  --emb_dim 64 \
  --batch_size 4096 \
  --epochs 1 \
  --out_dir runs/dlrm_baseline_day2_2m