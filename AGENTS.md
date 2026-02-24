# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Single-script Python project for preprocessing Criteo ad click-through rate data. The entire codebase is `src/preprocess_criteo.py`.

### Dependencies

- Python 3.6+ (3.12 available in the cloud VM)
- `numpy` (only non-stdlib dependency)
- `flake8` for linting, `pytest` for testing
- All listed in `requirements.txt`

### Running the application

```bash
python3 src/preprocess_criteo.py --in_txt <path-to-tsv> --out_dir <output-dir> [--hash_size 131072] [--max_rows 0]
```

The input file must be tab-separated with 40 columns (1 label + 13 dense + 26 sparse features). To generate sample data for testing:

```bash
mkdir -p data && python3 -c "
import random
rows = []
for _ in range(100):
    label = str(random.randint(0, 1))
    dense = [str(random.randint(0, 1000)) if random.random() > 0.1 else '' for _ in range(13)]
    sparse = [format(random.randint(0, 2**32), 'x') if random.random() > 0.15 else '' for _ in range(26)]
    rows.append('\t'.join([label] + dense + sparse))
with open('data/sample_criteo.tsv', 'w') as f:
    f.write('\n'.join(rows) + '\n')
"
```

### Linting

```bash
flake8 src/preprocess_criteo.py --max-line-length=120
```

Note: the existing codebase has pre-existing style issues (E302, E305, E401, W292). These are cosmetic only.

### Testing

No automated test suite exists yet. To verify the script works, run:

```bash
python3 src/preprocess_criteo.py --in_txt data/sample_criteo.tsv --out_dir data/processed --max_rows 100
```

### Gotchas

- `pip install` defaults to `--user` in the cloud VM; ensure `$HOME/.local/bin` is on `PATH` for `flake8`/`pytest` CLI tools.
- The `data/` directory is gitignored; sample data must be regenerated locally.
