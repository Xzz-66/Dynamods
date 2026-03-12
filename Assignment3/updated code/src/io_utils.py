import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_timeseries_csv(records, path):
    ensure_dir(Path(path).parent)
    pd.DataFrame(records).to_csv(path, index=False)


def save_summary_csv(records, path):
    ensure_dir(Path(path).parent)
    pd.DataFrame(records).to_csv(path, index=False)


def save_snapshots_npz(path, snapshots):
    ensure_dir(Path(path).parent)
    np.savez_compressed(path, **snapshots)