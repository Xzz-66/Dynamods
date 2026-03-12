from pathlib import Path
import pandas as pd
from tqdm import tqdm

from .runner import run_single
from .io_utils import save_summary_csv


def run_ensemble(condition_name, sI_value, seeds, out_dir, cfg):
    summaries = []
    for seed in tqdm(seeds, desc=f"{condition_name} ensemble"):
        summaries.append(run_single(condition_name, sI_value, seed, out_dir, cfg))

    cond_dir = Path(out_dir) / condition_name
    save_summary_csv(summaries, cond_dir / "ensemble_summary.csv")
    return pd.DataFrame(summaries)