from copy import deepcopy
from pathlib import Path

import numpy as np

from .config import Config, OPC, OLIGO, MICRO
from .state import count_cells_by_type
from .init import init_lesion_centered, compute_volumes
from .cpm import cpm_sweep, make_J
from .pde import build_mesh, init_fields, solve_pdes, get_fields
from .grn import grn_update_opcs, opc_differentiate_by_grn
from .myelin import update_myelin
from .metrics import (
    update_microglia_activation_from_myelin,
    microglia_activation_percent,
    lesion_metrics,
    mutant_fraction,
)
from .io_utils import save_json, save_timeseries_csv, save_snapshots_npz, ensure_dir


def opc_growth_update(Vt_cell, sigma, cells, I_arr, G_arr, cfg):
    from .init import cell_centroid
    for cid, cell in cells.items():
        if cell.ctype != OPC:
            continue
        cxy = cell_centroid(sigma, cid)
        if cxy is None:
            continue
        x, y = cxy
        inc = cfg.growth_coeff * (G_arr[x, y] / (0.6 + G_arr[x, y])) * (
            1.0 / (1.0 + I_arr[x, y] / cfg.current_K_I_grow)
        )
        Vt_cell[cid] = min(cfg.V_div_target, Vt_cell[cid] + inc)



def split_cell_pixels(sigma, cid):
    pts = np.argwhere(sigma == cid)
    if len(pts) < 16:
        return None
    cx, cy = int(np.round(pts[:, 0].mean())), int(np.round(pts[:, 1].mean()))
    left = pts[pts[:, 0] <= cx]
    right = pts[pts[:, 0] > cx]
    if len(left) < 6 or len(right) < 6:
        left = pts[pts[:, 1] <= cy]
        right = pts[pts[:, 1] > cy]
        if len(left) < 6 or len(right) < 6:
            return None
    return left, right


def add_cell(cells, volumes, Vt_cell, new_type, new_target, parent=None):
    from .state import Cell
    new_id = max(cells.keys()) + 1 if len(cells) > 0 else 1
    if parent is None:
        cells[new_id] = Cell(cid=new_id, ctype=new_type)
    else:
        cells[new_id] = Cell(
            cid=new_id,
            ctype=new_type,
            genotype=parent.genotype,
            r=0.0, p=0.0, q=0.0, p_hold=0
        )
    volumes[new_id] = 0
    Vt_cell[new_id] = float(new_target)
    return new_id


def opc_division_with_mutation(sigma, cells, volumes, Vt_cell, cfg):
    for cid, cell in list(cells.items()):
        if cell.ctype != OPC:
            continue
        if Vt_cell[cid] < cfg.V_div_target:
            continue
        if volumes[cid] < 0.9 * Vt_cell[cid]:
            continue
        split = split_cell_pixels(sigma, cid)
        if split is None:
            continue
        left, right = split
        new_id = add_cell(cells, volumes, Vt_cell, OPC, cfg.V0[OPC], parent=cell)
        if np.random.rand() < cfg.mu:
            cells[new_id].genotype = 1
        sigma[right[:, 0], right[:, 1]] = new_id
        volumes[cid] = int(len(left))
        volumes[new_id] = int(len(right))
        Vt_cell[cid] = float(cfg.V0[OPC])


def run_single(condition_name: str, sI_value: float, seed: int, out_dir: str, cfg: Config):
    np.random.seed(seed)

    cfg = deepcopy(cfg)
    cfg.current_sI = sI_value

    if condition_name == "mild":
        cfg.current_lesion_radius = cfg.lesion_radius_mild
        cfg.current_alpha_q = cfg.alpha_q_mild
        cfg.current_K_I_grow = cfg.K_I_grow_mild
        cfg.current_Ki = cfg.Ki_mild
    elif condition_name == "strong":
        cfg.current_lesion_radius = cfg.lesion_radius_strong
        cfg.current_alpha_q = cfg.alpha_q_strong
        cfg.current_K_I_grow = cfg.K_I_grow_strong
        cfg.current_Ki = cfg.Ki_strong

  
    

    run_dir = Path(out_dir) / condition_name / f"seed_{seed:03d}"
    ensure_dir(run_dir)

    sigma, cells, M, lesion = init_lesion_centered(cfg, seed=seed)
    volumes = compute_volumes(sigma, cells)
    Vt_cell = {cid: float(cfg.V0[cells[cid].ctype]) for cid in cells.keys()}
    micro_active = {cid: 1 for cid, c in cells.items() if c.ctype == MICRO}

    mesh = build_mesh(cfg)
    I, G = init_fields(mesh, cfg)
    J = make_J()

    records = []
    snapshots = {"times": [], "I": [], "G": [], "M": [], "sigma": []}

    tracked_opc = next((cid for cid, c in cells.items() if c.ctype == OPC), None)

    for step in range(cfg.n_macro):
        for _ in range(cfg.pde_substeps):
            solve_pdes(I, G, sigma, cells, micro_active, mesh, cfg)

        I_arr, G_arr = get_fields(I, G, cfg)

        cpm_sweep(sigma, cells, volumes, Vt_cell, G_arr, cfg, J)
        grn_update_opcs(sigma, cells, I_arr, G_arr, cfg)
        opc_differentiate_by_grn(cells, Vt_cell, cfg)
        opc_growth_update(Vt_cell, sigma, cells, I_arr, G_arr, cfg)
        opc_division_with_mutation(sigma, cells, volumes, Vt_cell, cfg)

        M = update_myelin(M, sigma, cells, I_arr, cfg)
        update_microglia_activation_from_myelin(cells, sigma, micro_active, M, threshold=0.6)

        tmin = step * cfg.dt_macro_min
        n_opc, n_ol, n_mg = count_cells_by_type(cells, OPC, OLIGO, MICRO)
        lm = lesion_metrics(M, I_arr, G_arr, lesion)

        rec = {
            "time_min": tmin,
            "mean_I": float(I_arr.mean()),
            "mean_G": float(G_arr.mean()),
            "mean_M": float(M.mean()),
            "n_opc": n_opc,
            "n_oligo": n_ol,
            "n_micro": n_mg,
            "micro_active_pct": microglia_activation_percent(cells, micro_active),
            "mutant_fraction": mutant_fraction(cells),
            **lm,
        }

        if tracked_opc is not None and tracked_opc in cells and cells[tracked_opc].ctype == OPC:
            rec["tracked_r"] = cells[tracked_opc].r
            rec["tracked_p"] = cells[tracked_opc].p
        else:
            rec["tracked_r"] = np.nan
            rec["tracked_p"] = np.nan

        records.append(rec)

        if cfg.save_snapshots and (step % cfg.snapshot_stride == 0 or step == cfg.n_macro - 1):
            snapshots["times"].append(tmin)
            snapshots["I"].append(I_arr.copy())
            snapshots["G"].append(G_arr.copy())
            snapshots["M"].append(M.copy())
            snapshots["sigma"].append(sigma.copy())

    save_json(
        {
            "condition": condition_name,
            "seed": seed,
            "config": cfg.to_dict(),
        },
        run_dir / "meta.json",
    )
    save_timeseries_csv(records, run_dir / "timeseries.csv")

    if cfg.save_snapshots:
        save_snapshots_npz(
            run_dir / "snapshots.npz",
            {
                "times": np.array(snapshots["times"]),
                "I": np.array(snapshots["I"]),
                "G": np.array(snapshots["G"]),
                "M": np.array(snapshots["M"]),
                "sigma": np.array(snapshots["sigma"]),
            },
        )

    final = records[-1]
    summary = {
        "condition": condition_name,
        "seed": seed,
        "final_mean_M_lesion": final["mean_M_lesion"],
        "final_mean_I_lesion": final["mean_I_lesion"],
        "final_mean_G_lesion": final["mean_G_lesion"],
        "final_lesion_area_dem": final["lesion_area_dem"],
        "final_lesion_area_repaired": final["lesion_area_repaired"],
        "final_n_opc": final["n_opc"],
        "final_n_oligo": final["n_oligo"],
        "final_n_micro": final["n_micro"],
        "final_mutant_fraction": final["mutant_fraction"],
        "peak_mean_I": max(r["mean_I"] for r in records),
    }

    return summary