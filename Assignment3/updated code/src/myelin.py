import numpy as np


def update_myelin(M, sigma, cells, I_arr, cfg):
    oligo_mask = np.zeros_like(M, dtype=float)
    for cid, cell in cells.items():
        if cell.ctype == 2:
            oligo_mask[sigma == cid] = 1.0

    if cfg.repair_spread:
        rep = oligo_mask.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                rep = np.maximum(rep, np.roll(np.roll(oligo_mask, dx, axis=0), dy, axis=1))
        repair_mask = rep
    else:
        repair_mask = oligo_mask

    dM = cfg.dt_macro_min * (
        cfg.alpha_rep * repair_mask
        - cfg.eta_dem * I_arr * M
    )
    M_new = np.clip(M + dM, 0.0, 1.0)
    return M_new