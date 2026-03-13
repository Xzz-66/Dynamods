import numpy as np


def differentiation_ratio(sigma, cells):
    tg = np.zeros_like(sigma, dtype=np.int32)
    for cid, cell in cells.items():
        tg[sigma == cid] = cell.ctype
    opc_px = float((tg == 1).sum())
    ol_px = float((tg == 2).sum())
    denom = opc_px + ol_px
    return 0.0 if denom <= 0 else ol_px / denom


def update_microglia_activation_from_myelin(cells, sigma, micro_active, M, threshold=0.6):
    for cid, cell in cells.items():
        if cell.ctype != 3:
            continue
        pts = np.argwhere(sigma == cid)
        if len(pts) == 0:
            micro_active[cid] = 0
            continue
        local_M = float(M[pts[:, 0], pts[:, 1]].mean())
        micro_active[cid] = 1 if local_M < threshold else 0


def microglia_activation_percent(cells, micro_active):
    mg_ids = [cid for cid, c in cells.items() if c.ctype == 3]
    if len(mg_ids) == 0:
        return 0.0
    return 100.0 * sum(micro_active.get(cid, 0) for cid in mg_ids) / len(mg_ids)


def lesion_metrics(M, I_arr, G_arr, lesion_mask):
    lesion_M = M[lesion_mask]
    lesion_I = I_arr[lesion_mask]
    lesion_G = G_arr[lesion_mask]

    return {
        "mean_M_lesion": float(lesion_M.mean()),
        "mean_I_lesion": float(lesion_I.mean()),
        "mean_G_lesion": float(lesion_G.mean()),
        "lesion_area_dem": int((lesion_M < 0.5).sum()),
        "lesion_area_repaired": int((lesion_M > 0.8).sum()),
    }


def mutant_fraction_opc(cells, opc_type=1):
    """
    Fraction of OPCs that are mutants.
    """
    opc_ids = [cid for cid, c in cells.items() if c.ctype == opc_type]
    if len(opc_ids) == 0:
        return 0.0
    n_mut = sum(1 for cid in opc_ids if cells[cid].genotype == 1)
    return n_mut / len(opc_ids)

def count_cells_in_region(sigma, cells, region_mask, cell_type):
    """
    Count how many distinct cells of a given type overlap a region mask.
    """
    count = 0
    for cid, cell in cells.items():
        if cell.ctype != cell_type:
            continue
        if np.any(region_mask & (sigma == cid)):
            count += 1
    return count


def count_mutant_opcs(cells, opc_type=1):
    """
    Count mutant OPC cells.
    """
    return sum(1 for c in cells.values() if c.ctype == opc_type and c.genotype == 1)


def outside_lesion_metrics(M, I_arr, G_arr, lesion_mask):
    """
    Means outside the lesion region.
    """
    outside = ~lesion_mask

    return {
        "mean_M_outside": float(M[outside].mean()),
        "mean_I_outside": float(I_arr[outside].mean()),
        "mean_G_outside": float(G_arr[outside].mean()),
    }