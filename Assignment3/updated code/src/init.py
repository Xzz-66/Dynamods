import numpy as np
from typing import Dict, Tuple

from .config import MEDIUM, OPC, OLIGO, MICRO
from .state import Cell


def in_bounds(x, y, nx, ny):
    return 0 <= x < nx and 0 <= y < ny


def make_disk(mask_arr, value, center, radius):
    cx, cy = center
    xs = np.arange(mask_arr.shape[0])[:, None]
    ys = np.arange(mask_arr.shape[1])[None, :]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    mask_arr[mask] = value


def lesion_mask(cfg):
    xs = np.arange(cfg.nx)[:, None]
    ys = np.arange(cfg.ny)[None, :]
    return (
        (xs - cfg.lesion_center_x) ** 2 + (ys - cfg.lesion_center_y) ** 2
        <= cfg.lesion_radius ** 2
    )


def ring_mask(cfg, r_in, r_out):
    xs = np.arange(cfg.nx)[:, None]
    ys = np.arange(cfg.ny)[None, :]
    d2 = (xs - cfg.lesion_center_x) ** 2 + (ys - cfg.lesion_center_y) ** 2
    return (d2 >= r_in ** 2) & (d2 <= r_out ** 2)


def sample_positions_from_mask(mask, n, margin=0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return []
    if n >= len(coords):
        idx = rng.choice(len(coords), size=len(coords), replace=False)
    else:
        idx = rng.choice(len(coords), size=n, replace=False)
    return [tuple(map(int, coords[i])) for i in idx]


def paint_cell_disk_if_free(sigma, cid, center, radius):
    cx, cy = center
    xs = np.arange(sigma.shape[0])[:, None]
    ys = np.arange(sigma.shape[1])[None, :]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    if np.any(sigma[mask] != 0):
        return False
    sigma[mask] = cid
    return True


def init_lesion_centered(cfg, seed=1):
    rng = np.random.default_rng(seed)
    sigma = np.zeros((cfg.nx, cfg.ny), dtype=np.int32)
    cells: Dict[int, Cell] = {}
    cid = 1

    lesion = lesion_mask(cfg)
    opc_ring = ring_mask(cfg, cfg.lesion_radius + 1, cfg.lesion_radius + 8)
    outer_ring = ring_mask(cfg, cfg.lesion_radius + 6, cfg.lesion_radius + 15)

    # myelin field
    M = np.full((cfg.nx, cfg.ny), cfg.M_init_healthy, dtype=float)
    M[lesion] = cfg.M_init_lesion

    # candidate positions
    micro_pos = sample_positions_from_mask(lesion, cfg.n_micro_core * 3, rng=rng)
    opc_pos = sample_positions_from_mask(opc_ring, cfg.n_opc_ring * 4, rng=rng)
    oligo_pos = sample_positions_from_mask(outer_ring, cfg.n_oligo_outer * 4, rng=rng)

    # place microglia in lesion core
    placed = 0
    for pos in micro_pos:
        if placed >= cfg.n_micro_core:
            break
        if paint_cell_disk_if_free(sigma, cid, pos, cfg.init_cell_radius):
            cells[cid] = Cell(cid=cid, ctype=MICRO)
            cid += 1
            placed += 1

    # place OPCs around lesion
    placed = 0
    for pos in opc_pos:
        if placed >= cfg.n_opc_ring:
            break
        if paint_cell_disk_if_free(sigma, cid, pos, cfg.init_cell_radius):
            cells[cid] = Cell(cid=cid, ctype=OPC, genotype=0)
            cid += 1
            placed += 1

    # place Oligos outside lesion
    placed = 0
    for pos in oligo_pos:
        if placed >= cfg.n_oligo_outer:
            break
        if paint_cell_disk_if_free(sigma, cid, pos, cfg.init_cell_radius):
            cells[cid] = Cell(cid=cid, ctype=OLIGO)
            cid += 1
            placed += 1

    return sigma, cells, M, lesion


def compute_volumes(sigma, cells):
    vols = {cid: 0 for cid in cells.keys()}
    ids, counts = np.unique(sigma, return_counts=True)
    for i, c in zip(ids, counts):
        if i != 0 and int(i) in vols:
            vols[int(i)] = int(c)
    return vols


def sigma_to_type_grid(sigma, cells, MEDIUM, OPC, OLIGO, MICRO, OPC_MUT=None):
    tg = np.zeros_like(sigma, dtype=np.int32)
    for cid, cell in cells.items():
        tg[sigma == cid] = cell.ctype
    return tg


def sigma_to_display_grid(sigma, cells, MEDIUM, OPC, OLIGO, MICRO, OPC_MUT):
    tg = np.zeros_like(sigma, dtype=np.int32)
    for cid, cell in cells.items():
        if cell.ctype == OPC and cell.genotype == 1:
            tg[sigma == cid] = OPC_MUT
        else:
            tg[sigma == cid] = cell.ctype
    return tg


def cell_centroid(sigma, cid):
    pts = np.argwhere(sigma == cid)
    if len(pts) == 0:
        return None
    return (int(np.round(pts[:, 0].mean())), int(np.round(pts[:, 1].mean())))