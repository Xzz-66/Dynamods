import numpy as np
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm


def build_mesh(cfg):
    return Grid2D(dx=cfg.dx_mm, dy=cfg.dx_mm, nx=cfg.nx, ny=cfg.ny)


def init_fields(mesh, cfg):
    I = CellVariable(name="I", mesh=mesh, value=cfg.I_init)
    G = CellVariable(name="G", mesh=mesh, value=cfg.G_init)
    return I, G


def get_fields(I, G, cfg):
    I_arr = np.array(I.value).reshape((cfg.nx, cfg.ny), order="F")
    G_arr = np.array(G.value).reshape((cfg.nx, cfg.ny), order="F")
    return I_arr, G_arr


def build_maps_from_cpm(sigma, cells, micro_active, I_arr, G_arr, cfg):
    tg = np.zeros_like(sigma, dtype=np.int32)
    for cid, cell in cells.items():
        tg[sigma == cid] = cell.ctype

    micro = (tg == 3).astype(float)
    oligo = (tg == 2).astype(float)
    opc_mask = (tg == 1)

    A = np.zeros_like(micro)
    for cid, cell in cells.items():
        if cell.ctype == 3 and micro_active.get(cid, 0) == 1:
            A[sigma == cid] = 1.0

    S_I = cfg.current_sI * micro * A
    S_G = cfg.s_G * oligo

    Vmax_map = np.zeros_like(G_arr)
    for cid, cell in cells.items():
        if cell.ctype == 1:
            Vmax_map[sigma == cid] = cfg.Vmax_mut if cell.genotype == 1 else cfg.Vmax_wt

    denom = cfg.Km * (1.0 + I_arr / (cfg.current_Ki + 1e-12)) + G_arr
    U_G = opc_mask.astype(float) * (Vmax_map * G_arr / (denom + 1e-12))
    U_I = cfg.eta_I_clear * opc_mask.astype(float) * I_arr

    return S_I, S_G, U_G, U_I


def solve_pdes(I, G, sigma, cells, micro_active, mesh, cfg):
    I_arr, G_arr = get_fields(I, G, cfg)
    S_I, S_G, U_G, U_I = build_maps_from_cpm(sigma, cells, micro_active, I_arr, G_arr, cfg)

    SI = CellVariable(mesh=mesh, value=S_I.reshape(-1, order="F"))
    SG = CellVariable(mesh=mesh, value=S_G.reshape(-1, order="F"))
    UG = CellVariable(mesh=mesh, value=U_G.reshape(-1, order="F"))
    UI = CellVariable(mesh=mesh, value=U_I.reshape(-1, order="F"))

    eqI = TransientTerm() == DiffusionTerm(coeff=cfg.D_I) + SI - cfg.k_I * I - UI
    eqG = TransientTerm() == DiffusionTerm(coeff=cfg.D_G) + SG - UG - cfg.k_G * G

    eqI.solve(var=I, dt=cfg.dt_pde_min)
    eqG.solve(var=G, dt=cfg.dt_pde_min)