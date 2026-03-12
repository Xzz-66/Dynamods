from .init import cell_centroid


def H_G(G, cfg):
    return (G ** cfg.n_hill) / (cfg.K_G ** cfg.n_hill + G ** cfg.n_hill + 1e-12)


def H_I(I, cfg):
    return (I ** cfg.m_I) / (cfg.K_I_q ** cfg.m_I + I ** cfg.m_I + 1e-12)


def grn_update_opcs(sigma, cells, I_arr, G_arr, cfg):
    for cid, cell in cells.items():
        if cell.ctype != 1:
            continue
        cxy = cell_centroid(sigma, cid)
        if cxy is None:
            continue
        x, y = cxy
        g_local = G_arr[x, y]
        i_local = I_arr[x, y]

        dq = cfg.alpha_q * H_I(i_local, cfg) - cfg.gamma_q * cell.q
        cell.q = max(0.0, cell.q + cfg.dt_macro_min * dq)

        inhib = 1.0 / (1.0 + (cell.q / cfg.K_q) ** cfg.h_q)
        drive = cfg.baseline_diff_drive + H_G(g_local, cfg)

        dr = cfg.alpha_r * drive * inhib - cfg.gamma_r * cell.r
        dp = cfg.alpha_p * cell.r - cfg.gamma_p * cell.p

        cell.r = max(0.0, cell.r + cfg.dt_macro_min * dr)
        cell.p = max(0.0, cell.p + cfg.dt_macro_min * dp)
        cell.p_hold = (cell.p_hold + 1) if (cell.p >= cfg.p_th) else 0


def opc_differentiate_by_grn(cells, Vt_cell, cfg):
    for cid, cell in list(cells.items()):
        if cell.ctype != 1:
            continue
        if cfg.mutants_cannot_diff and cell.genotype == 1:
            continue
        if cell.p_hold >= cfg.tau_hold_steps:
            cell.ctype = 2
            Vt_cell[cid] = float(cfg.V0[2])