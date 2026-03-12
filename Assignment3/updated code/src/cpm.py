import numpy as np


NBRS = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]


def in_bounds(x, y, nx, ny):
    return 0 <= x < nx and 0 <= y < ny


def make_J():
    J = np.zeros((4, 4), dtype=float)
    MEDIUM, OPC, OLIGO, MICRO = 0, 1, 2, 3
    J[MEDIUM, OPC] = J[OPC, MEDIUM] = 10.0
    J[MEDIUM, OLIGO] = J[OLIGO, MEDIUM] = 12.0
    J[MEDIUM, MICRO] = J[MICRO, MEDIUM] = 11.0
    J[OPC, OLIGO] = J[OLIGO, OPC] = 6.0
    J[OPC, MICRO] = J[MICRO, OPC] = 9.0
    J[OLIGO, MICRO] = J[MICRO, OLIGO] = 9.0
    return J


def chi_G(g, G_sat):
    return g / (1.0 + g / G_sat)


def boundary_energy_at(x, y, sigma_grid, cells, J, MEDIUM):
    sid = int(sigma_grid[x, y])
    stype = MEDIUM if sid == 0 else cells[sid].ctype
    E = 0.0
    nx, ny = sigma_grid.shape
    for dx_, dy_ in NBRS:
        xn, yn = x + dx_, y + dy_
        nid = 0 if not in_bounds(xn, yn, nx, ny) else int(sigma_grid[xn, yn])
        ntype = MEDIUM if nid == 0 else cells[nid].ctype
        if nid != sid:
            E += J[stype, ntype]
    return E


def delta_boundary_energy(x, y, source_id, sigma_grid, cells, J, MEDIUM):
    nx, ny = sigma_grid.shape
    affected = {(x, y)}
    for dx_, dy_ in NBRS:
        xn, yn = x + dx_, y + dy_
        if in_bounds(xn, yn, nx, ny):
            affected.add((xn, yn))

    E_old = sum(boundary_energy_at(a, b, sigma_grid, cells, J, MEDIUM) for a, b in affected)
    old_val = sigma_grid[x, y]
    sigma_grid[x, y] = source_id
    E_new = sum(boundary_energy_at(a, b, sigma_grid, cells, J, MEDIUM) for a, b in affected)
    sigma_grid[x, y] = old_val
    return E_new - E_old


def delta_volume_energy(source_id, target_id, cells, volumes, Vt_cell, lambda_V):
    dH = 0.0
    if target_id != 0:
        ttype = cells[target_id].ctype
        lam = lambda_V[ttype]
        Vt = Vt_cell[target_id]
        Vold = volumes[target_id]
        dH += lam * ((Vold - 1 - Vt) ** 2 - (Vold - Vt) ** 2)

    if source_id != 0:
        stype = cells[source_id].ctype
        lam = lambda_V[stype]
        Vt = Vt_cell[source_id]
        Vold = volumes[source_id]
        dH += lam * ((Vold + 1 - Vt) ** 2 - (Vold - Vt) ** 2)
    return dH


def delta_chemotaxis(x, y, xn, yn, source_id, G_arr, cells, lambda_chem_opc, G_sat, OPC):
    if source_id == 0:
        return 0.0
    stype = cells[source_id].ctype
    if stype != OPC:
        return 0.0
    return -lambda_chem_opc * (chi_G(G_arr[x, y], G_sat) - chi_G(G_arr[xn, yn], G_sat))


def cpm_sweep(sigma, cells, volumes, Vt_cell, G_arr, cfg, J):
    nx, ny = sigma.shape
    for _ in range(cfg.MCS_per_macro):
        x = np.random.randint(0, nx)
        y = np.random.randint(0, ny)
        dx_, dy_ = NBRS[np.random.randint(0, len(NBRS))]
        xn, yn = x + dx_, y + dy_
        if not in_bounds(xn, yn, nx, ny):
            continue

        source_id = int(sigma[xn, yn])
        target_id = int(sigma[x, y])
        if source_id == target_id:
            continue

        dH = 0.0
        dH += delta_boundary_energy(x, y, source_id, sigma, cells, J, 0)
        dH += delta_volume_energy(source_id, target_id, cells, volumes, Vt_cell, cfg.lambda_V)
        dH += delta_chemotaxis(
            x, y, xn, yn, source_id, G_arr, cells,
            cfg.lambda_chem_opc, cfg.G_sat, 1
        )

        if dH <= 0.0 or np.random.rand() < np.exp(-dH / cfg.T):
            sigma[x, y] = source_id
            if target_id != 0:
                volumes[target_id] -= 1
            if source_id != 0:
                volumes[source_id] += 1


def cell_boundary_mask(sigma):
    b = np.zeros_like(sigma, dtype=bool)
    for dx_, dy_ in NBRS:
        shifted = np.roll(np.roll(sigma, dx_, axis=0), dy_, axis=1)
        b |= (shifted != sigma)
    b[[0, -1], :] = True
    b[:, [0, -1]] = True
    return b