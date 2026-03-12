from dataclasses import dataclass, asdict
from pathlib import Path


MEDIUM = 0
OPC = 1
OLIGO = 2
MICRO = 3
OPC_MUT = 4


@dataclass
class Config:
    # --------------------------
    # Domain / time
    # --------------------------
    nx: int = 50
    ny: int = 50
    dx_um: float = 50.0
    dt_pde_min: float = 0.05
    pde_substeps: int = 10
    n_macro: int = 500
    MCS_per_macro: int = 40000

    # --------------------------
    # Initialization
    # --------------------------
    lesion_radius: int = 9
    lesion_center_x: int = 25
    lesion_center_y: int = 25

    n_opc_ring: int = 15
    n_oligo_outer: int = 8
    n_micro_core: int = 8
    init_cell_radius: int = 2

    M_init_healthy: float = 1.0
    M_init_lesion: float = 0.15
    G_init: float = 0.2
    I_init: float = 0.0

    # --------------------------
    # CPM parameters
    # --------------------------
    T: float = 30.0
    lambda_V_opc: float = 0.8
    lambda_V_oligo: float = 2.5
    lambda_V_micro: float = 2.0

    V0_opc: float = 45.0
    V0_oligo: float = 55.0
    V0_micro: float = 45.0
    V_div_factor: float = 1.4

    # Chemotaxis
    lambda_chem_opc: float = 180.0
    G_sat: float = 0.5

    # Growth
    growth_coeff: float = 3.0
    K_I_grow: float = 0.7

    # --------------------------
    # PDE parameters
    # --------------------------
    D_I: float = 0.03
    D_G: float = 0.02
    k_I: float = 0.0462   # ~ln(2)/15
    k_G: float = 0.0347   # ~ln(2)/20
    eta_I_clear: float = 0.06
    s_G: float = 0.06

    # Inflammation conditions
    sI_mild: float = 0.05
    sI_strong: float = 0.14

    # Michaelis-Menten uptake
    Vmax_wt: float = 0.12
    Vmax_mut: float = 0.192
    Km: float = 0.50
    Ki: float = 0.50

    # --------------------------
    # GRN
    # --------------------------
    alpha_r: float = 1.6
    gamma_r: float = 0.7
    alpha_p: float = 1.0
    gamma_p: float = 0.5

    alpha_q: float = 1.2
    gamma_q: float = 0.8
    K_I_q: float = 0.8
    m_I: int = 2

    K_q: float = 0.8
    h_q: int = 2

    K_G: float = 0.6
    n_hill: int = 3
    baseline_diff_drive: float = 0.07

    p_th: float = 0.4
    tau_hold_steps: int = 6
    mutants_cannot_diff: bool = False

    # --------------------------
    # Myelin field
    # --------------------------
    alpha_rep: float = 0.04
    eta_dem: float = 0.08
    repair_spread: bool = True

    # --------------------------
    # Mutation
    # --------------------------
    mu: float = 0.02

    # --------------------------
    # Output
    # --------------------------
    snapshot_stride: int = 10
    save_snapshots: bool = True
    save_animation_fields: bool = True

    # --------------------------
    # Plot style
    # --------------------------
    title_fs: int = 14
    label_fs: int = 12
    tick_fs: int = 10
    legend_fs: int = 10
    lw: float = 2.0

    @property
    def dx_mm(self) -> float:
        return self.dx_um / 1000.0

    @property
    def dt_macro_min(self) -> float:
        return self.dt_pde_min * self.pde_substeps

    @property
    def V0(self):
        return {
            OPC: self.V0_opc,
            OLIGO: self.V0_oligo,
            MICRO: self.V0_micro,
        }

    @property
    def lambda_V(self):
        return {
            OPC: self.lambda_V_opc,
            OLIGO: self.lambda_V_oligo,
            MICRO: self.lambda_V_micro,
        }

    @property
    def V_div_target(self) -> float:
        return self.V_div_factor * self.V0_opc

    def to_dict(self):
        return asdict(self)