import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import root


class Viterbi:
    """
    Viterbi algorithm

    Parameters
    ----------
    init_prob : array_like, shape (N,)
        Initial state probabilities π.
    trans_prob : array_like, shape (N, N)
        Transition matrix A.
    emis_prob : array_like, shape (N, M)
        Emission matrix B.
    state_names : list[str] | None
        Optional names for states, used in formatted output.
    obs_names : list[str] | None
        Optional names for observations, used in formatted output.
    """

    def __init__(self, init_prob, trans_prob, emis_prob, state_names=None, obs_names=None):
        self.init_prob = np.asarray(init_prob, dtype=float)
        self.trans_prob = np.asarray(trans_prob, dtype=float)
        self.emis_prob = np.asarray(emis_prob, dtype=float)
        self.state_names = state_names
        self.obs_names = obs_names

    def run(self, observation):
        """
        Run Viterbi to compute the most likely hidden state sequence.

        Parameters
        ----------
        observation : array_like, shape (T,)
            Sequence of observation indices in {0,1,...,M-1}.

        Returns
        -------
        path : np.ndarray, shape (T,)
            Most likely state indices.
        path_prob : float
            Probability of the returned path.
        delta : np.ndarray, shape (T, N)
            Viterbi dynamic programming table.
        psi : np.ndarray, shape (T, N)
            Backpointer table
        """
        obs = np.asarray(observation, dtype=int).ravel()
        T = obs.size
        N = self.init_prob.size

        delta = np.zeros((T, N), dtype=float)
        psi = np.full((T, N), fill_value=-1, dtype=int)

        o0 = obs[0]
        delta[0, :] = self.init_prob * self.emis_prob[:, o0]

        for t in range(1, T):
            ot = obs[t]
            prev = delta[t - 1, :][:, None] * self.trans_prob
            psi[t, :] = np.argmax(prev, axis=0)
            delta[t, :] = np.max(prev, axis=0) * self.emis_prob[:, ot]

        last_state = int(np.argmax(delta[T - 1, :]))
        path_prob = float(delta[T - 1, last_state])

        path = np.zeros(T, dtype=int)
        path[T - 1] = last_state
        for t in range(T - 1, 0, -1):
            path[t - 1] = psi[t, path[t]]

        return path, path_prob, delta, psi

    def format_tables(self, delta, psi, last_k=None, transpose=True, precision=6):
        """
        Format delta/psi tables for printing.

        Parameters
        ----------
        delta : np.ndarray, shape (T,N)
        psi : np.ndarray, shape (T,N)
        last_k : int | None
            If given, show only the last_k time steps.
        transpose : bool
            If True, output is (states x time). If False, (time x states).
        precision : int
            Decimal digits for delta.

        Returns
        -------
        delta_str : str
        psi_str : str
        """
        d = np.asarray(delta, float)
        p = np.asarray(psi, int)

        if last_k is not None:
            d = d[-last_k:, :]
            p = p[-last_k:, :]

        if transpose:
            d = d.T
            p = p.T

        state_labels = self.state_names if self.state_names is not None else [f"s{i}" for i in range(d.shape[0])]
        time_labels = [f"t{i+1}" for i in range(d.shape[1])]

        d_fmt = np.vectorize(lambda x: f"{x:.{precision}g}")(d)
        p_fmt = p.astype(str)

        delta_lines = ["delta (states x time):" if transpose else "delta (time x states):"]
        psi_lines = ["psi (states x time):" if transpose else "psi (time x states):"]

        delta_lines.append("      " + "  ".join(f"{t:>10}" for t in time_labels))
        for i, s in enumerate(state_labels):
            delta_lines.append(f"{s:>4}  " + "  ".join(f"{v:>10}" for v in d_fmt[i, :]))

        psi_lines.append("      " + "  ".join(f"{t:>10}" for t in time_labels))
        for i, s in enumerate(state_labels):
            psi_lines.append(f"{s:>4}  " + "  ".join(f"{v:>10}" for v in p_fmt[i, :]))

        return "\n".join(delta_lines), "\n".join(psi_lines)


def hill_activation(p, theta, n, eps=1e-12):
    """
    Hill activation function h_+(p;theta,n) = p^n / (theta^n + p^n).
    """
    p = np.maximum(np.asarray(p, float), 0.0)
    return (p**n) / (theta**n + p**n + eps)


def hill_repression(p, theta, n, eps=1e-12):
    """
    Hill repression function h_-(p;theta,n) = theta^n / (theta^n + p^n).
    """
    p = np.maximum(np.asarray(p, float), 0.0)
    return (theta**n) / (theta**n + p**n + eps)


def simulate_mech2_sde(params, T=10.0, dt=0.01, seed=0, clamp_nonneg=True):
    """
    Mechanism II (SDEVelo) simulation.

    Parameters
    ----------
    params : dict
        Required keys:
        aA,aB,mA,mB,betaA,betaB,gammaA,gammaB,nA,nB,thetaA,thetaB,
        kPA,kPB,deltaPA,deltaPB,sigma1A,sigma2A,sigma1B,sigma2B,
        UA0,SA0,UB0,SB0,pA0,pB0
    T : float
        Simulation horizon (s).
    dt : float
        Time step (s).
    seed : int
        RNG seed.
    clamp_nonneg : bool
        If True, restricts all concentrations to be nonnegative each step.

    Returns
    -------
    t : np.ndarray, shape (N+1,)
    X : np.ndarray, shape (N+1, 6)
        Columns are [UA, SA, UB, SB, pA, pB].
    """
    rng = np.random.default_rng(seed)
    N = int(np.ceil(T / dt))
    t = np.linspace(0.0, N * dt, N + 1)
    sqrt_dt = np.sqrt(dt)

    X = np.zeros((N + 1, 6), dtype=float)
    X[0, :] = [
        params["UA0"], params["SA0"], params["UB0"], params["SB0"], params["pA0"], params["pB0"]
    ]

    for k in range(N):
        UA, SA, UB, SB, pA, pB = X[k, :]

        act_A = hill_activation(pB, params["thetaB"], params["nB"])
        rep_B = hill_repression(pA, params["thetaA"], params["nA"])

        alphaA = params["aA"] + params["mA"] * act_A
        alphaB = params["aB"] + params["mB"] * rep_B

        betaA_star = params["betaA"] * act_A
        betaB_star = params["betaB"] * rep_B

        xi = rng.normal(size=4)

        UA_next = UA + (alphaA - betaA_star * UA) * dt + params["sigma1A"] * sqrt_dt * xi[0]
        SA_next = SA + (betaA_star * UA - params["gammaA"] * SA) * dt + params["sigma2A"] * sqrt_dt * xi[1]

        UB_next = UB + (alphaB - betaB_star * UB) * dt + params["sigma1B"] * sqrt_dt * xi[2]
        SB_next = SB + (betaB_star * UB - params["gammaB"] * SB) * dt + params["sigma2B"] * sqrt_dt * xi[3]

        pA_next = pA + (params["kPA"] * SA - params["deltaPA"] * pA) * dt
        pB_next = pB + (params["kPB"] * SB - params["deltaPB"] * pB) * dt

        X[k + 1, :] = [UA_next, SA_next, UB_next, SB_next, pA_next, pB_next]

        if clamp_nonneg:
            X[k + 1, :] = np.maximum(X[k + 1, :], 0.0)

    return t, X


def ensemble_stats(sim_fn, params, T=10.0, dt=0.01, n_runs=200, seed=0):
    """
    Run an ensemble of simulations and return mean/std trajectories.

    Returns
    -------
    t : np.ndarray
    mean : np.ndarray, shape (N+1, d)
    std : np.ndarray, shape (N+1, d)
    """
    rng = np.random.default_rng(seed)
    runs = []
    t_ref = None

    for _ in range(n_runs):
        s = int(rng.integers(1_000_000_000))
        t, X = sim_fn(params, T=T, dt=dt, seed=s)
        if t_ref is None:
            t_ref = t
        runs.append(X)

    stacked = np.stack(runs, axis=0)
    return t_ref, stacked.mean(axis=0), stacked.std(axis=0)


def ode_mech1_baseline(z, t, params):
    """
    Mechanism I baseline (4D ODE): [rA, rB, pA, pB].
    """
    rA, rB, pA, pB = z

    act_A = hill_activation(pB, params["theta_b"], params["nb"])
    rep_B = hill_repression(pA, params["theta_a"], params["na"])

    drA = params["ma"] * act_A - params["gamma_a"] * rA
    drB = params["mb"] * rep_B - params["gamma_b"] * rB
    dpA = params["k_pa"] * rA - params["delta_pa"] * pA
    dpB = params["k_pb"] * rB - params["delta_pb"] * pB

    return [drA, drB, dpA, dpB]


def ode_mech1_disabled_repression(z, t, params):
    """
    Mechanism I disabled repression (4D ODE): [rA, rB, pA, pB].
    """
    rA, rB, pA, pB = z

    act_A = hill_activation(pB, params["theta_b"], params["nb"])

    drA = params["ma"] * act_A - params["gamma_a"] * rA
    drB = params["mb"] - params["gamma_b"] * rB
    dpA = params["k_pa"] * rA - params["delta_pa"] * pA
    dpB = params["k_pb"] * rB - params["delta_pb"] * pB

    return [drA, drB, dpA, dpB]


def ode_mech2_drift(z, t, params):
    """
    Mechanism II ODE
    """
    UA, SA, UB, SB, pA, pB = z

    act_A = hill_activation(pB, params["thetaB"], params["nB"])
    rep_B = hill_repression(pA, params["thetaA"], params["nA"])

    alphaA = params["aA"] + params["mA"] * act_A
    alphaB = params["aB"] + params["mB"] * rep_B

    betaA_star = params["betaA"] * act_A
    betaB_star = params["betaB"] * rep_B

    dUA = alphaA - betaA_star * UA
    dSA = betaA_star * UA - params["gammaA"] * SA
    dUB = alphaB - betaB_star * UB
    dSB = betaB_star * UB - params["gammaB"] * SB
    dpA = params["kPA"] * SA - params["deltaPA"] * pA
    dpB = params["kPB"] * SB - params["deltaPB"] * pB

    return [dUA, dSA, dUB, dSB, dpA, dpB]


def find_fixed_point(ode_fn, x0, ode_args=(), method="hybr"):
    """
    Find a fixed point x*.

    Returns
    -------
    x_star : np.ndarray
    drift_at_star : np.ndarray
    solver : OptimizeResult
    """
    x0 = np.asarray(x0, float)

    def f(x):
        return np.asarray(ode_fn(x, 0.0, *ode_args), float)

    solver = root(f, x0, method=method)
    if not solver.success:
        raise RuntimeError(f"Fixed point solve failed: {solver.message}")

    x_star = solver.x
    return x_star, f(x_star), solver


def jacobian_finite_diff(ode_fn, x, ode_args=(), eps=1e-6):
    """
    Finite-difference Jacobian of f(x)=ode_fn(x, t0, *ode_args) at x.
    """
    x = np.asarray(x, float)

    def f(z):
        return np.asarray(ode_fn(z, 0.0, *ode_args), float)

    f0 = f(x)
    J = np.zeros((x.size, x.size), dtype=float)
    for j in range(x.size):
        xp = x.copy()
        xp[j] += eps
        J[:, j] = (f(xp) - f0) / eps
    return J


def plot_time_series_4d(t, sol, title):
    """
    Plot Mechanism I time series for [rA, rB, pA, pB].
    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax[0].plot(t, sol[:, 0], label=r"$r_A$")
    ax[0].plot(t, sol[:, 1], label=r"$r_B$")
    ax[0].set_ylabel("mRNA concentration (M)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, sol[:, 2], label=r"$p_A$")
    ax[1].plot(t, sol[:, 3], label=r"$p_B$")
    ax[1].set_ylabel("Protein concentration (M)")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig

def plot_time_series_mech2_6d(t, sol, title):
    """
    Plot Mechanism II ODE time series for
    [UA, SA, UB, SB, pA, pB].
    """
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    ax[0].plot(t, sol[:, 0], label=r"$U_A$")
    ax[0].plot(t, sol[:, 2], label=r"$U_B$")
    ax[0].set_ylabel("Unspliced mRNA (M)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, sol[:, 1], label=r"$S_A$")
    ax[1].plot(t, sol[:, 3], label=r"$S_B$")
    ax[1].set_ylabel("Spliced mRNA (M)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(t, sol[:, 4], label=r"$p_A$")
    ax[2].plot(t, sol[:, 5], label=r"$p_B$")
    ax[2].set_ylabel("Protein (M)")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig
    

def plot_ensemble_mech2_mrna(t, mean, std, gene_label):
    """
    Plot Mechanism II ensemble mean±std for unspliced/spliced mRNA.
    """
    fig = plt.figure(figsize=(8, 4))
    UA = mean[:, 0] if gene_label == "A" else mean[:, 2]
    SA = mean[:, 1] if gene_label == "A" else mean[:, 3]
    UA_s = std[:, 0] if gene_label == "A" else std[:, 2]
    SA_s = std[:, 1] if gene_label == "A" else std[:, 3]

    plt.plot(t, UA, label=rf"$U_{gene_label}$ mean")
    plt.fill_between(t, UA - UA_s, UA + UA_s, alpha=0.2)
    plt.plot(t, SA, label=rf"$S_{gene_label}$ mean")
    plt.fill_between(t, SA - SA_s, SA + SA_s, alpha=0.2)

    plt.xlabel("Time (s)")
    plt.ylabel("Concentration (M)")
    plt.title(f"Mechanism II (SDE): Gene {gene_label} mRNA dynamics (mean ± std)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_premrna_ratio(t, mean, gene="B"):
    """
    Plot pre-mRNA accumulation ratio U/S for a chosen gene from ensemble mean.
    """
    if gene == "A":
        U = mean[:, 0]
        S = mean[:, 1]
    else:
        U = mean[:, 2]
        S = mean[:, 3]

    ratio = U / (S + 1e-12)

    fig = plt.figure(figsize=(8, 4))
    plt.plot(t, ratio, label=rf"$U_{gene}/S_{gene}$")
    plt.xlabel("Time (s)")
    plt.ylabel("Ratio")
    plt.title(f"Mechanism II: pre-mRNA accumulation ratio for Gene {gene}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_phase_portrait_mech1(ode_fn, x0, params, title, pa_max=2.5, pb_max=2.5, grid_n=25, tmax=50.0, nt=2000):
    """
    Protein phase portrait (pA,pB) for Mechanism I, freezing (rA,rB) at their fixed-point values.
    """
    x_star, drift_star, _ = find_fixed_point(ode_fn, x0, ode_args=(params,))
    J = jacobian_finite_diff(ode_fn, x_star, ode_args=(params,))
    eig = np.linalg.eigvals(J)

    ra0, rb0 = x_star[0], x_star[1]

    pa = np.linspace(0.0, pa_max, grid_n)
    pb = np.linspace(0.0, pb_max, grid_n)
    PA, PB = np.meshgrid(pa, pb)

    U = np.zeros_like(PA)
    V = np.zeros_like(PB)

    for i in range(PA.shape[0]):
        for j in range(PA.shape[1]):
            z = [ra0, rb0, PA[i, j], PB[i, j]]
            dz = ode_fn(z, 0.0, params)
            U[i, j] = dz[2]
            V[i, j] = dz[3]

    t = np.linspace(0.0, tmax, nt)
    sol = odeint(ode_fn, x0, t, args=(params,))

    fig = plt.figure(figsize=(6, 6))
    plt.quiver(PA, PB, U, V, angles="xy")
    plt.plot(sol[:, 2], sol[:, 3], lw=2, label="trajectory")
    plt.scatter(sol[0, 2], sol[0, 3], s=40, label="start")
    plt.scatter(sol[-1, 2], sol[-1, 3], s=40, label="end")
    plt.scatter(x_star[2], x_star[3], s=120, marker="X", label="fixed point")
    plt.xlabel(r"$p_A$ (M)")
    plt.ylabel(r"$p_B$ (M)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    stats = {
        "fixed_point": x_star,
        "drift_norm": float(np.linalg.norm(drift_star)),
        "eigenvalues": eig,
        "stable": bool(np.all(np.real(eig) < 0)),
    }
    return fig, stats


def plot_phase_portrait_mech2(ode_fn, x0, params, title, pa_max=3.0, pb_max=3.0, grid_n=25, tmax=200.0, nt=5000):
    """
    Protein phase portrait (pA,pB) for Mechanism II drift, freezing (UA,SA,UB,SB) at their fixed-point values.
    """
    x_star, drift_star, _ = find_fixed_point(ode_fn, x0, ode_args=(params,))
    J = jacobian_finite_diff(ode_fn, x_star, ode_args=(params,))
    eig = np.linalg.eigvals(J)

    UA0, SA0, UB0, SB0 = x_star[0], x_star[1], x_star[2], x_star[3]

    pa = np.linspace(0.0, pa_max, grid_n)
    pb = np.linspace(0.0, pb_max, grid_n)
    PA, PB = np.meshgrid(pa, pb)

    U = np.zeros_like(PA)
    V = np.zeros_like(PB)

    for i in range(PA.shape[0]):
        for j in range(PA.shape[1]):
            z = [UA0, SA0, UB0, SB0, PA[i, j], PB[i, j]]
            dz = ode_fn(z, 0.0, params)
            U[i, j] = dz[4]
            V[i, j] = dz[5]

    t = np.linspace(0.0, tmax, nt)
    sol = odeint(ode_fn, x0, t, args=(params,))

    fig = plt.figure(figsize=(6, 6))
    plt.quiver(PA, PB, U, V, angles="xy")
    plt.plot(sol[:, 4], sol[:, 5], lw=2, label="trajectory")
    plt.scatter(sol[0, 4], sol[0, 5], s=40, label="start")
    plt.scatter(sol[-1, 4], sol[-1, 5], s=40, label="end")
    plt.scatter(x_star[4], x_star[5], s=120, marker="X", label="fixed point")
    plt.xlabel(r"$p_A$ (M)")
    plt.ylabel(r"$p_B$ (M)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    stats = {
        "fixed_point": x_star,
        "drift_norm": float(np.linalg.norm(drift_star)),
        "eigenvalues": eig,
        "stable": bool(np.all(np.real(eig) < 0)),
    }
    return fig, stats


def run_all():
    """
    Run all code.
    """
    emission = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.40, 0.40, 0.05, 0.15],
    ])
    transition = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
    ])
    initial = [0.5, 0.5]

    model = Viterbi(initial, transition, emission, state_names=["E", "I"], obs_names=["a", "u", "g", "c"])

    patient_alpha = [0, 2, 3, 2, 3]
    patient_beta = [0, 1, 1, 0, 1]

    for name, obs in [("Patient Alpha", patient_alpha), ("Patient Beta", patient_beta)]:
        path, prob, delta, psi = model.run(obs)
        d_str, p_str = model.format_tables(delta, psi, last_k=4, transpose=True, precision=8)
        print(f"\n{name}")
        print("path:", path[-4:].tolist())
        print("path_prob:", prob)
        print(d_str)
        print(p_str) 

    sde_params = dict(
        aA=1.0, aB=0.25,
        mA=2.35, mB=2.35,
        betaA=2.35, betaB=2.35,
        gammaA=1.0, gammaB=1.0,
        nA=3, nB=3,
        thetaA=0.21, thetaB=0.21,
        kPA=1.0, kPB=1.0,
        deltaPA=1.0, deltaPB=1.0,
        sigma1A=0.05, sigma2A=0.05,
        sigma1B=0.05, sigma2B=0.05,
        UA0=0.8, SA0=0.8,
        UB0=0.8, SB0=0.8,
        pA0=0.8, pB0=0.8,
    )

    t_sde, mean, std = ensemble_stats(simulate_mech2_sde, sde_params, T=100.0, dt=0.01, n_runs=100, seed=0)

    plot_ensemble_mech2_mrna(t_sde, mean, std, "A")
    plot_ensemble_mech2_mrna(t_sde, mean, std, "B")
    plot_premrna_ratio(t_sde, mean, gene="B")

    ode_params = dict(
        ma=2.35, mb=2.35,
        na=3.0, nb=3.0,
        gamma_a=1.0, gamma_b=1.0,
        theta_a=0.21, theta_b=0.21,
        delta_pa=1.0, delta_pb=1.0,
        k_pa=1.0, k_pb=1.0
    )

    z0 = [0.8, 0.8, 0.8, 0.8]
    t_ode = np.linspace(0.0, 20.0, 400)

    baseline = odeint(ode_mech1_baseline, z0, t_ode, args=(ode_params,))
    plot_time_series_4d(t_ode, baseline, "Mechanism I baseline: time series")

    mech1 = odeint(ode_mech1_disabled_repression, z0, t_ode, args=(ode_params,))
    plot_time_series_4d(t_ode, mech1, "Mechanism I disabled repression: time series")

    plot_phase_portrait_mech1(ode_mech1_baseline, z0, ode_params, "Mechanism I baseline: protein phase portrait")
    plot_phase_portrait_mech1(ode_mech1_disabled_repression, z0, ode_params, "Mechanism I disabled repression: protein phase portrait")

    x0 = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    plot_phase_portrait_mech2(ode_mech2_drift, x0, sde_params, "Mechanism II drift: protein phase portrait")

    plt.show()


if __name__ == "__main__":
    run_all()