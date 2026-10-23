import numpy as np
import networkx as nx
from scipy.optimize import minimize_scalar, brentq


def kvec_N2_from_a(a, gamma, kappa=1.0):
    """Build the N=2 capacity vector [a, a, b, b] from the resource constraint."""
    if a < 0:
        return None
    x = (kappa**gamma)/2.0 - a**gamma
    if x < 0:
        return None
    b = x**(1.0/gamma)
    return np.array([a, a, b, b], dtype=float)

def D_of_a_N2(a, gamma, mu, beta, kappa=1.0):
    """Evaluate the expected dissipation for the N=2 one-parameter reduction."""

    N = 2
    E = build_ring_incidence(N)
    Sigma = compute_ring_second_moment_matrix(mu, beta, N)
    k = kvec_N2_from_a(a, gamma, kappa=kappa)

    if k is None:
        return np.nan

    return expected_dissipation_ring_from_incidence(E, k, Sigma)

def global_optimum_N2_piecewise(gamma, kappa=1.0):
    """Return the analytic global optimum of the N=2, mu=0 problem as a piecewise branch."""
    if gamma >= 1.0:
        a = kappa * (4.0 ** (-1.0 / gamma))
        b = a
    elif gamma < 1.0:
        a = 0.0
        b = kappa * (2.0 ** (-1.0 / gamma))
    return a, b

def kvec_N3_from_ab(a, b, gamma, kappa=1.0):
    """Build the N=3 capacity vector [a, a, b, b, c, c] from the resource constraint."""
    if a < 0 or b < 0:
        return None
    x = (kappa**gamma)/2.0 - a**gamma - b**gamma
    if x < 0:
        return None
    c = x**(1.0/gamma)
    return np.array([a, a, b, b, c, c], dtype=float)

def meaningful_minima_N3(gamma, mu, beta, kappa=1.0, n_grid=300, eps_frac=0.02):
    """Find the interior and boundary local minima of the N=3 reduced dissipation landscape."""
    N = 3
    Sigma = compute_ring_second_moment_matrix(mu, beta, N)
    E = build_ring_incidence(N)

    a_max = ((kappa**gamma)/2.0)**(1.0/gamma)
    a_vals = np.linspace(0.0, a_max, n_grid)
    b_vals = a_vals

    # Exclude corners and points too close to the boundary where minima become ambiguous
    eps = eps_frac * a_max
    if eps <= 0:
        eps = 1e-12

    def D_ab(a, b):
        k = kvec_N3_from_ab(a, b, gamma, kappa=kappa)
        if k is None:
            return np.nan
        return expected_dissipation_ring_from_incidence(E, k, Sigma)

    # Search for the best interior minimum with all three branch capacities positive
    best_int = None
    for a in a_vals:
        if a <= eps:
            continue
        for b in b_vals:
            if b <= eps:
                continue
            x = (kappa**gamma)/2.0 - a**gamma - b**gamma
            if x <= 0:
                continue
            c = x**(1.0/gamma)
            if c <= eps:
                continue
            D = D_ab(a, b)
            if not np.isfinite(D):
                continue
            if best_int is None or D < best_int[2]:
                best_int = (float(a), float(b), float(D))

    # Boundary a=0, exclude corners: b in [eps, a_max-eps]
    bs = np.linspace(eps, a_max - eps, n_grid)
    Ds = np.array([D_ab(0.0, b) for b in bs])
    j = np.nanargmin(Ds)
    min_a0 = (0.0, float(bs[j]), float(Ds[j]))

    # Boundary b=0, exclude corners: a in [eps, a_max-eps]
    aas = np.linspace(eps, a_max - eps, n_grid)
    Ds = np.array([D_ab(a, 0.0) for a in aas])
    i = np.nanargmin(Ds)
    min_b0 = (float(aas[i]), 0.0, float(Ds[i]))

    # Boundary c=0, i.e., the curved constraint edge in the (a,b) plane
    aas = np.linspace(eps, a_max - eps, n_grid)
    bs_curve = ((kappa**gamma)/2.0 - aas**gamma)**(1.0/gamma)
    Ds = np.array([D_ab(a, b) for a, b in zip(aas, bs_curve)])
    i = np.nanargmin(Ds)
    min_c0 = (float(aas[i]), float(bs_curve[i]), float(Ds[i]))

    return best_int, {"a0": min_a0, "b0": min_b0, "c0": min_c0}

def global_candidates_from_minima(candidates, use_log=True, atol_log=5e-3, rtol=1e-10):
    """Select all candidates that are numerically indistinguishable from the global minimum."""
    vals = np.array([c[3] for c in candidates], float)

    if use_log:
        lv = np.log(vals)
        lmin = np.min(lv)
        global_mask = lv <= lmin + atol_log
    else:
        vmin = np.min(vals)
        tol = rtol * max(1.0, abs(vmin))
        global_mask = np.abs(vals - vmin) <= tol

    return [c for c, m in zip(candidates, global_mask) if m]

def compute_ring_second_moment_matrix(mu, beta, N):
    """Build the second-moment matrix of nodal injections for the ring model,
    where the number of generators and consumers is equal."""
    total_nodes = 2 * N 
    matrix = np.zeros((total_nodes, total_nodes)) 

    for i in range(total_nodes):
        for j in range(total_nodes):
            if i == j:  # Diagonal elements
                if i % 2 == 0:  # i even
                    matrix[i, j] = mu**2 + (N-1) * beta**2
                else:  # i odd
                    matrix[i, j] = mu**2
            else:  # Off-diagonal elements
                if i % 2 == 0 and j % 2 == 0:  # Both i and j are even
                    matrix[i, j] = mu**2 - beta**2
                elif i % 2 == 1 and j % 2 == 1:  # Both i and j are odd
                    matrix[i, j] = mu**2
                else:  # One index is even, the other is odd
                    matrix[i, j] = -mu**2

    return matrix

def build_ring_incidence(N):
    """Build the oriented incidence matrix of the 2N-cycle."""
    n = 2 * N
    E = np.zeros((n, n))
    for e in range(n):
        i = e
        j = (e + 1) % n
        # edge e starts at i, ends at j
        E[i, e] = 1.0
        E[j, e] = -1.0
    return E


def build_ring_graph(N):
    """Build the unweighted 2N-cycle with generators on odd Python indices and consumers on even ones."""
    G = nx.Graph()
    for n in range(2 * N):
        if n % 2 == 1:
            G.add_node(n, s=+1)
        else:
            G.add_node(n, s=-1)

    for e in range(2 * N):
        G.add_edge(e, (e + 1) % (2 * N))

    return G

def build_weighted_ring_graph_from_kvec(kvec):
    """Build a weighted cycle graph from an edge-capacity vector."""
    kvec = np.asarray(kvec, dtype=float)
    m = len(kvec)

    G = nx.Graph()
    for n in range(m):
        # even -> square, odd -> circle in draw_ring_inset
        G.add_node(n)

    for e in range(m):
        u = e
        v = (e + 1) % m
        G.add_edge(u, v, weight=float(kvec[e]))
    return G

def expected_dissipation_ring_from_incidence(E, k_vec, Sigma):
    """Compute the expected dissipation from a precomputed ring incidence matrix.
    <D> = <S^T L^+ S> = Tr(L^+ <S S^T>) with L = E K E^T."""
    k = np.asarray(k_vec, dtype=float).reshape(-1)

    if np.any(k < 0):
        raise ValueError("capacities must be >= 0")

    Sigma = np.asarray(Sigma, dtype=float)

    K = np.diag(k)
    L = E @ K @ E.T
    Lp = np.linalg.pinv(L)
    D = np.trace(Lp @ Sigma)

    return float(D)

def expected_dissipation_ring(N, k_vec, Sigma):
    """Compute the expected dissipation of a ring network from its capacities and injection covariance."""
    E = build_ring_incidence(N)
    return expected_dissipation_ring_from_incidence(E, k_vec, Sigma)

def k_symmetric(N, gamma, kappa=1.0):
    """Return the fully symmetric capacity vector for a 2N-edge ring."""
    m = 2 * N
    k0 = kappa * (m ** (-1.0 / gamma))
    return np.full(m, k0, dtype=float)

def kvec_N2_mu0_broken(gamma, kappa=1.0, which="left"):
    """Return a representative strongly symmetry-broken N=2, mu=0 capacity vector."""
    x = kappa * (2.0 ** (-1.0 / gamma))
    if which == "left":
        # [x,x,0,0]
        return np.array([x, x, 0.0, 0.0], dtype=float)
    else:
        # [0,0,x,x]
        return np.array([0.0, 0.0, x, x], dtype=float)

def compute_asymmetric_case_from_sigma(N, gamma, Sigma, kappa=1.0, broken_edge=-1):
    """Compute the strongly symmetry-broken path solution from a precomputed injection covariance matrix."""

    m = 2 * N

    # This path construction assumes that the removed edge is the closing edge
    if broken_edge not in (-1, m - 1):
        raise ValueError("This path formula assumes the broken edge is the closing edge (m-1).")

    # On the path, the flow through edge e equals the cumulative injection on nodes 0,...,e
    # Therefore <F_e^2> is the sum over the corresponding Sigma sub-block
    F2_path = np.array([np.sum(Sigma[:e+1, :e+1]) for e in range(m - 1)], dtype=float)

    # Clip tiny negative values caused by roundoff
    if F2_path.min() < -1e-10:
        raise ValueError(f"<F^2> became significantly negative: {F2_path.min()}")
    F2_path = np.maximum(F2_path, 0.0)

    # Corson update on the path edges, normalized to satisfy the resource constraint
    numer = F2_path ** (1.0 / (1.0 + gamma))
    denom = (np.sum(F2_path ** (gamma / (1.0 + gamma)))) ** (1.0 / gamma)
    k_path = (numer / denom) * kappa

    # Dissipation of the path solution
    D_path = np.sum(np.divide(F2_path, k_path, out=np.zeros_like(F2_path), where=(k_path > 0)))

    # Embed the path capacities back into the original ring edge list
    K = np.zeros(m, dtype=float)
    K[:m-1] = k_path
    K[m-1] = 0.0

    return float(D_path), K

def kb_from_constraint_2branch(ka, gamma, N, kappa=1.0):
    """Compute the second branch capacity from the two-value resource constraint."""
    rhs = (kappa**gamma) / N - ka**gamma
    if rhs < 0:
        return np.nan
    return rhs**(1.0 / gamma)


def D_alternating_scalar(ka, N, gamma, mu, beta, kappa=1.0):
    """Return the alternating-branch dissipation as a scalar function of ka."""
    kb = kb_from_constraint_2branch(ka, gamma, N, kappa)

    if not np.isfinite(kb) or ka <= 0 or kb <= 0:
        return np.inf

    return (mu**2 * N / (ka + kb) + beta**2 * N * (N**2 - 1) / 12.0 * (ka + kb) / (ka * kb))

def find_best_alternating(N, gamma, mu, beta, kappa=1.0):
    """Find the optimal alternating weakly broken branch."""
    ka_max = (kappa**gamma / N)**(1.0 / gamma)

    eps = 1e-12
    res = minimize_scalar(D_alternating_scalar, bounds=(eps, ka_max - eps), args=(N, gamma, mu, beta, kappa),
        method="bounded", options={"xatol": 1e-10})

    ka = res.x
    kb = kb_from_constraint_2branch(ka, gamma, N, kappa)
    k = np.array([ka if i % 2 == 0 else kb for i in range(2 * N)], dtype=float)

    return float(res.fun), k

def D_block_scalar(ka, gamma, mu, beta, kappa=1.0):
    """Return the N=2 block-branch dissipation as a scalar function of ka."""
    kb = kb_from_constraint_2branch(ka, gamma, 2, kappa)
    if not np.isfinite(kb) or ka <= 0 or kb <= 0:
        return np.inf

    return mu**2 * (ka + kb) / (2.0 * ka * kb) + 2.0 * beta**2 / (ka + kb)

def find_best_block_N2(gamma, mu, beta, kappa=1.0):
    """Find the optimal N=2 block-structured weakly broken branch."""
    ka_max = (kappa**gamma / 2.0)**(1.0 / gamma)

    eps = 1e-12
    res = minimize_scalar(D_block_scalar, bounds=(eps, ka_max - eps), args=(gamma, mu, beta, kappa),
        method="bounded", options={"xatol": 1e-10})

    ka = res.x
    kb = kb_from_constraint_2branch(ka, gamma, 2, kappa)
    k = np.array([ka, ka, kb, kb], dtype=float)

    return float(res.fun), k

# === Strong symmetry breaking transition in rings ===
def _F2_open_ring(N, mu, beta):
    """<F_e^2> on the 2N-1 edges of the open ring, from KCL on the path.

    Edge e (0-indexed) carries the cumulative injection of nodes 0..e, which
    contain n_g sources and n_c sinks (sources at even node indices).
    """
    e = np.arange(2 * N - 1)
    n_g = (e + 2) // 2
    n_c = (e + 1) // 2
    return mu**2 * (n_g - n_c)**2 + n_g * (N - n_g) * beta**2


def _lnD_sym_minus_lnD_open(gamma, N, mu, beta):
    """ln D_sym - ln D_open from the closed forms (SI Eqs. 65 and 67), kappa = 1."""
    gamma = np.asarray(gamma, dtype=float)
    p = np.atleast_1d(gamma / (1.0 + gamma))
    lnD_sym = np.log(2 * N) / gamma + np.log(N * mu**2 / 2 + N * (N**2 - 1) * beta**2 / 6)
    F2 = _F2_open_ring(N, mu, beta)
    lnD_open = np.log(np.sum(F2[:, None] ** p[None, :], axis=0)) / p
    return lnD_sym - (lnD_open if gamma.ndim else lnD_open[0])


def gamma_crossing_closed_form(N, mu, beta, gamma_grid=np.linspace(0.02, 0.999, 400)):
    """Smallest gamma at which the symmetric and open ring have equal dissipation."""
    v = _lnD_sym_minus_lnD_open(gamma_grid, N, mu, beta)
    idx = np.where(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return brentq(_lnD_sym_minus_lnD_open, gamma_grid[i], gamma_grid[i + 1],
                  args=(N, mu, beta), xtol=1e-14)


def gamma_c_vs_N_closed_form(N_values, mu, beta_bounds=(0.05, 3.0), n_beta=300):
    """gamma_c(N) as the minimum over beta of the sym/open crossing (SI Methods 7).

    A coarse beta scan brackets the minimum, which is then refined with a
    bounded scalar minimisation. Returns gamma_c and the minimising beta.
    """
    gamma_c, beta_c = [], []
    for N in N_values:
        betas = np.linspace(*beta_bounds, n_beta)
        gc = np.array([gamma_crossing_closed_form(N, mu, b) for b in betas])
        j = np.nanargmin(gc)
        res = minimize_scalar(lambda b: gamma_crossing_closed_form(N, mu, b),
                              bounds=(betas[max(j - 1, 0)], betas[min(j + 1, n_beta - 1)]),
                              method="bounded", options={"xatol": 1e-12})
        gamma_c.append(res.fun)
        beta_c.append(res.x)
    return np.array(gamma_c), np.array(beta_c)
    
def get_capacities_all_branches(N, mu, beta_values, gamma_values, kappa=1.0, verbose = True):
    """Compare all candidate branches across parameter space and record the optimal one."""

    # Allocate for output
    capacity_values = np.zeros((len(beta_values), len(gamma_values)))
    branch_labels = [[None for _ in gamma_values] for _ in beta_values]

    # Calculate only once
    E = build_ring_incidence(N)

    # Progress tracking
    total_points = len(beta_values) * len(gamma_values)
    point_counter = 0

    # Loop over parameter values
    for i, beta in enumerate(beta_values):
        Sigma = compute_ring_second_moment_matrix(mu, beta, N)

        for j, gamma in enumerate(gamma_values):
            point_counter += 1
            if verbose:
                print(f"[{point_counter}/{total_points}] ({i}, {j})", end="\r", flush=True)
            candidates = []

            # Symmetric
            k_sym = k_symmetric(N, gamma, kappa=kappa)
            D_sym = expected_dissipation_ring_from_incidence(E, k_sym, Sigma)
            candidates.append(("symmetric", D_sym, k_sym))

            # Strongly symmetry-broken
            D_brk, k_brk = compute_asymmetric_case_from_sigma(N, gamma, Sigma, kappa=kappa, broken_edge=-1)
            candidates.append(("strongly_broken", D_brk, k_brk))

            # Weakly symmetry-broken: alternating
            D_alt, k_alt = find_best_alternating(N, gamma, mu, beta, kappa=kappa)
            candidates.append(("weakly_broken", D_alt, k_alt))

            # Weakly symmetry-broken: block (only for N=2)
            if N == 2:
                D_blk, k_blk = find_best_block_N2(gamma, mu, beta, kappa=kappa)
                candidates.append(("block", D_blk, k_blk))

            label, D_best, k_best = min(candidates, key=lambda x: x[1])

            capacity_values[i, j] = np.min(k_best)
            branch_labels[i][j] = label

    return capacity_values, branch_labels

def k0_from_gamma(gamma, kappa=1.0):
    """Return the unperturbed broken-branch capacity scale used in the epsilon expansion."""
    return kappa * 4 ** (-1.0 / gamma)

def k_of_eps(eps, gamma, kappa=1.0):
    """Return the perturbed branch capacities as a function of epsilon."""
    eps = np.asarray(eps, dtype=float)
    rhs = kappa**gamma / 4.0 - eps**gamma / 2.0
    out = np.full_like(eps, np.nan, dtype=float)
    mask = rhs > 0
    out[mask] = rhs[mask] ** (1.0 / gamma)
    return out

def D_eps(eps, gamma, beta=1.0, kappa=1.0):
    """Evaluate the dissipation of the epsilon-perturbed branch."""
    k = k_of_eps(eps, gamma, kappa=kappa)
    return 4.0 * beta**2 * (eps + 2.0 * k) / (2.0 * eps * k + k**2)

def D0(gamma, beta=1.0, kappa=1.0):
    """Evaluate the dissipation of the unperturbed broken branch."""
    k0 = k0_from_gamma(gamma, kappa=kappa)
    return 8.0 * beta**2 / k0

def first_sign_change_x(x, y):
    """Locate the first sign change of y(x) using linear interpolation."""
    y = np.asarray(y, dtype=float)
    s = np.sign(y)

    # replace exact zeros by neighboring sign if possible
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    for i in range(len(s) - 2, -1, -1):
        if s[i] == 0:
            s[i] = s[i + 1]

    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) == 0:
        return np.nan

    i = idx[0]
    x1, x2 = x[i], x[i + 1]
    y1, y2 = y[i], y[i + 1]

    # Linear interpolation
    return x1 - y1 * (x2 - x1) / (y2 - y1)

def eps_star_asym(gamma, kappa=1.0):
    """From C1 eps^gamma \sim C2 eps:
        epsilon* \sim kappa * 4^{-1/gamma} * (3 gamma)^{-1/(1-gamma)}
    """
    return kappa * 4 ** (-1.0 / gamma) * (3.0 * gamma) ** (-1.0 / (1.0 - gamma))


def get_Figure10_data(gamma_scan, eps_vals, beta, kappa, numerical_precision = 1e-14):
    """Generate numerical and asymptotic epsilon* data for the Figure 10 comparison."""
    eps_cross_num = []

    for gamma in gamma_scan:
        delta = D_eps(eps_vals, gamma, beta=beta, kappa=kappa) - D0(gamma, beta=beta, kappa=kappa)
        eps_cross_num.append(first_sign_change_x(eps_vals, delta))
    
    eps_cross_num = np.array(eps_cross_num)
    eps_cross_asym = eps_star_asym(gamma_scan, kappa=kappa)

    mask = (eps_cross_num > numerical_precision) & (eps_cross_num < 1e-1)

    gamma_numerical = gamma_scan[mask]
    eps_numerical = eps_cross_num[mask]
    gamma_asymptotic = gamma_scan
    eps_asymptotic = eps_cross_asym

    return gamma_numerical, eps_numerical, gamma_asymptotic, eps_asymptotic