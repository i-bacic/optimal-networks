"""Complete bipartite networks K_{Ng,Nc}: scans of the optimal network and phase classification.

Node ordering is sources 0..Ng-1, then sinks Ng..Ng+Nc-1. Edge ordering is e = i*Nc + j
for the edge (i, j) oriented source -> sink, so an edge vector reshaped to (Ng, Nc) has
row index = source and column index = sink. Scan results are stored with shape
(len(beta_values), len(gamma_values), ...).

A coarse scan collects candidate configurations, which are reduced to one representative per support class.
A fine scan tries every representative at every point and keeps the lowest dissipation.
"""

from itertools import permutations
import networkx as nx
import numpy as np

from .cube_calc import representative_point

NONZERO_TOL = 1e-8

# === Geometry and injections ===

def build_bipartite_graph(Ng, Nc):
    """K_{Ng,Nc} as a networkx graph, sources first, with s = +1 (source) or -1 (sink)."""
    G = nx.Graph()
    for i in range(Ng):
        G.add_node(i, s=+1)
    for j in range(Nc):
        G.add_node(Ng + j, s=-1)
    for i in range(Ng):
        for j in range(Nc):
            G.add_edge(i, Ng + j)
    return G


def build_bipartite_incidence(Ng, Nc):
    """Oriented incidence matrix of K_{Ng,Nc}, edges ordered e = i*Nc + j, source -> sink."""
    E = np.zeros((Ng + Nc, Ng * Nc))
    for i in range(Ng):
        for j in range(Nc):
            e = i * Nc + j
            E[i, e] = +1.0
            E[Ng + j, e] = -1.0
    return E


def edge_index(Ng, Nc):
    """Arrays (src, snk) of length |E| giving the endpoints of each edge."""
    return np.repeat(np.arange(Ng), Nc), np.tile(np.arange(Nc), Ng)


def bipartite_second_moment_matrix(Ng, Nc, mu, beta):
    """Sigma = mu^2 v v^T + Ng beta^2 Q, Eq. (37) of the Methods."""
    n = Ng + Nc
    v = np.concatenate([(Nc / Ng) * np.ones(Ng), -np.ones(Nc)])
    Q = np.zeros((n, n))
    Q[:Ng, :Ng] = np.eye(Ng) - np.ones((Ng, Ng)) / Ng
    return mu ** 2 * np.outer(v, v) + Ng * beta ** 2 * Q


# === Corson iteration and dissipation ===

def selfconsistent_minimum_rel(E, k, Sigma, gamma, cap_k=1.0, rtol=1e-12, max_repetitions=400):
    """Corson iteration with a scale-invariant convergence test.

    Identical to core.corson_algorithm.selfconsistent_minimum except for the stopping
    rule. That routine tests sum((k - k_old)^2) < 1e-10 in absolute terms, but the
    constraint sum k^gamma = kappa^gamma forces k ~ |E|^{-1/gamma}, which is 1e-12 for
    |E| = 16 at gamma = 0.1. The absolute test is then met after a single sweep, while at
    gamma near 1 it is far too strict. Testing the relative change fixes both ends.
    """
    k = np.asarray(k, dtype=float).copy()
    s = np.sum(k ** gamma)
    if s <= 0:
        return k, np.inf
    k *= cap_k / s ** (1.0 / gamma)

    for _ in range(max_repetitions):
        L = E @ np.diag(k) @ E.T
        PTDF = np.diag(k) @ E.T @ np.linalg.pinv(L)
        F2 = np.maximum(np.diag(PTDF @ Sigma @ PTDF.T), 0.0)

        k_old = k
        denom = np.sum(F2 ** (gamma / (1.0 + gamma))) ** (1.0 / gamma)
        if not np.isfinite(denom) or denom <= 0:
            break
        k = F2 ** (1.0 / (1.0 + gamma)) / denom * cap_k

        scale = np.sum(k ** 2)
        if scale <= 0 or np.sum((k - k_old) ** 2) <= rtol * scale:
            break

    active = k > NONZERO_TOL * np.max(k)
    D = float(np.sum(F2[active] / k[active])) if np.any(active) else np.inf
    return k, D


def dissipation(E, k, Sigma):
    """Expected dissipation sum_e <F_e^2>/k_e, skipping inactive edges."""
    L = E @ np.diag(k) @ E.T
    PTDF = np.diag(k) @ E.T @ np.linalg.pinv(L)
    F2 = np.diag(PTDF @ Sigma @ PTDF.T)
    active = k > NONZERO_TOL * np.max(k)
    if not np.any(active):
        return np.inf
    return float(np.sum(F2[active] / k[active]))


def symmetric_state(Ng, Nc, gamma, kappa=1.0):
    """Uniform capacities k_0 = (Ng Nc)^{-1/gamma} kappa."""
    m = Ng * Nc
    return np.full(m, kappa * m ** (-1.0 / gamma))


# === Coarse scan: seed bank and continuation in gamma ===

def seed_bank(Ng, Nc, n_random=8, n_trees=6, rng=None):
    """Structured plus random initial guesses, all normalised later by the solver."""
    rng = np.random.default_rng(0) if rng is None else rng
    src, snk = edge_index(Ng, Nc)
    m = Ng * Nc
    eps = 1e-3
    seeds = [np.ones(m)]  # uniform

    # double star: all edges at source 0 plus all edges at sink 0 (a spanning tree)
    k = np.full(m, eps)
    k[(src == 0) | (snk == 0)] = 1.0
    seeds.append(k.copy())

    # matching enhanced: boost a near-perfect matching
    match = (snk % Ng) == (src % Ng) if Nc >= Ng else (src % Nc) == (snk % Nc)
    for amp in (0.5, 4.0):
        k = np.ones(m)
        k[match] += amp
        seeds.append(k.copy())
    k = np.full(m, eps)
    k[match] = 1.0
    seeds.append(k.copy())

    # hub enhanced: promote the first s sinks
    for s in range(1, min(Nc, 4)):
        k = np.ones(m)
        k[snk < s] += 4.0
        seeds.append(k.copy())
        k = np.full(m, eps)
        k[snk < s] = 1.0
        seeds.append(k.copy())

    # source enhanced: promote the first t sources
    for t in range(1, min(Ng, 3)):
        k = np.ones(m)
        k[src < t] += 4.0
        seeds.append(k.copy())

    # random restarts
    for _ in range(n_random):
        seeds.append(rng.random(m))

    # random spanning trees
    G = build_bipartite_graph(Ng, Nc)
    for _ in range(n_trees):
        T = nx.random_spanning_tree(G, weight=None, seed=int(rng.integers(1 << 30)))
        k = np.zeros(m)
        for e in range(m):
            if T.has_edge(int(src[e]), Ng + int(snk[e])):
                k[e] = 1.0 + 0.1 * rng.random()
        seeds.append(k)

    return seeds


def best_minimum(E, Sigma, gamma, seeds, kappa=1.0, max_repetitions=400):
    """Run the Corson iteration from every seed and keep the lowest-dissipation result."""
    best_k, best_D = None, np.inf
    for s in seeds:
        s = np.asarray(s, dtype=float).copy()
        if np.all(s <= 0):
            continue
        try:
            k, _ = selfconsistent_minimum_rel(E, s, Sigma, gamma, cap_k=kappa,
                                              max_repetitions=max_repetitions)
        except (np.linalg.LinAlgError, FloatingPointError):
            continue
        if not np.all(np.isfinite(k)):
            continue
        D = dissipation(E, k, Sigma)
        if np.isfinite(D) and D < best_D:
            best_D, best_k = D, k.copy()
    return best_k, best_D


def gamma_scan(Ng, Nc, mu, beta, gamma_values, kappa=1.0, rng=None):
    """Downward continuation in gamma at fixed beta, with the previous solution as a seed."""
    E = build_bipartite_incidence(Ng, Nc)
    Sigma = bipartite_second_moment_matrix(Ng, Nc, mu, beta)
    base = seed_bank(Ng, Nc, rng=rng)

    order = np.argsort(-np.asarray(gamma_values, dtype=float))
    k_out = np.zeros((len(gamma_values), Ng * Nc))
    D_out = np.full(len(gamma_values), np.inf)

    carried = None
    for idx in order:
        gamma = float(gamma_values[idx])
        seeds = list(base) + [symmetric_state(Ng, Nc, gamma, kappa)]
        if carried is not None:
            seeds.append(carried.copy())
            seeds.append(carried + 1e-2 * np.max(carried) * np.ones_like(carried))
        k, D = best_minimum(E, Sigma, gamma, seeds, kappa=kappa)
        if k is None:
            continue
        k_out[idx], D_out[idx] = k, D
        carried = k
    return k_out, D_out


def _summarise(k, D, beta_values, gamma_values):
    kmax = k.max(axis=2)
    return {"beta_values": np.asarray(beta_values), "gamma_values": np.asarray(gamma_values),
            "k": k, "D": D, "kmin_rel": k.min(axis=2) / kmax,
            "n_active": (k > NONZERO_TOL * kmax[..., None]).sum(axis=2)}


def compute_coarse_phase_diagram(Ng, Nc, mu, beta_values, gamma_values, kappa=1.0, seed=0,
                                 verbose=True):
    """Coarse scan: one downward continuation in gamma per beta, with the full seed bank."""
    rng = np.random.default_rng(seed)
    nb, ng = len(beta_values), len(gamma_values)
    k = np.zeros((nb, ng, Ng * Nc))
    D = np.zeros((nb, ng))
    for ib, beta in enumerate(beta_values):
        k[ib], D[ib] = gamma_scan(Ng, Nc, mu, beta, gamma_values, kappa=kappa, rng=rng)
        if verbose:
            print(f"\r  beta {ib + 1}/{nb}", end="", flush=True)
    if verbose:
        print()
    return _summarise(k, D, beta_values, gamma_values)


# === Support classes and the configuration bank ===

def canonical_support(k, Ng, Nc):
    """Support of k, relabelled to a canonical form under S_Ng x S_Nc."""
    M = (np.asarray(k) > NONZERO_TOL * np.max(k)).reshape(Ng, Nc)
    return min(M[np.ix_(r, c)].tobytes() for r in permutations(range(Ng))
               for c in permutations(range(Nc)))


def extract_configuration_bank(results, Ng, Nc, uniform_tol=1e-6, block_seeds=True):
    """Seed bank for the fine scan, built from the configurations of a scan.

    Keeps one representative per support class, and within each class separately a
    uniform and a non-uniform one, so that a weakly broken state is not merged with the
    symmetric state of the same support. Configurations in the same class differ by
    a relabelling of sources and sinks, so one representative per kind suffices.

    With block_seeds, two configurations along the critical blocks of the symmetric state
    are added: one sink slightly strengthened (sink block) and one matching slightly
    strengthened (loop block). Near a continuous transition the broken state is reachable
    only from such a seed, since a uniform start stays uniform under the Corson update.
    """
    bank = {}
    for k in results["k"].reshape(-1, Ng * Nc):
        if np.max(k) <= 0:
            continue
        active = k > NONZERO_TOL * np.max(k)
        uniform = np.min(k[active]) / np.max(k[active]) > 1 - uniform_tol
        bank.setdefault((canonical_support(k, Ng, Nc), uniform), k.copy())
    bank = list(bank.values())

    if block_seeds:
        src, snk = edge_index(Ng, Nc)
        ones = np.ones(Ng * Nc)
        bank.append(ones * np.where(snk == 0, 1.05, 1.0))       # sink block
        bank.append(ones * np.where(src == snk, 1.05, 1.0))     # loop block, needs Ng <= Nc
    return bank


# === Fine scan: every configuration of the bank at every point ===

def compute_fine_phase_diagram(Ng, Nc, mu, beta_values, gamma_values, bank, kappa=1.0,
                               max_repetitions=20000, verbose=True):
    """Fine scan: at every point, start from every bank configuration and the symmetric
    state, and keep the lowest dissipation. No continuation, so neighbouring rows cannot
    end up on different local minima."""
    E = build_bipartite_incidence(Ng, Nc)
    nb, ng = len(beta_values), len(gamma_values)
    k = np.zeros((nb, ng, Ng * Nc))
    D = np.full((nb, ng), np.inf)
    for ib, beta in enumerate(beta_values):
        Sigma = bipartite_second_moment_matrix(Ng, Nc, mu, beta)
        for ig, gamma in enumerate(gamma_values):
            seeds = list(bank) + [symmetric_state(Ng, Nc, gamma, kappa)]
            kk, dd = best_minimum(E, Sigma, gamma, seeds, kappa=kappa,
                                  max_repetitions=max_repetitions)
            if kk is not None:
                k[ib, ig], D[ib, ig] = kk, dd
        if verbose:
            print(f"\r  beta {ib + 1}/{nb}", end="", flush=True)
    if verbose:
        print()
    return _summarise(k, D, beta_values, gamma_values)

# === Phases ===

def classify_phase_diagram(results, Ng, Nc, sym_tol=1e-6):
    """Support-class label at every scan point; -1 marks the symmetric state.

    Returns the label array, shape (len(beta), len(gamma)).
    """
    k = results["k"]
    nb, ng = k.shape[:2]
    active = k > NONZERO_TOL * k.max(axis=2, keepdims=True)
    masks, inv = np.unique(active.reshape(-1, Ng * Nc), axis=0, return_inverse=True)
    canon = [canonical_support(m.astype(float), Ng, Nc) for m in masks]
    ids = {c: i for i, c in enumerate(dict.fromkeys(canon))}
    label = np.array([ids[canon[j]] for j in np.ravel(inv)]).reshape(nb, ng)
    return np.where(results["kmin_rel"] > 1 - sym_tol, -1, label)


def select_phases(results, label, beta_star, n_show=8, min_fraction=0.005, skip=()):
    """Symmetric state plus the n_show largest broken classes, each at its most interior point.

    Classes whose label is in skip are left out. The representative point is the one furthest
    from the boundary of its class, as for the cube. Returned in gallery order: symmetric,
    then the classes shown at beta > beta*, then those at beta <= beta*, each group by
    decreasing number of active edges. The second return value is the number of classes
    shown at beta > beta*.
    """
    beta_values, gamma_values, k = results["beta_values"], results["gamma_values"], results["k"]
    phases = []
    for lab in np.unique(label):
        mask = label == lab
        if lab in skip or (lab != -1 and mask.mean() < min_fraction):
            continue
        ib, ig = representative_point(mask)
        phases.append(dict(label=int(lab), symmetric=bool(lab == -1), fraction=float(mask.mean()),
                           n_edges=int(results["n_active"][ib, ig]), k=k[ib, ig],
                           beta=float(beta_values[ib]), gamma=float(gamma_values[ig])))
    sym = [p for p in phases if p["symmetric"]]
    broken = sorted([p for p in phases if not p["symmetric"]], key=lambda p: -p["fraction"])[:n_show]
    strong = sorted([p for p in broken if p["beta"] > beta_star], key=lambda p: -p["n_edges"])
    weak = sorted([p for p in broken if p["beta"] <= beta_star], key=lambda p: -p["n_edges"])
    return sym + strong + weak, len(strong)

# === Local stability boundaries, Supplementary Methods 10 ===

def gamma_s_loop(Ng, beta_rel):
    """Eq. (157) of the SI, as a function of beta / beta*."""
    b2 = np.asarray(beta_rel, dtype=float) ** 2
    return ((Ng - 3) * b2 + 1) / ((Ng - 1) * b2 + 1)

def gamma_s_sink(Ng, beta_rel):
    """Eq. (156) of the SI, as a function of beta / beta*."""
    b2 = np.asarray(beta_rel, dtype=float) ** 2
    return ((Ng - 1) * b2 - 1) / ((Ng - 1) * b2 + 1)