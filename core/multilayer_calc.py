"""Extended multilayer network (Supplementary Note 4): scans of the optimal network and phase measures.

Two L x L square lattices stacked in z, with interlayer edges between nodes at equal (x, y).
Node n = z*L*L + x*L + y. Edges are indexed in the order of G.edges(), which is the column
order of nx.incidence_matrix and therefore of all capacity vectors. Scan results are stored
with shape (len(beta_values), len(gamma_values), ...), as for the bipartite network.

The pipeline follows bipartite_calc. A coarse scan, one downward continuation in gamma per
beta, collects candidate configurations, which are reduced to one representative per support
class modulo Gamma (D_2d, order 8). A fine scan tries every representative at every point and
keeps the lowest dissipation, without continuation.

Initial conditions are continuation, all ones,
random dense, random spanning tree, drawn several times.

Reproducibility: every row of a scan draws from its own generator, default_rng([seed, row]),
so a scan is identical whether it runs serially, in parallel over rows (n_jobs), or is
resumed from a checkpoint. Both scans write their file after every finished row of beta and
resume from it. Set OMP_NUM_THREADS=1 before importing numpy: the matrices are small, so BLAS
threads only slow things down, and single-threaded LAPACK keeps the results bit-reproducible.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import numpy as np

from .bipartite_calc import selfconsistent_minimum_rel, dissipation, NONZERO_TOL
from .cube_calc import symmetry_group, representative_point

SOURCES = ((1, 1, 0), (3, 3, 0), (1, 3, 1), (3, 1, 1))


# === Geometry and injections ===

def build_multilayer_graph(L=5, num_layers=2, source_positions=SOURCES):
    """Stacked L x L lattices with s = +1 (source) or -1 (sink), and 3D positions for plotting.

    Same construction, node numbering and edge order as multilayer_lattice_network in the
    earlier multilayer notebooks.
    """
    G, pos = nx.Graph(), {}
    for z in range(num_layers):
        for x in range(L):
            for y in range(L):
                n = z * L * L + x * L + y
                pos[n] = [x, y, z]
                G.add_node(n, s=+1 if (x, y, z) in source_positions else -1)
                if x > 0:
                    G.add_edge(n, z * L * L + (x - 1) * L + y)
                if y > 0:
                    G.add_edge(n, z * L * L + x * L + (y - 1))
                if z > 0:
                    G.add_edge(n, (z - 1) * L * L + x * L + y)
    return G, pos


def incidence(G):
    return nx.incidence_matrix(G, oriented=True).toarray().astype(float)


def second_moment_matrix(G, mu, beta):
    """Sigma = mu^2 v v^T + Ng beta^2 Q, Eq. (37) of the Methods, in the node order of G."""
    s = np.array([G.nodes[u]["s"] for u in G.nodes()])
    src = s > 0
    Ng, Nc = int(src.sum()), int((~src).sum())
    v = np.where(src, Nc / Ng, -1.0)
    Q = np.zeros((len(s), len(s)))
    Q[np.ix_(src, src)] = np.eye(Ng) - 1.0 / Ng
    return mu ** 2 * np.outer(v, v) + Ng * beta ** 2 * Q


def default_mu(G):
    """mu such that the mean source injection Nc/Ng mu is 1, i.e. 4/46 for L = 5."""
    s = np.array([G.nodes[u]["s"] for u in G.nodes()])
    return (s > 0).sum() / (s <= 0).sum()


def edge_layers(G, pos):
    """Index arrays of the intralayer edges in z = 0 (A), z = 1 (B), and of the interlayer edges."""
    z = np.array([[pos[u][2], pos[v][2]] for u, v in G.edges()])
    return (np.where((z[:, 0] == 0) & (z[:, 1] == 0))[0],
            np.where((z[:, 0] == 1) & (z[:, 1] == 1))[0],
            np.where(z[:, 0] != z[:, 1])[0])


# === Seeds and the minimiser ===

def _random_tree(G, rng):
    """Capacities 1 + 0.1 U on a random spanning tree, zero elsewhere, as in the old notebooks."""
    T = nx.random_spanning_tree(G, weight=None, seed=int(rng.integers(1 << 30)))
    return np.array([1.0 + 0.1 * rng.random() if T.has_edge(u, v) else 0.0 for u, v in G.edges()])


def seed_bank(G, pos, rng, n_random=4, n_trees=4):
    """All ones, one layer strengthened, random dense, random spanning trees.

    All ones is Gamma-invariant and stays so under the Corson update, so it follows the
    symmetric branch. Strengthening layer A breaks the layer exchange, the symmetry lost
    at strong fluctuations (Supplementary Note 4); layer B is equivalent by symmetry.
    """
    m = G.number_of_edges()
    A, _, _ = edge_layers(G, pos)
    seeds = [np.ones(m)]
    k = np.ones(m)
    k[A] *= 2.0
    seeds.append(k)
    seeds += [0.5 + rng.random(m) for _ in range(n_random)]
    seeds += [_random_tree(G, rng) for _ in range(n_trees)]
    return seeds


def best_minimum(E, Sigma, gamma, seeds, max_repetitions=400):
    """Corson iteration from every seed, keeping the lowest dissipation."""
    best_k, best_D = None, np.inf
    for s in seeds:
        s = np.asarray(s, dtype=float)
        if not np.any(s > 0):
            continue
        try:
            k, _ = selfconsistent_minimum_rel(E, s.copy(), Sigma, gamma,
                                              max_repetitions=max_repetitions)
        except (np.linalg.LinAlgError, FloatingPointError):
            continue
        if not np.all(np.isfinite(k)):
            continue
        D = dissipation(E, k, Sigma)
        if np.isfinite(D) and D < best_D:
            best_k, best_D = k.copy(), D
    return best_k, best_D


# === Checkpointed row scan ===

def _save(path, res):
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **res)
    os.replace(tmp, path)          # atomic: an interrupted save never corrupts the file


def _row_scan(path, beta_values, gamma_values, m, mu, worker, args, n_jobs, verbose):
    """Run worker(ib, *args) -> (k_row, D_row) for every beta, saving after each row and resuming.

    Rows are independent and deterministic, so they can run in parallel (n_jobs processes) and
    in any order. The file holds the fields of the bipartite scans (grids, k, D, n_active) plus
    mu and rows_done, a boolean mask of finished rows.
    """
    beta_values, gamma_values = np.asarray(beta_values, float), np.asarray(gamma_values, float)
    nb, ng = len(beta_values), len(gamma_values)
    res = None
    if path is not None and os.path.exists(path):
        res = dict(np.load(path))
        same = (np.array_equal(res["beta_values"], beta_values)
                and np.array_equal(res["gamma_values"], gamma_values)
                and np.isclose(float(res["mu"]), mu))
        if not same:
            raise ValueError(f"{path} holds a different grid or mu; delete it or change the path")
        if verbose and not res["rows_done"].all():
            print(f"resuming {path}: {int(res['rows_done'].sum())}/{nb} rows done")
    if res is None:
        res = dict(beta_values=beta_values, gamma_values=gamma_values, mu=np.float64(mu),
                   k=np.zeros((nb, ng, m)), D=np.full((nb, ng), np.inf),
                   n_active=np.zeros((nb, ng), int), rows_done=np.zeros(nb, bool))

    todo = [ib for ib in range(nb) if not res["rows_done"][ib]]
    t0, n_done = time.time(), 0

    def store(ib, k_row, D_row):
        nonlocal n_done
        res["k"][ib], res["D"][ib] = k_row, D_row
        kmax = np.maximum(k_row.max(axis=1, keepdims=True), 1e-300)
        res["n_active"][ib] = (k_row > NONZERO_TOL * kmax).sum(axis=1)
        res["rows_done"][ib] = True
        if path is not None:
            _save(path, res)
        n_done += 1
        if verbose:
            eta = (time.time() - t0) / n_done * (len(todo) - n_done)
            print(f"\r  {int(res['rows_done'].sum())}/{nb} rows, remaining ~{eta / 3600:.1f} h   ",
                  end="", flush=True)

    if n_jobs == 1:
        for ib in todo:
            store(ib, *worker(ib, *args))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(worker, ib, *args): ib for ib in todo}
            for f in as_completed(futures):
                store(futures[f], *f.result())
    if verbose:
        print()
    return res


# === Coarse scan ===

def _coarse_row(ib, G, pos, mu, beta_values, gamma_values, seed, n_random, n_trees):
    E = incidence(G)
    rng = np.random.default_rng([seed, ib])
    Sigma = second_moment_matrix(G, mu, beta_values[ib])
    bank = seed_bank(G, pos, rng, n_random=n_random, n_trees=n_trees)
    ng, m = len(gamma_values), E.shape[1]
    k_row, D_row, carried = np.zeros((ng, m)), np.full(ng, np.inf), None
    for ig in np.argsort(-np.asarray(gamma_values, float)):
        seeds = list(bank)
        if carried is not None:
            seeds += [carried, carried + 1e-2 * carried.max()]
        k, D = best_minimum(E, Sigma, float(gamma_values[ig]), seeds)
        if k is not None:
            k_row[ig], D_row[ig], carried = k, D, k
    return k_row, D_row


def compute_coarse_phase_diagram(G, pos, mu, beta_values, gamma_values, path=None, seed=0,
                                 n_random=4, n_trees=4, n_jobs=1, verbose=True):
    """Coarse scan: downward continuation in gamma at every beta, with the full seed bank.

    At every point the seeds are the bank, plus the previous solution and a copy of it with a
    small uniform offset, which lets a continued state reactivate edges.
    """
    args = (G, pos, mu, np.asarray(beta_values, float), np.asarray(gamma_values, float),
            seed, n_random, n_trees)
    return _row_scan(path, beta_values, gamma_values, G.number_of_edges(), mu,
                     _coarse_row, args, n_jobs, verbose)


# === Support classes and the configuration bank ===

def is_symmetric(k, edge_perm, rtol=1e-6, support_tol=None):
    """Whether k is invariant under all of Gamma. With support_tol, edges with
    k_e <= support_tol * max k are set to zero first."""
    if support_tol is not None:
        k = np.where(k > support_tol * np.max(k), k, 0.0)
    atol = rtol * np.max(k)
    return all(np.allclose(k[P], k, rtol=rtol, atol=atol) for P in edge_perm)


def canonical_support(k, edge_perm, support_tol=NONZERO_TOL):
    """Support of k (k_e > support_tol * max k), relabelled to a canonical form under Gamma."""
    a = np.asarray(k) > support_tol * np.max(k)
    return min(a[P].tobytes() for P in edge_perm)


def extract_configuration_bank(results, G, pos, edge_perm, block_seeds=True):
    """Seed bank for the fine scan: one configuration per (support class, symmetric or not).

    The symmetric flag plays the role of the uniform flag of the bipartite bank, since the
    network is not edge-transitive and the symmetric state is not uniform. With block_seeds,
    all ones with layer A strengthened by 5 % is added, along the layer-exchanging blocks,
    since a Gamma-invariant start stays invariant under the Corson update.
    """
    bank = {}
    for k in results["k"].reshape(-1, results["k"].shape[-1]):
        if np.max(k) <= 0:
            continue
        bank.setdefault((canonical_support(k, edge_perm), is_symmetric(k, edge_perm)), k.copy())
    bank = list(bank.values())
    if block_seeds:
        A, _, _ = edge_layers(G, pos)
        k = np.ones(G.number_of_edges())
        k[A] *= 1.05
        bank.append(k)
    return bank


# === Fine scan ===

def _fine_row(ib, G, mu, beta_values, gamma_values, seeds, max_repetitions):
    E = incidence(G)
    Sigma = second_moment_matrix(G, mu, beta_values[ib])
    ng, m = len(gamma_values), E.shape[1]
    k_row, D_row = np.zeros((ng, m)), np.full(ng, np.inf)
    for ig, gamma in enumerate(gamma_values):
        k, D = best_minimum(E, Sigma, float(gamma), seeds, max_repetitions=max_repetitions)
        if k is not None:
            k_row[ig], D_row[ig] = k, D
    return k_row, D_row


def compute_fine_phase_diagram(G, mu, beta_values, gamma_values, bank, path=None,
                               max_repetitions=20000, n_jobs=1, verbose=True):
    """Fine scan: at every point, start from every bank configuration and from all ones, and
    keep the lowest dissipation. No continuation, so neighbouring rows cannot end up on
    different local minima. Deterministic given the bank."""
    seeds = list(bank) + [np.ones(G.number_of_edges())]
    args = (G, mu, np.asarray(beta_values, float), np.asarray(gamma_values, float), seeds,
            max_repetitions)
    return _row_scan(path, beta_values, gamma_values, G.number_of_edges(), mu,
                     _fine_row, args, n_jobs, verbose)


def time_per_point(G, mu, bank, beta=1.0, gammas=(0.45, 0.7, 0.95), max_repetitions=20000):
    """Wall time of one fine-scan point, averaged over a few gammas, for planning a run."""
    E = incidence(G)
    Sigma = second_moment_matrix(G, mu, beta)
    seeds = list(bank) + [np.ones(G.number_of_edges())]
    t = time.time()
    for g in gammas:
        best_minimum(E, Sigma, g, seeds, max_repetitions=max_repetitions)
    return (time.time() - t) / len(gammas)


# === Measures of symmetry breaking, Supplementary Note 4 ===

def layer_measures(results, G, pos, active_tol=NONZERO_TOL):
    """Layer imbalance I, Eq. (32), max(N_A, N_B), N_I, and the coefficient of variation C of the
    active interlayer capacities, Eq. (33), each of shape (len(beta), len(gamma)).
    An edge is active if k_e > active_tol * max k."""
    A, B, I = edge_layers(G, pos)
    k = results["k"]
    act = k > active_tol * np.maximum(k.max(axis=2, keepdims=True), 1e-300)
    sA, sB = k[..., A].sum(-1), k[..., B].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = np.abs(sA - sB) / (sA + sB)
        kI, aI = k[..., I], act[..., I]
        n_I = aI.sum(-1)
        mean = np.where(aI, kI, 0).sum(-1) / n_I
        var = np.where(aI, (kI - mean[..., None]) ** 2, 0).sum(-1) / n_I
        cv = np.sqrt(var) / mean
    return dict(I=imb, n_max=np.maximum(act[..., A].sum(-1), act[..., B].sum(-1)),
                n_inter=n_I, C=cv)


# === Residual symmetry and phases ===

def element_types(G, pos, node_perm):
    """Geometric type of every element of Gamma ~ D_2d."""
    nodes = list(G.nodes())
    P = np.array([pos[u] for u in nodes])
    ident = np.arange(len(nodes))
    types = []
    for g in node_perm:
        h, order = g.copy(), 1
        while not np.array_equal(h, ident):
            h, order = g[h], order + 1
        swaps = P[g[0], 2] != P[0, 2]
        fixed = int(np.sum(g == ident))
        if order == 1:
            types.append("id")
        elif order == 4:
            types.append("S4")
        elif swaps:
            types.append("C2'")        # twofold rotation about an in-plane axis
        elif fixed > 2:
            types.append("sigma_d")    # reflection in a diagonal plane
        else:
            types.append("C2")         # twofold rotation about z
    return types


def residual_group(k, types, edge_perm, rtol=1e-4, support_tol=NONZERO_TOL):
    """Schoenflies symbol of Gamma*, the subgroup of Gamma ~ D_2d leaving all capacities invariant.
    Only edges with k_e > support_tol * max k enter; support_tol = 0.1, say, gives the residual
    symmetry of the backbone, ignoring the weak branches feeding the sinks."""
    k = np.asarray(k, float)
    k = np.where(k > support_tol * k.max(), k, 0.0)
    atol = rtol * k.max()
    stab = [types[i] for i, P in enumerate(edge_perm) if np.allclose(k[P], k, rtol=rtol, atol=atol)]
    n = len(stab)
    if n == 8:
        return "D_{2d}"
    if n == 1:
        return "C_1"
    if n == 2:
        t = [s for s in stab if s != "id"][0]
        return {"sigma_d": "C_s", "C2": "C_2", "C2'": "C_2'"}[t]
    if n == 4:
        if "S4" in stab:
            return "C_4^{(i)}"
        return "C_{2v}" if "sigma_d" in stab else "D_2"
    return f"order {n}"


def classify_phase_diagram(results, edge_perm, sym_rtol=1e-4, support_tol=NONZERO_TOL):
    """Support-class label at every point, shape (len(beta), len(gamma)); -1 marks
    Gamma-symmetric configurations, at the tolerance used for Gamma* in residual_group.
    Only edges with k_e > support_tol * max k enter."""
    k = results["k"]
    nb, ng, m = k.shape
    flat = k.reshape(-1, m)
    keys = [(canonical_support(kk, edge_perm, support_tol),
             is_symmetric(kk, edge_perm, sym_rtol, support_tol)) if kk.max() > 0
            else None for kk in flat]
    ids = {}
    lab = np.array([-2 if key is None else (-1 if key[1] else ids.setdefault(key[0], len(ids)))
                    for key in keys])
    return lab.reshape(nb, ng)


def list_classes(results, labels, types, edge_perm, rtol=1e-4, support_tol=NONZERO_TOL,
                 active_tol=NONZERO_TOL):
    """One row per class of classify_phase_diagram: label, N* (k_e > active_tol * max k),
    Gamma* (rtol and support_tol as in residual_group), fraction of the plane, and the scan
    point furthest from the class boundary (gamma, beta), sorted by fraction."""
    rows = []
    for lab in np.unique(labels):
        if lab == -2:
            continue
        mask = labels == lab
        ib, ig = representative_point(mask)
        k = results["k"][ib, ig]
        rows.append(dict(label=int(lab), n_edges=int((k > active_tol * k.max()).sum()),
                         group=residual_group(k, types, edge_perm, rtol, support_tol),
                         fraction=float(mask.mean()),
                         gamma=float(results["gamma_values"][ig]),
                         beta=float(results["beta_values"][ib])))
    return sorted(rows, key=lambda r: -r["fraction"])


def symmetry_mismatch(k, types, edge_perm, strong=0.1, active_tol=1e-5):
    """For every element g of Gamma: max |k_g(e) - k_e| / max k over the strong edges
    (k > strong * max k in k or its image), and the number of edges whose activity
    (k > active_tol * max k) differs from that of their image. Shows whether symmetry is
    broken by the backbone or only by the weak branches feeding the sinks."""
    r = np.asarray(k, float) / np.max(k)
    rows = []
    for t, P in zip(types, edge_perm):
        if t == "id":
            continue
        s_ = (r > strong) | (r[P] > strong)
        rows.append((t, float(np.abs(r[P] - r)[s_].max()),
                     int(((r > active_tol) != (r[P] > active_tol)).sum())))
    return rows


def gallery_at(results, points, active_tol=NONZERO_TOL):
    """Gallery entries at given (gamma, beta) points, snapped to the nearest scan point.
    N* counts edges with k_e > active_tol * max k."""
    gal = []
    for g, b in points:
        ib = int(np.argmin(np.abs(results["beta_values"] - b)))
        ig = int(np.argmin(np.abs(results["gamma_values"] - g)))
        k = results["k"][ib, ig]
        gal.append(dict(k=k, n_edges=int((k > active_tol * k.max()).sum()),
                        beta=float(results["beta_values"][ib]),
                        gamma=float(results["gamma_values"][ig])))
    return gal


def gallery_auto(results, labels, n_show=8, min_fraction=0.005):
    """The n_show largest classes, each at the scan point furthest from its boundary."""
    gal = []
    for lab in np.unique(labels):
        mask = labels == lab
        if lab == -2 or mask.mean() < min_fraction:
            continue
        ib, ig = representative_point(mask)
        gal.append(dict(k=results["k"][ib, ig], n_edges=int(results["n_active"][ib, ig]),
                        beta=float(results["beta_values"][ib]),
                        gamma=float(results["gamma_values"][ig]), fraction=float(mask.mean())))
    gal = sorted(gal, key=lambda p: -p["fraction"])[:n_show]
    return sorted(gal, key=lambda p: (-p["beta"], p["gamma"]))