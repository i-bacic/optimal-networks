"""Cube network: symmetry analysis of optimal networks and analytic stability lines.
"""

import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher
from scipy.ndimage import distance_transform_edt

def symmetry_group(G):
    """Symmetry group Gamma of the optimisation problem as permutations of nodes and edges.

    Gamma consists of the graph automorphisms that map sources onto sources (T_d for the
    cube, order 24). Edges are indexed in the order of G.edges(), which is the column order
    of nx.incidence_matrix and therefore of the capacity vectors returned by the scans.
    """
    edges = list(G.edges())
    eidx = {frozenset(e): i for i, e in enumerate(edges)}
    autos = [m for m in GraphMatcher(G, G, node_match=lambda a, b: a["s"] == b["s"]).isomorphisms_iter()]
    node_perm = np.array([[m[n] for n in G.nodes()] for m in autos])
    edge_perm = np.array([[eidx[frozenset((m[u], m[v]))] for (u, v) in edges] for m in autos])
    return node_perm, edge_perm


def residual_group(k, node_perm, edge_perm, rtol=1e-4):
    """Schoenflies symbol of the residual symmetry Gamma* of a capacity vector, Eq. (10).

    Gamma* consists of the elements of Gamma that leave every capacity invariant. Passing an
    indicator vector of the active edges instead gives the symmetry of the support.
    The group is identified from the orders of its elements and their fixed nodes: within
    T_d, a reflection fixes four nodes, a twofold rotation none, and the improper fourfold
    rotation is the only element of order four.
    """
    k = np.asarray(k, dtype=float)
    atol = rtol * np.max(np.abs(k))
    stab = [i for i, P in enumerate(edge_perm) if np.allclose(k[P], k, rtol=rtol, atol=atol)]
    identity = np.arange(node_perm.shape[1])
    orders, fixed = [], []
    for i in stab:
        g, h, m = node_perm[i], node_perm[i].copy(), 1
        while not np.array_equal(h, identity):
            h, m = g[h], m + 1
        orders.append(m)
        fixed.append(int(np.sum(g == identity)))
    n_refl = sum(1 for o, f in zip(orders, fixed) if o == 2 and f == 4)
    n = len(stab)
    if n == len(edge_perm):
        return "T_d"
    if n == 1:
        return "C_1"
    if n == 2:
        return "C_s" if n_refl == 1 else "C_2"
    if n == 3:
        return "C_3"
    if n == 4:
        return "C_4^{(i)}" if 4 in orders else ("C_{2v}" if n_refl == 2 else "D_2")
    if n == 6:
        return "C_{3v}"
    return f"order {n}"

# Phases of the cube, keyed by (number of active edges, order of the residual group).
# The two eight-edge phases have residual groups of different order.
CUBE_PHASES = {(12, "T_d"): "symmetric", (11, "C_s"): "eleven", (10, "C_{2v}"): "ten",
               (9, "C_3"): "chiral", (8, "C_4^{(i)}"): "ring", (8, "C_1"): "kite",
               (7, "C_1"): "tree"}


def classify_network(k, node_perm, edge_perm, nonzero_tol=1e-8, rtol=1e-3):
    """Phase name, number of active edges and residual symmetry Gamma* of one capacity vector.

    The phase is determined from the support (active edges and their symmetry). Gamma* is taken
    from the capacities, so it is smaller than the symmetry of the support when
    the capacities on that support are unequal, as for the alternating ring.
    """
    k = np.asarray(k, dtype=float)
    active = (k > nonzero_tol).astype(float)
    n = int(active.sum())
    name = CUBE_PHASES.get((n, residual_group(active, node_perm, edge_perm)))
    group = residual_group(np.where(k > nonzero_tol, k, 0.0), node_perm, edge_perm, rtol)
    return (name if name is not None else f"N*={n}"), n, group


def classify_phase_diagram(results, G, nonzero_tol=1e-8):
    """Phase label at every point of a fine scan, from its stored capacity vectors.

    The phase depends only on the support, of which a scan contains just a handful, so each
    distinct support is classified once and looked up afterwards.

    Returns an integer array of shape (len(gamma_values), len(beta_values)) indexing into the
    returned list of phase names.
    """
    node_perm, edge_perm = symmetry_group(G)
    k_values = results["k_values"]
    ng, nb = k_values.shape[:2]
    active = k_values > nonzero_tol

    names, labels, cache = [], np.full((ng, nb), -1, int), {}
    for i in range(ng):
        for j in range(nb):
            key = active[i, j].tobytes()
            if key not in cache:
                support = active[i, j].astype(float)
                n = int(support.sum())
                name = CUBE_PHASES.get((n, residual_group(support, node_perm, edge_perm)),
                                       f"N*={n}")
                if name not in names:
                    names.append(name)
                cache[key] = names.index(name)
            labels[i, j] = cache[key]
    return labels, names


# Local stability thresholds of the symmetric state, Supplementary Methods 8, and the
# alternating mode of the embedded N = 4 ring, Supplementary Methods 9.

def gamma_s_T1(beta, mu=1.0):
    """Eq. (129) of the SI."""
    return 8 * mu**2 / (8 * mu**2 + 27 * beta**2)

def gamma_s_T2_minus(beta, mu=1.0):
    """Eq. (131) of the SI, lower sign."""
    return (9 * np.sqrt(6) * beta**2 - 8 * mu**2) / (8 * mu**2 + 27 * beta**2)

def gamma_s_max(beta, mu=1.0):
    """Threshold of the symmetric state: the larger of the two blocks that can be critical."""
    return np.maximum(gamma_s_T1(beta, mu), gamma_s_T2_minus(beta, mu))

def gamma_s_alt_ring(beta, mu=1.0):
    """Alternating mode of the eight-edge ring, Eq. (138) of the SI."""
    return (mu**2 - 5 * beta**2) / (mu**2 + 5 * beta**2)

def phase_sequence(cut, node_perm, edge_perm, min_points=3):
    """(phase name, gamma in the middle of its interval, residual group) along a one-beta cut.

    Runs shorter than min_points are skipped, which drops isolated scan points that landed
    on a different local minimum.
    """
    g = cut["gamma_values"]
    k = cut["k_values"][:, 0, :]
    names = [classify_network(kk, node_perm, edge_perm)[0] for kk in k]
    runs, start = [], 0
    for i in range(1, len(g) + 1):
        if i == len(g) or names[i] != names[start]:
            if i - start >= min_points:
                mid = (start + i - 1) // 2
                runs.append((names[start], g[mid], classify_network(k[mid], node_perm, edge_perm)[2]))
            start = i
    return runs

def representative_point(mask):
    """Grid point furthest from the boundary of a region (outside the scan counts as boundary)."""
    d = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    return np.unravel_index(np.argmax(d), d.shape)