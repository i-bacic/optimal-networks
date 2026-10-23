import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.signal import argrelextrema

from .rings_calc import build_ring_graph, compute_ring_second_moment_matrix, D_of_a_N2, global_optimum_N2_piecewise, kvec_N3_from_ab, expected_dissipation_ring, meaningful_minima_N3, global_candidates_from_minima, build_weighted_ring_graph_from_kvec
from .corson_algorithm import random_minimum

class ScalarFormatterOneDecimal(ScalarFormatter):
    """Scalar formatter that displays one decimal place in scientific notation."""
    def _set_format(self):
        self.format = '%.1f'

# === Figure 2 ===

def minima_indices(y, include_endpoints=True):
    """Return indices of local minima, optionally including endpoint minima."""
    y = np.asarray(y)
    idx = list(argrelextrema(y, np.less)[0])  # interior strict minima
    if include_endpoints and len(y) >= 2:
        if np.isfinite(y[0]) and np.isfinite(y[1]) and y[0] <= y[1]:
            idx.append(0)
        if np.isfinite(y[-1]) and np.isfinite(y[-2]) and y[-1] <= y[-2]:
            idx.append(len(y) - 1)
    return sorted(set(idx))

def mark_minima(ax, x, y, rtol=1e-10, s_circle=220, s_cross=140, lw=1.0):
    """Mark local and global minima of a 1D curve on an existing axis.

    Local minima use the same white fill as the boundary minima drawn by
    plot_Figure3, so that the shared legend applies to every panel of the
    merged ring figure.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    idx = minima_indices(y, include_endpoints=True)
    idx = [i for i in idx if np.isfinite(y[i])]
    if not idx:
        return

    y_min = np.min(y[idx])
    tol = rtol * max(1.0, abs(y_min))

    global_idx = [i for i in idx if abs(y[i] - y_min) <= tol]
    local_only = [i for i in idx if i not in global_idx]

    # Minima sitting exactly on the domain boundary would otherwise be sliced in
    # half by the spine, so they are drawn unclipped.
    # local (non-global): filled circle only
    if local_only:
        ax.scatter(x[local_only], y[local_only], marker="o", s=s_circle, facecolors="white", edgecolors="black", linewidths=lw, zorder=20, clip_on=False)

    # global: x on top of a filled circle
    ax.scatter(x[global_idx], y[global_idx], marker="o", s=s_circle, facecolors="white", edgecolors="black", linewidths=lw, zorder=22, clip_on=False)
    ax.scatter(x[global_idx], y[global_idx], marker="x", s=s_cross, c="black", linewidths=lw, zorder=23, clip_on=False)


def plot_dissipation_N2(gammas, mu=0.0, beta=0.1, kappa=1.0, ax=None, legend_elements=None,
                        lw=2.0, s_circle=220, s_cross=140, marker_lw=1.0, legend_loc="center",
                        legend_ncol=1):
    """Plot the N=2 dissipation as a function of the reduced variable a for selected gamma values."""
    colors = ["#003366", "#E69F00"]

    for i, gamma in enumerate(gammas):
        a_max = kappa * (2.0 ** (-1.0 / gamma))
        a_vals = np.linspace(1e-16, a_max, 800)

        D_vals = np.array([D_of_a_N2(a, gamma, mu, beta, kappa=kappa) for a in a_vals])

        ax.plot(a_vals, D_vals, label=fr"$\gamma={gamma}$", lw=lw, color=colors[i], zorder=1)

        mark_minima(ax, a_vals, D_vals, s_circle=s_circle, s_cross=s_cross, lw=marker_lw)

    ax.set_xlabel(r"$k_1$")
    ax.set_ylabel(r"$\bar D$")

    if legend_elements is not None:
        handles, labels = ax.get_legend_handles_labels()
        # take the labels from the proxy artists rather than hardcoding them,
        # so the caller controls the wording
        ax.legend(handles + legend_elements,
                  labels + [h.get_label() for h in legend_elements], loc=legend_loc, ncol=legend_ncol)
    else:
        ax.legend(loc='upper right')

    return ax

def plot_fig2b_N2(gamma_vals, kappa=1.0, ax=None, lw_thick=4.0, lw_thin=2.0, legend_loc="lower right"):
    """Plot the analytic N=2 optimal capacities as a function of gamma.

    Where the two branches coincide (gamma > 1 they are equal) the orange curve
    is drawn thicker than the navy one so that both remain visible; this is a
    drawing device, not a difference in the underlying values.
    """
    k1 = []
    k3 = []
    for g in gamma_vals:
        a, b = global_optimum_N2_piecewise(g, kappa=kappa)
        k1.append(a)  # k1=k2=a
        k3.append(b)  # k3=k4=b

    k1 = np.array(k1)
    k3 = np.array(k3)

    g = np.asarray(gamma_vals)
    mask_left  = g < 1.0
    mask_right = g > 1.0

    # left branch (gamma < 1)
    ax.plot(g[mask_left],  k1[mask_left],  color="#E69F00", linewidth=lw_thick, label=r"$k_1^*=k_2^*$")
    ax.plot(g[mask_left],  k3[mask_left],  color="#003366", linewidth=lw_thin, label=r"$k_3^*=k_4^*$")

    # right branch (gamma > 1)
    ax.plot(g[mask_right], k3[mask_right], color="#E69F00", linewidth=lw_thick)
    ax.plot(g[mask_right], k1[mask_right], color="#003366", linewidth=lw_thin)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$k_e^*$")
    ax.legend(loc=legend_loc)

# === Figure 3 ===

def plot_Figure3(gamma, mu, beta, kappa=1.0, grid=260, ax=None, cax=None,
                 label_above=False, cbar_labelsize=None, s_circle=260, s_cross=140,
                 marker_lw=2.0, gamma_label_xy=(0.12, 0.95), minima_grid=None,
                 aspect="auto"):
    """Plot the reduced N=3 dissipation and mark its relevant minima.

    With label_above=True the colour bar is titled above the bar rather than
    alongside it. That is what keeps the ln(D) label clear of the neighbouring
    panel's y-axis label when the three panels sit side by side.

    The heat map is rasterised into the vector PDF, so `grid` sets its
    resolution: it must be large enough to reach 300 dpi at the final printed
    panel width (see figstyle for the physical sizes).
    """

    # Create (a, b) parameter grid
    a_max = ((kappa**gamma)/2.0)**(1.0/gamma)
    a_vals = np.linspace(0.0, a_max, grid)
    b_vals = a_vals

    # Evaluate the dissipation on the grid
    N = 3
    Sigma = compute_ring_second_moment_matrix(mu, beta, N)
    D = np.full((grid, grid), np.nan, float)

    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            k = kvec_N3_from_ab(a, b, gamma, kappa=kappa)
            if k is None:
                continue
            D[i, j] = expected_dissipation_ring(N, k, Sigma)
            
    # Plot heatmap of dissipation
    with np.errstate(invalid="ignore"):
        logD = np.log(D)

    im = ax.imshow(logD.T, origin="lower", aspect=aspect, interpolation="nearest",
                   extent=[a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]])

    # "ln" is a multi-letter function, so it must be roman rather than italic.
    cbar_label = r"$\ln \bar D$"
    fig = ax.get_figure()

    if cax is not None:
        cbar = fig.colorbar(im, cax=cax)
    else:
        cbar = fig.colorbar(im, ax=ax)

    if label_above:
        cbar.ax.set_title(cbar_label, fontsize=cbar_labelsize, pad=3)
    else:
        cbar.set_label(cbar_label, fontsize=cbar_labelsize)

    if cbar_labelsize is not None:
        cbar.ax.tick_params(labelsize=cbar_labelsize)

    # Compute minima (interior + boundaries). This search is a separate, much
    # more expensive scan than the heat map, so its resolution is decoupled from
    # `grid`: the heat map needs a fine grid to satisfy the 300 dpi rule, the
    # minima do not.
    if minima_grid is None:
        minima_grid = grid

    interior_min, boundary_mins = meaningful_minima_N3(gamma, mu, beta, kappa=kappa, n_grid=minima_grid, eps_frac=0.01)

    candidates = []
    if interior_min is not None:
        candidates.append(("sym", *interior_min))
    for key, (a_m, b_m, D_m) in boundary_mins.items():
        candidates.append((key, a_m, b_m, D_m))

    # Markers. These minima lie on the boundary of the feasible domain, so they
    # are drawn unclipped to stop the spines slicing them in half.
    # Boundary minima: yellow filled circles with black edge
    if gamma < 1: # we analytically showed this
        for (a_b, b_b, D_b) in boundary_mins.values():
            ax.scatter([a_b], [b_b], marker="o", s=s_circle, facecolors="white", edgecolors="black", linewidths=marker_lw, zorder=9, clip_on=False)
    # Interior minimum (if exists)
    if interior_min is not None:
        a_i, b_i, D_i = interior_min
        ax.scatter([a_i], [b_i], marker="o", s=s_circle, facecolors="white", edgecolors="black", linewidths=marker_lw, zorder=10, clip_on=False)
    globals_ = global_candidates_from_minima(candidates, use_log=True, atol_log=5e-3)

    for tag, a_g, b_g, D_g in globals_:
        ax.scatter([a_g], [b_g], marker="x", s=s_cross, c="black", linewidths=marker_lw, zorder=11, clip_on=False)

    ax.set_xlabel(r"$k_1=k_2$")
    ax.set_ylabel(r"$k_3=k_4$")
    # fixed to two decimals so the three panels do not mix 0.55, 0.6 and 1.05
    ax.text(gamma_label_xy[0], gamma_label_xy[1], rf"$\gamma={gamma:.2f}$",
            transform=ax.transAxes, ha="left", va="top")
    return ax

# Insets for Fig 3
def add_ring_inset(ax, kvec, bounds, color="white", edge_scale=8, node_size=300, lw_scale=1.0):
    """Add a weighted ring inset at a specified position in axis coordinates:
    bounds = [x0, y0, w, h] in ax.transAxes coordinates.
    """
    iax = ax.inset_axes(bounds, transform=ax.transAxes)
    G = build_weighted_ring_graph_from_kvec(kvec)
    draw_ring_inset(G, iax, color=color, node_size=node_size, edge_scale=edge_scale, lw_scale=lw_scale)
    return iax

def add_insets_to_Figure3(ax, gamma, kappa=1.0, node_size=180, edge_scale=5, lw_scale=1.0,
                          bounds=None):
    """
    Places representative insets in panels of Figure 3.
    Bounds are [x0, y0, w, h] in axes coordinates.
    """
    def k_sym_N3(gamma, kappa=1.0):
        x = kappa * 6.0 ** (-1.0 / gamma)
        return x, x, x

    def k_brk_N3(gamma, kappa=1.0):
        x = kappa * 4.0 ** (-1.0 / gamma)
        return 0.0, x, x

    def kvec_hex_from_abc(a, b, c):
        return np.array([a, a, b, b, c, c], dtype=float)
    
    gc = np.log(3/2) / np.log(2)
    gb = 1.0

    # Representative capacity patterns
    a_sym, b_sym, c_sym = k_sym_N3(gamma, kappa)
    a_brk, b_brk, c_brk = k_brk_N3(gamma, kappa)

    k_sym = kvec_hex_from_abc(a_sym, b_sym, c_sym)
    k_brk = kvec_hex_from_abc(a_brk, b_brk, c_brk)

    if gamma < gc:
        # gamma=0.55: global broken, local symmetric
        default_bounds = [0.62, 0.5, 0.23, 0.23]
        kvec = k_brk
    elif gamma < gb:
        # gamma=0.60: global symmetric, local broken
        default_bounds = [0.60, 0.5, 0.23, 0.23]
        kvec = k_sym
    else:
        # gamma=1.05: only symmetric local/global minimum
        default_bounds = [0.58, 0.5, 0.24, 0.24]
        kvec = k_sym

    add_ring_inset(ax, kvec, bounds=default_bounds if bounds is None else bounds,
                   color="black", edge_scale=edge_scale, node_size=node_size, lw_scale=lw_scale)

# === Figure 4 ===

def plot_Figure4(ax=None, kappa=1.0, gamma_min=0.32, gamma_max=1.3, n=500,
                 lw=2.5, guide_lw=1.0, legend_fontsize=12, legend_loc="upper left"):
    """Plot the analytic N=3 branch diagram showing global and local minima versus gamma."""

    def k_sym_N3(gamma, kappa=1.0):
        return kappa * 6.0 ** (-1.0 / gamma)

    def k_brk_N3(gamma, kappa=1.0):
        return kappa * 4.0 ** (-1.0 / gamma)

    gc = np.log(3/2) / np.log(2)
    gb = 1.0

    g_left = np.linspace(gamma_min, gc, n)
    g_mid  = np.linspace(gc, gb, n)

    # Branch values
    ksym_left = k_sym_N3(g_left, kappa)
    kbrk_left = k_brk_N3(g_left, kappa)
    kbrk_mid  = k_brk_N3(g_mid, kappa)
    color_brk = "#003366"
    color_sym = "#E69F00"

    # Symmetry-broken branch: k1*=k3*=x, k2*=0
    # solid for global (g < gc), dashed for local (gc < g < 1)
    ax.plot(g_left, kbrk_left, color=color_brk, lw=lw)
    ax.plot(g_left, np.zeros_like(g_left), color=color_brk, lw=lw)

    ax.plot(g_mid, kbrk_mid, color=color_brk, lw=lw, ls=(4, (4, 3)))
    ax.plot(g_mid, np.zeros_like(g_mid), color=color_brk, lw=lw, ls=(4, (4, 3)))

    # Symmetric branch: k1*=k2*=k3*=y
    # dashed for local (g < gc), solid for global (g > gc)
    ax.plot(g_left, ksym_left, color=color_sym, lw=lw, ls="--")
    ax.plot(np.linspace(gc, gamma_max, n), k_sym_N3(np.linspace(gc, gamma_max, n), kappa), color=color_sym, lw=lw)

    # Labels and style
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$k_e^\ast$")

    ax.plot([], [], color=color_brk, lw=lw, label="Symmetry-broken")
    ax.plot([], [], color=color_sym, lw=lw, label="Symmetric")

    ax.legend(loc=legend_loc, fontsize=legend_fontsize)

    ax.set_xlim(gamma_min, gamma_max)
    ax.set_ylim(-0.02 * k_sym_N3(gamma_max, kappa) / 0.25,
                1.05 * max(k_brk_N3(gamma_min, kappa), k_sym_N3(gamma_max, kappa)))
    ax.axvline(1.0, linestyle="dashdot", color='k', lw=guide_lw)
    ax.axvline(np.log(3/2)/np.log(2), linestyle=":", color='k', lw=guide_lw)

    return ax

# === Merged ring figure (old Figures 2, 3 and 4) ===

GAMMA_C_N3 = np.log(3.0 / 2.0) / np.log(2.0)  # exchange of global minima, ~0.585
GAMMA_B_N3 = 1.0                              # bifurcation of the broken branch


def stretched_xlim(x1, x2, f1, f2):
    """Return (xmin, xmax) placing data value x1 at axes fraction f1 and x2 at f2."""
    span = (x2 - x1) / (f2 - f1)
    xmin = x1 - f1 * span
    return xmin, xmin + span


def plot_capacity_values(capacity_values, N, beta_values, gamma_values,
                         ax, cax=None, fontsize=None, label_above=False, N_label_xy=(0.62, 0.96)):
    """Plot the ring phase diagram using the minimum edge capacity as the displayed observable.

    Pass ax (and optionally cax) to draw into an existing figure, which is what
    the merged figure does. With label_above=True the colour bar is titled above
    the bar rather than alongside it, keeping it clear of the next panel's
    y-axis label. The panel letter is not drawn here: in a composed figure it is
    positioned in figure coordinates alongside all the others.
    """
    fig = ax.get_figure()

    im = ax.imshow(capacity_values, aspect='auto', origin='lower', interpolation='nearest',
            extent=[gamma_values[0], gamma_values[-1], beta_values[0], beta_values[-1]], cmap='viridis')

    cbar = fig.colorbar(im, cax=cax) if cax is not None else fig.colorbar(im, ax=ax)

    formatter = ScalarFormatterOneDecimal(useMathText=True)
    formatter.set_powerlimits((0, 0))   # always scientific notation
    cbar.formatter = formatter
    cbar.update_ticks()
    fig.canvas.draw()

    cbar_label = r'$\min_e(k^*_e)$'
    cbar.ax.tick_params(labelsize=fontsize)
    offset = cbar.ax.yaxis.get_offset_text()
    if fontsize is not None:
        offset.set_size(fontsize)

    if label_above:
        # The offset text ("x10^-1") is drawn at the top of the bar, which is
        # where the title goes, so merge the two and hide the original.
        offset_text = offset.get_text()
        offset.set_visible(False)
        title = cbar_label if not offset_text else f"{cbar_label} {offset_text}"
        cbar.ax.set_title(title, fontsize=fontsize, pad=3)
    else:
        cbar.set_label(cbar_label, fontsize=fontsize)

    ax.set_xlabel(r'$\gamma$', fontsize=fontsize)
    ax.set_ylabel(r'$\beta$', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    if N_label_xy is None:
        ax.set_title(rf'$N={N}$', fontsize=fontsize, pad=3)
    else:
        ax.text(N_label_xy[0], N_label_xy[1], rf'$N={N}$', transform=ax.transAxes,
                fontsize=fontsize, color='white', ha='left', va='top')

    return fig, ax

def draw_ring_inset(G, ax, color="white", node_size=500, edge_scale=8, lw_scale=1.0, bg=None):
    """Draw a weighted 4-, 6-, or 8-node ring inset on an existing axis.

    Edge widths follow the quantised rule floor(1 + edge_scale * k/k_max),
    multiplied by lw_scale. The rule is kept as-is so the insets keep their
    original look; lw_scale exists only to bring the resulting point widths down
    when the figure is drawn at true printed size rather than oversized.
    """

    num_nodes = G.number_of_nodes()

    if num_nodes == 4:
        pos = {
            0: (0, 0),
            1: (-0.15, 0),
            2: (-0.15, -0.15),
            3: (0, -0.15),
        }
        xlim = (-0.2, 0.2)
        ylim = (-0.2, 0.2)

    elif num_nodes == 6:
        pos = {
            0: (0, 0.2),
            1: (-0.173, 0.1),
            2: (-0.173, -0.1),
            3: (0, -0.2),
            4: (0.173, -0.1),
            5: (0.173, 0.1),
        }
        xlim = (-0.25, 0.25)
        ylim = (-0.25, 0.25)

    elif num_nodes == 8:
        pos = {
            0: (0, 0.2),
            1: (-0.141, 0.141),
            2: (-0.2, 0),
            3: (-0.141, -0.141),
            4: (0, -0.2),
            5: (0.141, -0.141),
            6: (0.2, 0),
            7: (0.141, 0.141),
        }
        xlim = (-0.25, 0.25)
        ylim = (-0.25, 0.25)

    else:
        raise ValueError("draw_ring_inset only supports 4, 6, or 8 nodes.")

    if G.number_of_edges() == 0:
        ew_max = 1.0
    else:
        ew_max = max(G.edges[u, v].get("weight", 1.0) for u, v in G.edges())
        if ew_max <= 0:
            ew_max = 1.0

    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ew = G.edges[u, v].get("weight", 1.0)

        if ew <= 1e-9:
            continue

        lw = np.floor(1 + edge_scale * ew / ew_max) * lw_scale

        ax.plot([x1, x2], [y1, y2], linewidth=lw, color=color, zorder=0)

    for u in G.nodes():
        x, y = pos[u]
        marker = "s" if u % 2 == 0 else "o"
        ax.scatter(x, y, marker=marker, s=node_size, c=color, zorder=10, clip_on=False)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    if bg is None:
        ax.axis("off")
    else:
        # keep the patch so the inset masks whatever it sits on, but drop every
        # axis decoration; used where an inset has to overlap a guide line
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_facecolor(bg)

def draw_weighted_network_on_ax(G, ax, pos, color="black", edge_scale=8, node_size=120, lw_scale=0.5):
    """Draw a weighted network with prescribed node positions on an existing axis."""
    ew_max = max([G.edges[u, v]["weight"] for u, v in G.edges()])
    if ew_max <= 0:
        ew_max = 1.0

    for u, v in G.edges():
        ew = G.edges[u, v]["weight"]

        # hide truly vanishing edges
        if ew <= 1e-14:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        lw = np.floor(1 + edge_scale * ew / ew_max) * lw_scale
        ax.plot([x1, x2], [y1, y2], linewidth=float(lw), color=color, zorder=0)

    for u in G.nodes():
        x, y = pos[u]
        marker = "s" if u % 2 == 0 else "o"
        ax.scatter(x, y, marker=marker, s=node_size, c=color, zorder=10, clip_on=False)

    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_aspect("equal")
    ax.axis("off")

def build_optimized_ring_graph(N, beta, gamma, mu, threshold=1e-9):
    """Build the numerically optimized ring graph for use in phase-diagram insets."""

    G = build_ring_graph(N)

    MSM = compute_ring_second_moment_matrix(mu, beta, N)

    num_edges = G.number_of_edges()
    E = nx.incidence_matrix(G, oriented=True)

    # Solve the full optimization problem numerically
    k_min, D_min = random_minimum(E, G, MSM, gamma)

    # Store only active edges and attach optimized capacities as weights
    Gres = G.copy()
    for e in range(num_edges):
        f = np.argmax(E[:, [e]])
        t = np.argmin(E[:, [e]])

        if k_min[e] < threshold:
            Gres.remove_edge(f, t)
        else:
            Gres.edges[f, t]["weight"] = k_min[e]

    return Gres, k_min, D_min

def add_network_inset(ax, G, beta, gamma, size=0.25, color="white", node_size=500, lw_scale=1.0, bg=None):
    """Add a ring-network inset centered at the data point (gamma, beta)."""    
    # Convert data -> axis coordinates
    x_ax, y_ax = ax.transAxes.inverted().transform(ax.transData.transform((gamma, beta)))

    # Center inset on that point
    rect = [x_ax - size/2, y_ax - size/2, size, size]

    axins = ax.inset_axes(rect)

    draw_ring_inset(G, axins, color=color, node_size=node_size, lw_scale=lw_scale, bg=bg)

    return axins

def plot_curves_fixed_betas(fixed_beta, gamma_values, minK, branch_labels,
                            ax, colors, fontsize=None, lw=2.0, jump_lw=1.8,
                            legend_loc='best', jump_positions=None, break_at_changes=True, break_positions=None):
    """Plot minimum capacities versus gamma for selected beta values, marking branch changes.

    Pass ax to draw into an existing figure.
    """
    fig = ax.get_figure()

    all_jump_positions = []

    for c, beta in enumerate(fixed_beta):
        y = np.asarray(minK[c, :], dtype=float)

        # Extract 1D labels for this fixed beta
        labels_1d = np.array(branch_labels[c], dtype=object)

        # Smoothing against flickers
        labels_smooth = labels_1d.copy()
        for i in range(1, len(labels_smooth) - 1):
            if labels_smooth[i-1] == labels_smooth[i+1] != labels_smooth[i]:
                labels_smooth[i] = labels_smooth[i-1]

        # Detect true branch changes
        change_idx = np.where(labels_smooth[:-1] != labels_smooth[1:])[0]
        if jump_positions is None:
            jp = 0.5 * (gamma_values[change_idx] + gamma_values[change_idx + 1])
            jp = jp[:2]
        else:
            jp = np.asarray(jump_positions, dtype=float)

        # Break the colored curve at jumps
        y_plot = y.copy()
        if break_positions is not None:
            # split only at genuine discontinuities, so the curve is not drawn
            # with a vertical connector across a jump
            for xb in np.asarray(break_positions, dtype=float):
                # first sample past the boundary; blanking it drops the segment
                # that would otherwise join the two branches vertically
                k = int(np.searchsorted(np.asarray(gamma_values), xb))
                if 0 < k < len(y_plot):
                    y_plot[k] = np.nan
        elif break_at_changes:
            for j in change_idx:
                y_plot[j + 1] = np.nan

        ax.plot(gamma_values, y_plot, color=colors[c], lw=lw, label=rf'$\beta = {beta}$')

        # Draw dashed vertical lines at actual branch transitions
        for xj in jp:
            ax.axvline(x=xj, color="black", lw=jump_lw, ls="--", zorder=0)
        all_jump_positions.append(jp)

    ax.set_xlabel(r'$\gamma$', fontsize=fontsize)
    ax.set_ylabel(r'$\min_e(k^*_e)$', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.legend(loc=legend_loc, fontsize=fontsize)

    return fig, ax, all_jump_positions

def plot_curves_fixed_gammas(beta_values, fixed_gamma, minK, branch_labels,
                             ax, colors, fontsize=None, lw=2.0, jump_lw=2.0,
                             legend_loc='lower center', jump_positions=None, break_at_changes=True, break_positions=None):
    """Plot minimum capacities versus beta for selected gamma values, marking branch changes.

    Pass ax to draw into an existing figure.
    """
    fig = ax.get_figure()

    all_jump_positions = []

    for c, gamma in enumerate(fixed_gamma):
        y = np.asarray(minK[:, c], dtype=float)

        # Extract 1D labels for this fixed gamma
        labels_1d = np.array([row[c] for row in branch_labels], dtype=object)

        # Smoothing against flickers
        labels_smooth = labels_1d.copy()
        for i in range(1, len(labels_smooth) - 1):
            if labels_smooth[i-1] == labels_smooth[i+1] != labels_smooth[i]:
                labels_smooth[i] = labels_smooth[i-1]

        # Detect true branch changes
        change_idx = np.where(labels_smooth[:-1] != labels_smooth[1:])[0]
        if jump_positions is None:
            jp = 0.5 * (beta_values[change_idx] + beta_values[change_idx + 1])
            jp = np.array([jp[0], jp[1], jp[-2], jp[-1]])
        else:
            jp = np.asarray(jump_positions, dtype=float)

        # Break the curve at jumps
        y_plot = y.copy()
        if break_positions is not None:
            # split only at genuine discontinuities, so the curve is not drawn
            # with a vertical connector across a jump
            for xb in np.asarray(break_positions, dtype=float):
                # first sample past the boundary; blanking it drops the segment
                # that would otherwise join the two branches vertically
                k = int(np.searchsorted(np.asarray(beta_values), xb))
                if 0 < k < len(y_plot):
                    y_plot[k] = np.nan
        elif break_at_changes:
            for j in change_idx:
                y_plot[j + 1] = np.nan

        ax.plot(beta_values, y_plot, color=colors[c], lw=lw, label=rf'$\gamma = {gamma}$')

        # Draw one dashed vertical line per actual branch transition
        for xj in jp:
            ax.axvline(x=xj, color="black", lw=jump_lw, ls="--", zorder=0)

    ax.set_xlabel(r'$\beta$', fontsize=fontsize)
    ax.set_ylabel(r'$\min_e(k^*_e)$', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.legend(loc=legend_loc, fontsize=fontsize)

    return fig, ax

def add_network_inset_xy(ax, G, x, y, size=0.25, color="white", node_size=500, lw_scale=1.0, bg=None):
    """Add a ring-network inset centered at generic data coordinates (x, y)."""
    # convert data -> axis coordinates
    x_ax, y_ax = ax.transAxes.inverted().transform(ax.transData.transform((x, y)))

    rect = [x_ax - size/2, y_ax - size/2, size, size]
    axins = ax.inset_axes(rect)

    draw_ring_inset(G, axins, color=color, node_size=node_size, lw_scale=lw_scale, bg=bg)

    return axins