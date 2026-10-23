import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import figstyle
from .cube_calc import (classify_network, gamma_s_T1, gamma_s_T2_minus, gamma_s_alt_ring,
                        phase_sequence, representative_point)

# === Figure 4 ===

def plot_phase_diagram(gamma_values, beta_values, num_nonzero_edges, ax, cax, cmap="viridis"):
    """Plot the phase diagram showing the number of active edges across parameter space."""

    # Identify discrete values from the data
    discrete_values = np.unique(num_nonzero_edges[np.isfinite(num_nonzero_edges)])

    # Build boundaries so each integer value is centered in its color bin
    vmin = discrete_values.min() - 0.5
    vmax = discrete_values.max() + 0.5
    boundaries = np.arange(vmin, vmax + 1)

    norm = BoundaryNorm(boundaries, ncolors=plt.get_cmap(cmap).N, clip=True)

    # Transpose so axes align with (gamma horizontal, beta vertical)
    im = ax.imshow(num_nonzero_edges.T, origin="lower", aspect="auto", interpolation="nearest",
                   extent=[gamma_values[0], gamma_values[-1], beta_values[0], beta_values[-1]],
                   cmap=cmap, norm=norm)

    # Use only values that actually appear as colorbar ticks
    cbar = ax.get_figure().colorbar(im, cax=cax, ticks=discrete_values)
    cbar.set_ticklabels([str(int(v)) for v in discrete_values])
    cbar.ax.set_title(r"$\mathcal{N}^*$", pad=3)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\beta$")

    return im, cbar


def graph_from_capacities(G, k, nonzero_tol=1e-8):
    """Copy of G with capacities as edge weights and inactive edges removed.

    The entries of k follow the order of G.edges(), which is the column order of
    nx.incidence_matrix and therefore of the capacity vectors returned by the scans.
    """
    Gres = G.copy()
    for (u, v), ke in zip(list(G.edges()), k):
        if ke < nonzero_tol:
            Gres.remove_edge(u, v)
        else:
            Gres.edges[u, v]["weight"] = ke
    return Gres


def draw_network_3d(ax, G, pos, node_size=40, lw_scale=1.0):
    """Draw a weighted multiplex network in 3D with node types and edge thickness encoding capacities."""
    ew_max = max(G.edges[u, v].get("weight", 0.0) for u, v in G.edges())
    if ew_max <= 0:
        ew_max = 1.0

    # Nodes
    for u in G.nodes():
        x, y, z = pos[u]
        if G.nodes[u]["s"] > 0:
            ax.scatter(x, y, z, marker="s", s=node_size, c="#1f77b4", depthshade=False)
        else:
            ax.scatter(x, y, z, marker="o", s=node_size, c="#ff7f0e", depthshade=False)

    # Edges
    for u, v in G.edges():
        x1, y1, z1 = pos[u]
        x2, y2, z2 = pos[v]
        ew = G.edges[u, v].get("weight", 0.0)
        lw = (1 + 6 * ew / ew_max) * lw_scale
        ax.plot([x1, x2], [y1, y2], [z1, z2], linewidth=lw, color=str(0.5 - 0.5 * ew / ew_max))

    # Layout
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.set_facecolor((1, 1, 1, 0))
    ax.view_init(elev=22, azim=-60)


def plot_k_vs_gamma_with_insets(gamma_values, k_values, num_nonzero_edges, ax, G, pos,
                                inset_gammas=None, inset_rects=None, inset_texts=None,
                                lw=0.8, jump_lw=0.5, node_size=4, lw_scale=0.25,
                                legend_kw=None):
    """Plot optimized edge capacities versus gamma with inset network visualizations.

    inset_rects are [x0, y0, w, h] in axes coordinates of ax. Each inset shows the optimal
    network at the gamma of the curve closest to the requested inset gamma, and its text is
    placed centred below it.
    """
    gamma_values = np.asarray(gamma_values)
    k_values = np.asarray(k_values)
    num_nonzero_edges = np.asarray(num_nonzero_edges)

    # Detect jumps
    jumps = np.where(np.diff(num_nonzero_edges) != 0)[0]
    jump_gamma_values = gamma_values[jumps + 1]

    # Sort k values
    sorted_k_values = np.sort(k_values, axis=1)
    num_edges = sorted_k_values.shape[1]

    # Edge colors
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, num_edges))

    # Edge capacities, numbered from 1 as in the main text
    for e in range(num_edges):
        ax.plot(gamma_values, sorted_k_values[:, e], alpha=0.6, label=f"e={e + 1}",
                color=colors[e], lw=lw)

    # Vertical jump lines
    for x in jump_gamma_values:
        ax.axvline(x=x, color="k", linestyle="--", linewidth=jump_lw)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$k_e^*$")

    # Inset networks, placed in axes coordinates of ax
    if inset_gammas is not None and inset_rects is not None:
        fig = ax.get_figure()
        bbox = ax.get_position()
        for n, (rect, gamma) in enumerate(zip(inset_rects, inset_gammas)):
            fig_rect = [bbox.x0 + rect[0] * bbox.width, bbox.y0 + rect[1] * bbox.height,
                        rect[2] * bbox.width, rect[3] * bbox.height]
            ax_in = fig.add_axes(fig_rect, projection="3d")
            i = int(np.argmin(np.abs(gamma_values - gamma)))
            draw_network_3d(ax_in, graph_from_capacities(G, k_values[i]), pos,
                            node_size=node_size, lw_scale=lw_scale)
            if inset_texts is not None:
                ax.text(rect[0] + rect[2] / 2, rect[1], inset_texts[n], transform=ax.transAxes,
                        ha="center", va="top")

    if legend_kw is not None:
        ax.legend(**legend_kw)

    return ax

def plot_figure_cube(fine_results, labels, phase_names, cut, G, pos, node_perm, edge_perm,
                     savepath="Figure4.pdf",
                     a_xmin=0.12, a_centers=(0.195, 0.41, 0.612, 0.755), a_label_top_mm=1.0,
                     a_line_mm=3.0, a_combine=("symmetric",), a_combine_label_x=0.69,
                     inset_mm=13.0, a_ypad=0.04, gal_shift=-1.0,
                     inset_style=None, pd_letter_offset=None):
    """Compose the cube figure: (a) capacities along one cut, (b)-(e) the remaining phases,
    (f) the phase diagram with stability thresholds.

    a_* arguments place the insets and labels of panel a: a_centers is the gamma at the centre
    of each inset, and phases in a_combine get N* and Gamma* on one line, left-aligned at
    a_combine_label_x. Inset sizes are in mm, letter offsets in points.

    Returns the figure and, for the caption, one tuple (letter, name, gamma, beta) per gallery panel.
    """
    inset_style = dict(node_size=4, lw_scale=0.25) if inset_style is None else inset_style
    pd_letter_offset = {"e": (-8, -1), "f": (-8, -3)} if pd_letter_offset is None else pd_letter_offset

    figstyle.apply()
    mpl.rcParams["hatch.linewidth"] = 0.4
    mpl.rcParams["hatch.color"] = "w"

    # Phase masks on the fine scan, shape (len(gamma), len(beta))
    gam, bet = fine_results["gamma_values"], fine_results["beta_values"]
    idx = {nm: phase_names.index(nm) for nm in phase_names}
    is_ring = labels == idx["ring"]
    is_hex = labels == idx["kite"]
    alt = is_ring & (gam[:, None] < gamma_s_alt_ring(bet[None, :]))

    gallery = [("ring, alternating", alt),
               ("kite", is_hex),
               ("ten edges", labels == idx["ten"]) if "ten" in idx else None,
               ("eleven edges", labels == idx["eleven"]) if "eleven" in idx else None]
    gallery = [g for g in gallery if g is not None and g[1].any()]

    # Layout, in mm
    width, height = 179.0, 86.0
    left = 12.0
    col_w = 80.0                                             # panel a
    a_bottom, a_h = 48.0, 32.0
    label_x = left - 10.0                                    # x of the panel letters a and b
    gal_bottom = 24.0                                        # bottom of the gallery cubes
    gal_col = (left + col_w - label_x) / len(gallery)        # gallery spans from letter a to the right edge of a
    pd_size = 64.0
    pd_left, pd_bottom = 104.0, a_bottom + a_h - pd_size     # phase diagram, top aligned with panel a
    cb_w, cb_pad = 2.2, 1.5

    def rect(x, y, w, h):
        return [x / width, y / height, w / width, h / height]

    fig = plt.figure(figsize=figstyle.mm(width, height))

    # (a): capacities along the cut 
    ax_a = fig.add_axes(rect(left, a_bottom, col_w, a_h))
    seq = phase_sequence(cut, node_perm, edge_perm)
    g_cut, k_cut = cut["gamma_values"], cut["k_values"][:, 0, :]
    a_xmax = g_cut[-1]

    def gamma_to_axes(g):
        return (g - a_xmin) / (a_xmax - a_xmin)

    # each inset hangs directly below its own label: one line or two
    w_ax, h_ax = inset_mm / col_w, inset_mm / a_h
    a_rects = []
    for (nm, _, _), gc in zip(seq, a_centers):
        rows = 1 if nm in a_combine else 2
        top = 1.0 - (a_label_top_mm + rows * a_line_mm) / a_h
        a_rects.append([gamma_to_axes(gc) - w_ax / 2, top - h_ax, w_ax, h_ax])

    plot_k_vs_gamma_with_insets(g_cut, k_cut, cut["num_nonzero_edges"][:, 0], ax_a, G, pos,
                                inset_gammas=[gm for _, gm, _ in seq], inset_rects=a_rects,
                                legend_kw=dict(loc="lower right", ncol=3, fontsize=figstyle.FONT_ANNOT,
                                               handlelength=1.0, columnspacing=0.8),
                                **inset_style)

    y_top = 1.0 - a_label_top_mm / a_h
    for r, (nm, gm, grp) in zip(a_rects, seq):
        n_active = int(np.sum(k_cut[np.argmin(np.abs(g_cut - gm))] > 1e-8))
        if nm in a_combine:
            ax_a.text(gamma_to_axes(a_combine_label_x), y_top,
                      rf"$\mathcal{{N}}^*={n_active},\ \Gamma^*={grp}$",
                      transform=ax_a.transAxes, ha="left", va="top")
        else:
            ax_a.text(r[0] + r[2] / 2, y_top,
                      rf"$\mathcal{{N}}^*={n_active}$" + "\n" + rf"$\Gamma^*={grp}$",
                      transform=ax_a.transAxes, ha="center", va="top")

    kmax = k_cut.max()
    ax_a.set_xlim(a_xmin, a_xmax)
    ax_a.set_ylim(-a_ypad * kmax, (1 + a_ypad) * kmax)

    # (b)-(e): remaining phases, each at a representative point of the scan 
    gal_letters = "cdefgh"[:len(gallery)]
    gal_info = []
    for n, (name, mask) in enumerate(gallery):
        i, j = representative_point(mask)
        k_pt = fine_results["k_values"][i, j]
        _, n_active, group = classify_network(k_pt, node_perm, edge_perm)
        gal_info.append((gal_letters[n], name, gam[i], bet[j]))

        x0 = label_x + n * gal_col # left edge of this gallery column
        xc = x0 + gal_col / 2
        ax3d = fig.add_axes(rect(xc - inset_mm / 2 + gal_shift, gal_bottom, inset_mm, inset_mm),
                            projection="3d")
        draw_network_3d(ax3d, graph_from_capacities(G, k_pt), pos, **inset_style)
        fig.text(xc / width, (gal_bottom - 0.5) / height,
                 f"$\\mathcal{{N}}^*={n_active}$\n$\\Gamma^*={group}$",
                 ha="center", va="top")
        figstyle.panel_label(fig, gal_letters[n], x0 / width, (gal_bottom + inset_mm) / height)

    # (f): phase diagram
    ax_f = fig.add_axes(rect(pd_left, pd_bottom, pd_size, pd_size))
    cax_f = fig.add_axes(rect(pd_left + pd_size + cb_pad, pd_bottom, cb_w, pd_size))
    plot_phase_diagram(gam, bet, fine_results["num_nonzero_edges"], ax_f, cax_f)

    # beta_h: ring / kite boundary, only where the two eight-edge phases meet
    bh_style = dict(color="w", lw=1.2)
    sep = np.ma.masked_where(~(is_ring | is_hex), is_hex.astype(float))
    ax_f.contour(gam, bet, sep.T, levels=[0.5], colors="w", linewidths=bh_style["lw"])

    # alternating ring
    ax_f.contourf(gam, bet, alt.T.astype(float), levels=[0.5, 1.5], colors="none", hatches=["xxxx"])

    # local stability thresholds
    bb = np.linspace(bet[0], bet[-1], 600)
    lines = [(gamma_s_T1, dict(color="w", lw=1.2, ls=(0, (5, 2))), r"$\gamma_s(T_1)$"),
             (gamma_s_T2_minus, dict(color="w", lw=1.2, ls=(0, (5, 1.5, 1, 1.5))), r"$\gamma_s(T_2^-)$"),
             (gamma_s_alt_ring, dict(color="w", lw=1.2, ls=(0, (1, 1.2))), r"$\gamma_s(\mathrm{alt})|_{N=4}$")]
    handles, texts = [], []
    for f, style, label in lines:
        y = f(bb)
        m = (y > gam[0]) & (y < gam[-1])
        ax_f.plot(y[m], bb[m], **style)
        handles.append(Line2D([], [], **style))
        texts.append(label)
    handles.append(Line2D([], [], **bh_style))
    texts.append(r"$\beta_h$")
    handles.append(Patch(facecolor="none", edgecolor="w", hatch="xxxx", lw=0.4))
    texts.append("alt. ring")

    # points shown in the gallery panels
    for letter, _, g_pt, b_pt in gal_info:
        ax_f.plot(g_pt, b_pt, "o", ms=2.0, mfc="w", mec="k", mew=0.4, zorder=6)
        ax_f.annotate(letter, (g_pt, b_pt), xytext=pd_letter_offset.get(letter, (2, 2)),
                      textcoords="offset points", color="w", fontweight="bold",
                      path_effects=[pe.withStroke(linewidth=1.2, foreground="k")])

    ax_f.set_xlim(gam[0], gam[-1])
    ax_f.set_ylim(bet[0], bet[-1])
    ax_f.legend(handles, texts, loc="upper left", ncol=1, handlelength=2.4, frameon=False,
                labelcolor="w")

    # panel labels
    pd_letter = "b"
    figstyle.panel_label(fig, "a", label_x / width, (a_bottom + a_h + 4.0) / height)
    figstyle.panel_label(fig, pd_letter, (pd_left - 10.0) / width, (pd_bottom + pd_size + 4.0) / height)

    if savepath is not None:
        fig.savefig(savepath, format="pdf")

    return fig, gal_info