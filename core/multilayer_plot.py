"""Figure of the extended multilayer network (Supplementary Figure, Supplementary Note 4)."""

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import figstyle
from .bipartite_calc import NONZERO_TOL
from .multilayer_calc import residual_group

C_SRC, C_SNK = "#1f77b4", "#ff7f0e"


def draw_multilayer_3d(ax, G, pos, k, lw_max=2.2, lw_min=0.35, src_size=9, snk_size=1.6,
                       plane_color="0.88", elev=22, azim=-60, zoom=1.25, active_tol=NONZERO_TOL):
    """Both layers as grey planes, active edges (k_e > active_tol * max k) with width and
    darkness increasing with k_e, sources as squares and sinks as circles. Inactive edges are
    not drawn."""
    ax.computed_zorder = False     # draw by zorder; otherwise the planes veil everything below them
    P = np.array([pos[u] for u in G.nodes()], float)
    x0, x1, y0, y1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    pad = 0.35
    for z in np.unique(P[:, 2]):
        quad = [[x0 - pad, y0 - pad, z], [x1 + pad, y0 - pad, z],
                [x1 + pad, y1 + pad, z], [x0 - pad, y1 + pad, z]]
        ax.add_collection3d(Poly3DCollection([quad], facecolor=plane_color, edgecolor="none",
                                             alpha=0.6, zorder=0))

    k = np.asarray(k, float)
    w = k / k.max()
    for (u, v), we in sorted(zip(G.edges(), w), key=lambda t: t[1]):   # strong edges on top
        if we <= active_tol:
            continue
        (xa, ya, za), (xb, yb, zb) = pos[u], pos[v]
        ax.plot([xa, xb], [ya, yb], [za, zb], color=str(0.55 * (1 - we)),
                lw=lw_min + (lw_max - lw_min) * we, solid_capstyle="round", zorder=2)

    s = np.array([G.nodes[u]["s"] for u in G.nodes()])
    ax.scatter(*P[s <= 0].T, marker="o", s=snk_size, c=C_SNK, depthshade=False, linewidths=0, zorder=3)
    ax.scatter(*P[s > 0].T, marker="s", s=src_size, c=C_SRC, depthshade=False, linewidths=0, zorder=4)

    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_zlim(-0.1, 1.1)
    ax.set_box_aspect((x1 - x0 + 2 * pad, y1 - y0 + 2 * pad, 2.2), zoom=zoom)
    ax.set_axis_off()
    ax.set_facecolor((1, 1, 1, 0))
    ax.view_init(elev=elev, azim=azim)


def _discrete(ax, cax, gam, bet, Z, cmap="viridis_r"):
    vals = np.unique(Z[np.isfinite(Z)]).astype(int)
    bounds = np.concatenate(([vals[0] - 0.5], (vals[:-1] + vals[1:]) / 2, [vals[-1] + 0.5]))
    norm = BoundaryNorm(bounds, plt.get_cmap(cmap).N)
    im = ax.pcolormesh(gam, bet, Z, shading="auto", cmap=cmap, norm=norm, rasterized=True)
    cb = ax.get_figure().colorbar(im, cax=cax, ticks=(bounds[:-1] + bounds[1:]) / 2)
    cb.set_ticklabels([str(v) for v in vals])
    cb.ax.minorticks_off()
    return cb


def plot_figure_multilayer(results, measures, gallery, G, pos, types, edge_perm,
                           savepath="FigureS_multilayer.pdf", hatch_inter=5, inter_style="contour",
                           sym_rtol=1e-4, sym_support_tol=NONZERO_TOL, active_tol=NONZERO_TOL,
                           stipple_stride=2, stipple_size=0.8, n_col=4,
                           gamma_range=None, beta_range=None, letter_offset=None,
                           mark_panels=("a", "b", "c")):
    """(a) layer imbalance I, (b) max(N_A, N_B), with the region N_I = hatch_inter outlined
    (inter_style="contour"), hatched (inter_style="hatch") or stippled with a dot on every
    stipple_stride-th grid point (inter_style="stipple"), or not marked if hatch_inter is None,
    (c) coefficient
    of variation C of the active interlayer capacities, (d onwards) the optimal network at each
    gallery point, marked by its letter in the panels listed in mark_panels.

    measures is multilayer_calc.layer_measures(results, G, pos); gallery is a list of dicts with
    k, n_edges, gamma, beta (multilayer_calc.gallery_at or gallery_auto). Returns the figure and,
    for the caption, one tuple (letter, N*, Gamma*, gamma, beta) per gallery panel.
    """
    figstyle.apply()
    mpl.rcParams["hatch.linewidth"] = 0.4
    mpl.rcParams["hatch.color"] = "k"
    letter_offset = {} if letter_offset is None else letter_offset

    gam, bet = results["gamma_values"], results["beta_values"]
    gamma_range = (gam.min(), gam.max()) if gamma_range is None else gamma_range
    beta_range = (bet.min(), bet.max()) if beta_range is None else beta_range

    # layout, in mm
    width, height = 179.0, 104.0
    pd_w, pd_h, pd_bottom = 36.0, 38.0, 60.0
    pd_lefts = (12.0, 71.0, 130.0)
    cb_w, cb_pad = 2.2, 1.5
    n_row = int(np.ceil(len(gallery) / n_col))
    label_x = 2.0
    gal_col = (width - label_x) / n_col
    gal_w, gal_h = 30.0, 21.0
    gal_row_h = (pd_bottom - 14.0) / max(n_row, 1)

    def rect(x, y, w, h):
        return [x / width, y / height, w / width, h / height]

    fig = plt.figure(figsize=figstyle.mm(width, height))
    axes, caxes = [], []
    for x in pd_lefts:
        axes.append(fig.add_axes(rect(x, pd_bottom, pd_w, pd_h)))
        caxes.append(fig.add_axes(rect(x + pd_w + cb_pad, pd_bottom, cb_w, pd_h)))
    ax_a, ax_b, ax_c = axes

    # (a) layer imbalance
    im = ax_a.pcolormesh(gam, bet, measures["I"], shading="auto", cmap="viridis_r",
                         vmin=0, rasterized=True)
    fig.colorbar(im, cax=caxes[0]).set_label(r"$\mathcal{I}$")

    # (b) active edges in the fuller layer, hatched where N_I = hatch_inter
    cb = _discrete(ax_b, caxes[1], gam, bet, measures["n_max"].astype(float))
    cb.set_label(r"$\max(\mathcal{N}_A,\mathcal{N}_B)$")
    if hatch_inter is not None:
        hatch = (measures["n_inter"] == hatch_inter).astype(float)
    if hatch_inter is not None and hatch.any():
        if inter_style == "stipple":
            sub = hatch[::stipple_stride, ::stipple_stride] > 0
            gg, bb = np.meshgrid(gam[::stipple_stride], bet[::stipple_stride])
            ax_b.scatter(gg[sub], bb[sub], s=stipple_size, c="k", marker="o", linewidths=0,
                         rasterized=True)
            handle = Line2D([], [], ls="none", marker="o", ms=1.5, color="k")
        elif inter_style == "hatch":
            ax_b.contourf(gam, bet, hatch, levels=[0.5, 1.5], colors="none", hatches=["////"])
            handle = Patch(facecolor="none", edgecolor="k", hatch="////", lw=0.4)
        else:   # outline of the region
            ax_b.contour(gam, bet, hatch, levels=[0.5], colors="w", linewidths=0.8, linestyles="--")
            handle = Line2D([], [], color="k", lw=0.8, ls="--")
        ax_b.legend([handle],
                    [rf"$\mathcal{{N}}_I={hatch_inter}$"], loc="lower right", bbox_to_anchor=(1.0, 1.0),
                    borderaxespad=0.1,
                    handlelength=1.2, handleheight=1.0)

    # (c) coefficient of variation of the active interlayer capacities
    im = ax_c.pcolormesh(gam, bet, measures["C"], shading="auto", cmap="viridis_r",
                         vmin=0, rasterized=True)
    fig.colorbar(im, cax=caxes[2]).set_label(r"$\mathcal{C}$")

    for ax in axes:
        ax.set_xlim(*gamma_range)
        ax.set_ylim(*beta_range)
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\beta$")

    # (d onwards) gallery
    letters = "defghijklmnop"[:len(gallery)]
    info = []
    for n, p in enumerate(gallery):
        r, c = divmod(n, n_col)
        x0 = label_x + c * gal_col
        y0 = pd_bottom - 12.0 - (r + 1) * gal_row_h + 4.0
        ax3 = fig.add_axes(rect(x0 + (gal_col - gal_w) / 2, y0, gal_w, gal_h), projection="3d")
        draw_multilayer_3d(ax3, G, pos, p["k"], active_tol=active_tol)
        grp = residual_group(p["k"], types, edge_perm, sym_rtol, sym_support_tol)
        fig.text((x0 + gal_col / 2) / width, (y0 - 0.2) / height,
                 rf"$\mathcal{{N}}^*={p['n_edges']},\ \Gamma^*={grp}$", ha="center", va="top")
        figstyle.panel_label(fig, letters[n], x0 / width, (y0 + gal_h - 1.0) / height)
        info.append((letters[n], p["n_edges"], grp, p["gamma"], p["beta"]))
        for name, ax in zip("abc", axes):
            if name in mark_panels:
                ax.annotate(letters[n], (p["gamma"], p["beta"]),
                            xytext=letter_offset.get(letters[n], (0, 0)),
                            textcoords="offset points", ha="center", va="center", color="w",
                            fontweight="bold",
                            path_effects=[pe.withStroke(linewidth=1.2, foreground="k")])

    for letter, x in zip("abc", pd_lefts):
        figstyle.panel_label(fig, letter, (x - 10.0) / width, (pd_bottom + pd_h + 5.0) / height)

    if savepath is not None:
        fig.savefig(savepath, format="pdf")
    return fig, info