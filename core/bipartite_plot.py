"""Figure of the complete bipartite network K_{Ng,Nc}."""

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

from . import figstyle
from .bipartite_calc import NONZERO_TOL, gamma_s_loop, gamma_s_sink

C_SRC, C_SNK = "#1f77b4", "#ff7f0e"


def draw_bipartite(ax, k, Ng, Nc, lw_max=2.6, node_size=(20, 24)):
    """Sources (squares) on the left, sinks (circles) on the right, line width ~ k_e / max k_e.

    Nodes are reordered so that equivalent configurations look alike: sinks by decreasing
    total capacity, so a hub is always the top sink, and each source next to the sink it is
    most strongly connected to. Relabelling sources and sinks is a symmetry, so this changes
    only the drawing.
    """
    K = np.asarray(k).reshape(Ng, Nc) / np.max(k)

    # sinks: strongest first, so a hub sits at the top
    K = K[:, np.argsort(-K.sum(axis=0), kind="stable")]
    # sources: grouped by their strongest sink, top to bottom, then by that edge's capacity
    strongest = K.argmax(axis=1)
    order = np.lexsort((-K.sum(axis=1), -K[np.arange(Ng), strongest], strongest))
    K = K[order]

    ys, yc = -np.linspace(0, 1, Ng), -np.linspace(0, 1, Nc)
    for i in range(Ng):
        for j in range(Nc):
            if K[i, j] > NONZERO_TOL:
                ax.plot([0, 1], [ys[i], yc[j]], color="k", lw=lw_max * K[i, j],
                        solid_capstyle="round", zorder=1)
    ax.scatter(np.zeros(Ng), ys, marker="s", s=node_size[0], c=C_SRC, zorder=3, linewidths=0)
    ax.scatter(np.ones(Nc), yc, marker="o", s=node_size[1], c=C_SNK, zorder=3, linewidths=0)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-1.12, 0.12)
    ax.axis("off")

def plot_figure_bipartite(results, gallery, n_strong, Ng, Nc, mu, savepath="Figure5.pdf",
                          labels=None, min_fraction=0.005, letter_offset=None):
    """Compose the bipartite figure: (a) min/max capacity, (b) number of active edges, with
    the local stability boundaries of the loop and sink blocks, and (c onwards) the optimal
    network of each phase in gallery, marked by its letter in (b).

    gallery and n_strong are the outputs of bipartite_calc.select_phases.
    With labels given, (b) also outlines distinct classes that share the same number of
    active edges, for classes covering at least min_fraction of the plane. letter_offset
    moves individual letters in (b), in points, e.g. {"j": (7, 0)}.
    Returns the figure and, for the caption, one tuple (letter, n_edges, gamma, beta) per
    gallery panel.
    """
    figstyle.apply()
    letter_offset = {} if letter_offset is None else letter_offset

    beta_star = (Nc / Ng) * mu
    gam, b_rel = results["gamma_values"], results["beta_values"] / beta_star

    #  layout, in mm
    width, height = 179.0, 104.0
    left = 12.0
    top_bottom, top_h, top_w = 48.0, 50.0, 64.0
    a_left, b_left = left, 104.0
    cb_w, cb_pad = 2.2, 1.5
    label_x = left - 10.0
    gal_bottom, gal_h = 12.0, 18.0
    gal_col = (width - 3.0 - label_x) / len(gallery)
    gal_w = 0.75 * gal_col

    def rect(x, y, w, h):
        return [x / width, y / height, w / width, h / height]

    fig = plt.figure(figsize=figstyle.mm(width, height))
    ax_a = fig.add_axes(rect(a_left, top_bottom, top_w, top_h))
    cax_a = fig.add_axes(rect(a_left + top_w + cb_pad, top_bottom, cb_w, top_h))
    ax_b = fig.add_axes(rect(b_left, top_bottom, top_w, top_h))
    cax_b = fig.add_axes(rect(b_left + top_w + cb_pad, top_bottom, cb_w, top_h))

    # (a) min/max capacity
    im_a = ax_a.pcolormesh(gam, b_rel, results["kmin_rel"], shading="auto", cmap="viridis",
                           vmin=0, vmax=1, rasterized=True)
    cb_a = fig.colorbar(im_a, cax=cax_a)
    cb_a.set_label(r"$\min_e k_e^*/\max_e k_e^*$")

    # (b) number of active edges
    m = Ng * Nc
    n_lo = int(results["n_active"].min())
    norm = BoundaryNorm(np.arange(n_lo - 0.5, m + 1.5), plt.get_cmap("viridis").N)
    im_b = ax_b.pcolormesh(gam, b_rel, results["n_active"], shading="auto", cmap="viridis",
                           norm=norm, rasterized=True)
    cb_b = fig.colorbar(im_b, cax=cax_b, ticks=np.arange(n_lo, m + 1))
    cb_b.ax.set_title(r"$\mathcal{N}^*$", pad=3)

    # boundaries between distinct classes with the same number of active edges,
    # which the N* colouring alone does not show
    if labels is not None:
        for n in np.unique(results["n_active"]):
            region = results["n_active"] == n
            classes = [c for c in np.unique(labels[region]) if c != -1 and (labels == c).mean() >= min_fraction]
            for c in classes[1:]:
                sep = np.ma.masked_where(~region, (labels == c).astype(float))
                ax_b.contour(gam, b_rel, sep, levels=[0.5], colors="w", linewidths=0.6)

    # local stability boundaries
    bb = np.logspace(np.log10(b_rel.min()), np.log10(b_rel.max()), 800)
    styles = [(gamma_s_loop(Ng, bb), dict(color="w", lw=1.0, ls=(0, (5, 2))), "loop block"),
              (gamma_s_sink(Ng, bb), dict(color="w", lw=1.0, ls=(0, (5, 1.5, 1, 1.5))), "sink block")]
    for ax in (ax_a, ax_b):
        for g_line, style, _ in styles:
            ok = g_line > 0
            ax.plot(g_line[ok], bb[ok], **style)
        ax.set_yscale("log")
        ax.set_xlim(gam.min(), gam.max())
        ax.set_ylim(b_rel.min(), b_rel.max())
        ax.set_xlabel(r"$\gamma$")
    ax_a.set_ylabel(r"$\beta/\beta^*$")
    ax_a.legend([Line2D([], [], **st) for _, st, _ in styles], [lab for _, _, lab in styles],
                loc="lower left", frameon=False, labelcolor="w", handlelength=2.4)

    # (c onwards) gallery, marked by letter in (b)
    letters = "cdefghijklmnop"[:len(gallery)]
    info = []
    axes_gal = []
    for n, p in enumerate(gallery):
        x0 = label_x + n * gal_col
        ax = fig.add_axes(rect(x0 + (gal_col - gal_w) / 2, gal_bottom, gal_w, gal_h))
        draw_bipartite(ax, p["k"], Ng, Nc)
        title = "symmetric" if p["symmetric"] else rf"$\mathcal{{N}}^*={p['n_edges']}$"
        fig.text((x0 + gal_col / 2) / width, (gal_bottom + gal_h + 0.5) / height, title,
                 ha="center", va="bottom")
        figstyle.panel_label(fig, letters[n], x0 / width, (gal_bottom + gal_h + 4.0) / height)
        axes_gal.append(ax)
        info.append((letters[n], p["n_edges"], p["gamma"], p["beta"]))

        ax_b.annotate(letters[n], (p["gamma"], p["beta"] / beta_star),
                      xytext=letter_offset.get(letters[n], (0, 0)), textcoords="offset points",
                      ha="center", va="center", color="w", fontweight="bold",
                      path_effects=[pe.withStroke(linewidth=1.2, foreground="k")])

    # brackets under the gallery: beta > beta* and beta < beta*
    n_sym = sum(p["symmetric"] for p in gallery)
    groups = [(range(n_sym, n_sym + n_strong), r"$\beta > \beta^*$"),
              (range(n_sym + n_strong, len(gallery)), r"$\beta < \beta^*$")]
    y_line = (gal_bottom - 2.0) / height
    for grp, lab in groups:
        if len(grp):
            x0 = (label_x + grp[0] * gal_col + 1.5) / width
            x1 = (label_x + (grp[-1] + 1) * gal_col - 1.5) / width
            fig.add_artist(Line2D([x0, x1], [y_line, y_line], color="0.4", lw=0.6))
            fig.text((x0 + x1) / 2, y_line - 0.5 / height, lab, ha="center", va="top")

    # panel labels
    figstyle.panel_label(fig, "a", label_x / width, (top_bottom + top_h + 4.0) / height)
    figstyle.panel_label(fig, "b", (b_left - 10.0) / width, (top_bottom + top_h + 4.0) / height)

    if savepath is not None:
        fig.savefig(savepath, format="pdf")

    return fig, info