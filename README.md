# Symmetry breaking in minimum dissipation networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23729507.svg)](https://doi.org/10.5281/zenodo.23729507)

Code and data accompanying the publication

> A. Parameswaran, P. Ebrahimzadeh, A. Benigni, D. Witthaut, and I. Bačić,
> **Symmetry breaking in minimum dissipation networks** (manuscript under review)

## Overview

We study optimal supply networks whose edge capacities minimize the average dissipation of linear flows under a global resource constraint, $\sum_e k_e^\gamma \le \kappa^\gamma$, with fluctuating injections. The repository contains the code that computes locally optimal networks with the self-consistent iteration of Corson (Phys. Rev. Lett. 104, 048703, 2010), scans the ($\gamma$, $\beta$) parameter space, classifies the optimal networks as symmetric, weakly symmetry-broken or strongly symmetry-broken, and reproduces all numerical figures of the paper. The model networks are rings with alternating sources and sinks, the cube, the complete bipartite network $K_{4,4}$, an extended two-layer lattice, and a six-node ring driven by measured wind and solar generation.

## Repository structure

```
.
├── core/                   Python module with all computational and plotting routines
│   ├── corson_algorithm.py   self-consistent iteration for locally optimal capacities
│   ├── rings_calc.py         ring networks: dissipation, branches, stability thresholds, scaling
│   ├── cube_calc.py          cube network: scans, symmetry classification, stability thresholds
│   ├── bipartite_calc.py     complete bipartite networks K_{Ng,Nc}: scans and phase classification
│   ├── multilayer_calc.py    extended multilayer network: (parallel) coarse and fine scans
│   ├── *_plot.py             figure assembly for each network class
│   └── figstyle.py           common figure style
├── scan_data/              precomputed parameter scans (.npz) for the cube, K_{4,4} and multilayer networks
├── energy_data/            hourly wind and PV capacity factors from renewables.ninja (Jülich, 2019)
├── intro_figure.ipynb      Fig. 1
├── rings.ipynb             Figs. 2 and 3, Supplementary Fig. 2
├── cube.ipynb              Fig. 4
├── bipartite.ipynb         Fig. 5 and the data for Supplementary Table 10
├── energy_grids.ipynb      Fig. 6
└── multilayer.ipynb        extended multilayer network (Supplementary Note 4)
```

A description of the algorithms, with pseudocode, is given in ALGORITHM.md.

## System requirements

The code is written in Python and runs on a standard desktop or laptop computer. No non-standard hardware is required. It has been tested on Windows 11 with Python 3.10.14 and the following package versions:

| Package    | Version |
|------------|---------|
| numpy      | 1.26.4  |
| scipy      | 1.15.3  |
| matplotlib | 3.9.1   |
| networkx   | 3.4.2   |
| pandas     | 2.3.3   |
| ipykernel  | 6.29.5  |

`pandas` is only needed for `energy_grids.ipynb`.

## Installation

```bash
git clone https://github.com/i-bacic/optimal-networks.git
cd optimal-networks
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Installation takes a few minutes on a typical desktop computer. The code is not packaged. The notebooks import the `core` module from the repository root, so they must be started from that directory.

## Reproducing the results

Open the notebooks in any Jupyter front end, e.g. VS Code with the Jupyter extension, or JupyterLab (`pip install jupyterlab`, then run `jupyter lab` from the repository root). Run each notebook top to bottom.

| Notebook | Paper content | Typical run time |
|----------|---------------|------------------|
| `intro_figure.ipynb` | Fig. 1 | < 1 min |
| `rings.ipynb` | Figs. 2, 3; Supplementary Fig. 2 | 10 min |
| `cube.ipynb` | Fig. 4 | 5 min with cached scans, 5 h from scratch |
| `bipartite.ipynb` | Fig. 5; Supplementary Table 10 | 5 min with cached scans, 2 h from scratch |
| `energy_grids.ipynb` | Fig. 6 | 15 min  | 
| `multilayer.ipynb` | Supplementary Fig. 1 | 5 min with cached scans, 57 h from scratch on 11 worker processes |

Run times were measured on a 13th Gen Intel Core i7-1365U (1.80 GHz, 10 cores, 12 logical processors) with 32 GB RAM. Each notebook writes its figure as a PDF to the repository root.

**Cached parameter scans.** The phase diagrams of the cube, $K_{4,4}$ and multilayer networks require many runs of the self-consistent iteration. The corresponding notebooks load the precomputed scans from `scan_data/` if present and recompute them otherwise. To reproduce the scans from scratch, delete (or move) the relevant files in `scan_data/` before running the notebook. The multilayer scan runs in parallel over all but one logical processor.

**Multistability.** For $\gamma < 1$ the optimization problem is non-convex and may have several local minima, so the outcome of a single run of the self-consistent iteration depends on its initial condition. The scans therefore proceed in two stages: a coarse scan collects candidate configurations from several initial conditions, which are reduced to one representative per support class, and a fine scan starts from every representative at every parameter point and keeps the configuration with the lowest dissipation. All random initial conditions use a fixed seed (`SEED = 0`) for reproducibility. Changing the seed or the scan resolution can change the result in narrow regions near phase boundaries.

## Data

`energy_data/` contains hourly capacity factors for wind and solar PV generation at Jülich, Germany (50.9221° N, 6.3611° E) for the year 2019, obtained from [renewables.ninja](https://www.renewables.ninja) and described in

- S. Pfenninger and I. Staffell, *Energy* **114**, 1251–1265 (2016), doi: [10.1016/j.energy.2016.08.060](https://doi.org/10.1016/j.energy.2016.08.060)
- I. Staffell and S. Pfenninger, *Energy* **114**, 1224–1239 (2016), doi: [10.1016/j.energy.2016.08.068](https://doi.org/10.1016/j.energy.2016.08.068)

These data are redistributed under the renewables.ninja license ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)), not under the license of this repository.

## License

The code in this repository is released under the [MIT License](LICENSE).

Exception: the files in energy_data/ are third-party data from
renewables.ninja and are licensed under CC BY-NC 4.0
(https://creativecommons.org/licenses/by-nc/4.0/), not under the MIT License.

## Citation

If you use this code, please cite the associated paper and code:

```bibtex
@unpublished{parameswaran2026symmetry,
  author = {Parameswaran, Aarathi and Ebrahimzadeh, Pezhman and Benigni, Andrea and Witthaut, Dirk and Ba{\v{c}}i{\'c}, Iva},
  title  = {Symmetry breaking in minimum dissipation networks},
  note   = {Manuscript under review},
  year   = {2026}
}

@software{parameswaran2026code,
  author    = {Parameswaran, Aarathi and Ebrahimzadeh, Pezhman and Benigni, Andrea and Witthaut, Dirk and Ba{\v{c}}i{\'c}, Iva},
  title     = {optimal-networks: code for ``Symmetry breaking in minimum dissipation networks''},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.23729507}
}
```

## Funding

This work was supported by the German Federal Ministry of Research, Technology and Space (BMFTR) via the project 03SF0751, and in part by the Air Force Office of Scientific Research under grant FA9550-24-1-0039.
