# Algorithmic workflow

This document summarizes the computational workflow implemented in this repository: computing locally optimal networks, searching for the global optimum in a multistable landscape, classifying optimal networks by their symmetry, and reproducing the figures of the manuscript. Equation numbers refer to the manuscript.

## 1. Input and data structures

A network is a connected graph $G = (\mathcal{V}, \mathcal{E})$, stored as a `networkx` graph together with its oriented node–edge incidence matrix $\boldsymbol{E}$ (`nx.incidence_matrix(G, oriented=True)`). Edge capacities are a vector $\boldsymbol{k} \in \mathbb{R}^{|\mathcal{E}|}_{\ge 0}$ whose ordering follows the columns of $\boldsymbol{E}$.

The main computational inputs are:
- the incidence matrix $\boldsymbol{E}$,
- the second-moment matrix of the injections, $\boldsymbol{\Sigma} = \langle \boldsymbol{S}\boldsymbol{S}^\top \rangle$, Eq. (6),
- the scaling exponent $\gamma$ and the budget $\kappa$ of the resource constraint, Eq. (7),
- the fluctuation strength $\beta$ and the mean load $\mu$, Eqs. (20)–(21), or, for the energy example, the share of fluctuating generation $\zeta$, Eq. (30).

Throughout, $\kappa = \mu = 1$.

## 2. Second moments of the injections

For the model networks, the injections enter only through $\boldsymbol{\Sigma}$, given in closed form by Eq. (32).

```text
INPUT: sources V_g, sinks V_c, mu, beta

for i, j in V:
    if i = j in V_g:         Sigma_ij = (N_c/N_g)^2 mu^2 + (N_g - 1) beta^2
    if i != j, both in V_g:  Sigma_ij = (N_c/N_g)^2 mu^2 - beta^2
    if i, j in V_c:          Sigma_ij = mu^2
    if i in V_g, j in V_c:   Sigma_ij = -(N_c/N_g) mu^2   (and symmetric)

OUTPUT: Sigma
```

For the energy example (Section 9), $\boldsymbol{\Sigma}$ is estimated empirically from the hourly time series.

### File mapping
- `rings_calc.compute_ring_second_moment_matrix`
- `cube_calc.compute_second_moment_matrix`
- `bipartite_calc.bipartite_second_moment_matrix`
- `multilayer_calc.second_moment_matrix`

## 3. Locally optimal networks: the self-consistent iteration

Stationarity of the Lagrangian, Eq. (49), gives the fixed-point condition Eq. (31), which is iterated following Corson (Phys. Rev. Lett. 104, 048703, 2010). Every fixed point is a stationary point of $\bar D$ on the constraint surface; in practice the iteration converges to local minima.

### Pseudocode: self-consistent iteration

```text
INPUT: E, Sigma, gamma, kappa, initial capacities k0 >= 0, tolerance tol, max_iter

k = kappa * k0 / (sum_e k0_e^gamma)^(1/gamma)          # project onto the budget

repeat:
    L    = E diag(k) E^T                               # weighted Laplacian
    PTDF = diag(k) E^T L^+                             # flows F = PTDF S, Eq. (4)
    F2_e = (PTDF Sigma PTDF^T)_ee   for all e          # <F_e^2>

    k_old = k
    k_e   = kappa * F2_e^(1/(1+gamma)) / (sum_l F2_l^(gamma/(1+gamma)))^(1/gamma)   # Eq. (31)
until convergence or max_iter iterations

D = sum over active edges of F2_e / k_e                # Eq. (6)

OUTPUT: k, D
```

An edge with $k_e = 0$ carries no flow and therefore stays inactive in all later iterations, so the support of the initial condition bounds the support of the result. A uniform (or otherwise $\Gamma$-invariant) initial condition stays invariant under the update.

Two variants of the convergence test are used:
- `corson_algorithm.selfconsistent_minimum` (rings, cube): absolute test $\lVert \boldsymbol{k} - \boldsymbol{k}_{\mathrm{old}} \rVert^2 < 10^{-10}$, at most 1000 iterations.
- `bipartite_calc.selfconsistent_minimum_rel` (complete bipartite and multilayer networks): relative test $\lVert \boldsymbol{k} - \boldsymbol{k}_{\mathrm{old}} \rVert^2 \le 10^{-12} \lVert \boldsymbol{k} \rVert^2$, at most 400 iterations in the coarse scan and 20000 in the fine scan. The relative test is needed because the resource constraint forces $k_e \sim |\mathcal{E}|^{-1/\gamma}$, which becomes very small at small $\gamma$.

## 4. Approximating the global optimum: multistart search

For $0 < \gamma < 1$ the problem is non-convex and multistable, so the outcome of the iteration depends on its initial condition. The global optimum is approximated by running the iteration from a set of initial conditions (a *seed bank*) and keeping the result with the lowest dissipation.

### Pseudocode: best minimum over a seed bank

```text
INPUT: E, Sigma, gamma, seed bank B

D_best = infinity
for each seed s in B:
    (k, D) = self-consistent iteration from s
    if the iteration failed, or k or D is not finite:
        skip
    if D < D_best:
        (k_best, D_best) = (k, D)

OUTPUT: k_best, D_best
```

The seed banks are network specific and combine structured and random initial conditions:

| Network | Seed bank |
|---------|-----------|
| Cube | uniform and random dense capacities; random dense capacities with one edge removed, for every edge; spanning-tree initial conditions; and seven configurations identified in preliminary runs |
| $K_{N_g,N_c}$ | uniform; a double star; matching-enhanced and hub-enhanced configurations (dense and sparse); source-enhanced configurations; 8 random dense; 6 random spanning trees |
| Multilayer | uniform; layer A strengthened by a factor of 2; 4 random dense; 4 random spanning trees |

All random initial conditions are generated from fixed seeds, so every scan is deterministic.

### File mapping
- `corson_algorithm.random_minimum` (cube)
- `bipartite_calc.seed_bank`, `bipartite_calc.best_minimum`
- `multilayer_calc.seed_bank`, `multilayer_calc.best_minimum`

## 5. Phase diagrams: two-stage scan

The phase diagrams of the cube, the complete bipartite network and the multilayer network are computed in two stages. A coarse scan explores the landscape broadly and collects candidate optimal configurations. These are reduced to a small set of representatives, from which a fine scan starts at every parameter point.

### Pseudocode: coarse scan, reduction, fine scan

```text
INPUT: G, symmetry group Gamma, coarse grid (beta_i, gamma_j), fine grid (beta_m, gamma_n)

# Stage 1: coarse scan
for each beta_i:
    Sigma = second moments at beta_i
    c = none                                         # continuation seed
    for each gamma_j in decreasing order:            # start in the convex region
        B = seed bank  +  symmetric configuration
        if c exists:
            B = B + {c, c + 0.01 max(c)}             # continuation from the previous gamma
        k*_ij = best minimum over B
        c = k*_ij

# Reduction
R = {}
for each coarse optimum k*_ij:
    key = (support of k*_ij modulo Gamma, k*_ij symmetric or not)
    keep the first configuration for each key in R
add to R the symmetric configuration perturbed by 5% along the critical blocks

# Stage 2: fine scan
for each (beta_m, gamma_n):
    k*_mn = best minimum over R + {symmetric configuration}
    classify k*_mn (Section 6)

OUTPUT: optimal capacities, dissipation and phase label at every fine grid point
```

The fine scan uses no continuation, so neighbouring grid points cannot end up on different branches through path dependence. The perturbed symmetric seeds are needed near continuous transitions: the update preserves symmetry, so a weakly symmetry-broken optimum is reached only from an initial condition that already breaks the symmetry along the critical block.

Network-specific details:
- **Cube.** The coarse scan runs the multistart search of Section 4 independently at every grid point, without continuation. The reduction keeps every configuration that differs from all previously kept ones by more than an absolute tolerance of $10^{-2}$ in some component.
- **Complete bipartite network $K_{4,4}$.** The support is brought to a canonical form under $\Gamma \cong S_{N_g} \times S_{N_c}$ by minimizing over all row and column permutations of the $N_g \times N_c$ adjacency pattern. The perturbed seeds strengthen one sink (sink block) or a matching (loop block).
- **Multilayer network.** The support is brought to a canonical form under $\Gamma \cong D_{2d}$. The perturbed seed strengthens layer A, along the layer-exchanging block. The scans run in parallel over rows of $\beta$, each row with its own random generator, `default_rng([seed, row])`, so the results do not depend on the number of processes. Results are checkpointed after every row, and an interrupted scan resumes from the checkpoint.

### File mapping
- `cube_calc.compute_coarse_phase_diagram`, `extract_unique_guesses`, `compute_fine_phase_diagram`
- `bipartite_calc.gamma_scan`, `compute_coarse_phase_diagram`, `canonical_support`, `extract_configuration_bank`, `compute_fine_phase_diagram`
- `multilayer_calc.compute_coarse_phase_diagram`, `canonical_support`, `extract_configuration_bank`, `compute_fine_phase_diagram`

## 6. Classification of optimal networks

Each optimum is classified as symmetric, weakly symmetry-broken or strongly symmetry-broken (Eqs. (9)–(11)) and assigned its residual symmetry group $\Gamma^*$.

### Pseudocode: classification

```text
INPUT: optimal capacities k*, symmetry group Gamma with edge permutations pi_g

active edges:   A = {e : k*_e > threshold}                # support, Eq. (11)
residual group: Gamma* = {g in Gamma : k*[pi_g] = k* within relative tolerance}   # Eq. (10)

if Gamma* = Gamma:
    phase = symmetric
else if A is invariant under Gamma:
    phase = weakly symmetry-broken
else:
    phase = strongly symmetry-broken

OUTPUT: phase, number of active edges |A|, Gamma*
```

The symmetry group $\Gamma$, Eq. (9), is computed as the set of graph automorphisms that map sources onto sources and sinks onto sinks, each represented by its node and edge permutations.

Network-specific tolerances:
- **Cube.** An edge is active if $k^*_e > 10^{-8}$. The phase is looked up from the number of active edges and the residual symmetry of the support.
- **Complete bipartite network.** An edge is active if $k^*_e > 10^{-8} \max_f k^*_f$, and an optimum is symmetric if $\min_e k^*_e / \max_e k^*_e > 1 - 10^{-6}$.
- **Multilayer network.** An edge is active if $k^*_e > 10^{-8} \max_f k^*_f$, and an optimum is symmetric if it is invariant under every element of $\Gamma$ within a relative tolerance of $10^{-6}$. This network is not edge-transitive, so its symmetric optimum is not uniform.

### File mapping
- `cube_calc.symmetry_group`, `residual_group`, `classify_network`, `classify_phase_diagram`
- `bipartite_calc.classify_phase_diagram`, `select_phases`
- `multilayer_calc.is_symmetric`, `residual_group`, `classify_phase_diagram`, `layer_measures`

## 7. Ring networks: comparison of candidate branches

For the ring with alternating sources and sinks, the competing configurations are known explicitly, Eq. (23), so the phase diagram follows from comparing their dissipations rather than from a multistart search.

### Pseudocode: ring phase diagram

```text
INPUT: N, mu, grid (beta_i, gamma_j)

for each (beta_i, gamma_j):
    Sigma = ring second moments
    candidates:
        symmetric:          k_e = (2N)^(-1/gamma) kappa,  D from Eq. (6)
        strongly broken:    open ring (path); <F_e^2> follows from KCL alone,
                            capacities from one application of Eq. (31)
        weakly broken:      alternating capacities (k_a, k_b), k_b fixed by the constraint;
                            minimize D over k_a by bounded scalar minimization
        (N = 2 only)        sink-paired capacities, minimized in the same way
    optimum = candidate with the lowest D
    store min_e k*_e and the branch label

OUTPUT: phase diagram (Fig. 3a-e)
```

On a tree the flows are fixed by Kirchhoff's current law and do not depend on the capacities, so a single application of Eq. (31) gives the exact optimum of the open ring.

Further ring workflows:
- **Landscape for $N = 3$, $\mu = 0$ (Fig. 2a–d).** The reduced dissipation $\bar D(k_1, k_3)$ is evaluated on a grid over the feasible domain, and the best interior and boundary minima are located.
- **Reactivation capacity $\varepsilon_*$ (Fig. 2e).** The change $\bar D(\varepsilon) - \bar D(0)$ is evaluated on a logarithmic grid in $\varepsilon$, its first sign change is located by linear interpolation, and the result is compared with the leading-order prediction (Supplementary Methods 3).
- **Scaling of $\gamma_c$ with $N$ (Fig. 3f,g).** For each $N$, the crossing $\bar D^*_{\mathrm{sym}}(\gamma) = \bar D^*_{\mathrm{sb}}(\gamma)$ is computed from closed-form expressions (Supplementary Methods 7) and minimized over $\beta$, first on a grid and then by bounded scalar minimization. A power law $\gamma_c = c\,N^{-\alpha}$ is fitted with `scipy.optimize.curve_fit`.

### File mapping
- `rings_calc.get_capacities_all_branches`
- `rings_calc.compute_asymmetric_case_from_sigma`, `find_best_alternating`, `find_best_block_N2`
- `rings_calc.meaningful_minima_N3`
- `rings_calc.get_Figure10_data`, `eps_star_asym`
- `rings_calc.gamma_crossing_closed_form`, `gamma_c_vs_N_closed_form`

## 8. Local stability thresholds

The local stability thresholds of the symmetric configuration, Eqs. (24), (27), (28) and (29), are evaluated from their closed forms and drawn as lines in the phase diagrams. They are derived in Supplementary Methods 6, 8 and 10 and are not computed numerically from the Hessian.

### File mapping
- `cube_calc.gamma_s_T1`, `gamma_s_T2_minus`, `gamma_s_alt_ring`
- `bipartite_calc.gamma_s_loop`, `gamma_s_sink`

## 9. Energy grid example

### Pseudocode: renewable-driven ring

```text
INPUT: hourly wind and PV capacity factors (energy_data/), grid (zeta_i, gamma_j)

normalize each time series to unit mean: Z_1 = wind, Z_3 = solar
build injections: fluctuating part X1 (node 1 wind, node 3 solar, node 5 = 3 - Z_1 - Z_3,
                  sinks -1) and mean part X2 (alternating +1, -1)
empirical second moments: M11 = <X1 X1^T>, M22 = <X2 X2^T>, M12 = <X1 X2^T> + <X2 X1^T>

for each zeta_i:
    Sigma = zeta^2 M11 + (1 - zeta)^2 M22 + zeta (1 - zeta) M12
    for each gamma_j:
        candidates: closed ring with equal capacities, and
                    the open ring obtained by removing each of the 2N edges in turn
        optimum = candidate with the lowest D
        store min_e k*_e

OUTPUT: phase diagram (Fig. 6d)
```

### File mapping
- `energy_grids.ipynb`: `load_capacity_factors`, `compute_renewable_second_moments`, `compute_capacity_scan`

## 10. Notebook workflows

### 10.1 `intro_figure.ipynb` (Fig. 1)

```text
1. Build the six-node ring and K_{3,3}
2. Draw both networks with sources and sinks marked
```

### 10.2 `rings.ipynb` (Figs. 2, 3; Supplementary Fig. 1)

```text
1. N = 2, mu = 0: analytic optimum as a function of gamma
2. N = 3, mu = 0: dissipation landscape, local and global minima, branch capacities
3. Reactivation capacity epsilon_* versus gamma
4. Phase diagrams for N = 2, 3, 4 by branch comparison
5. Cuts at fixed beta and fixed gamma for N = 2
6. gamma_c(N) for N = 2, ..., 9 and power-law fit
```

### 10.3 `cube.ipynb` (Fig. 4)

```text
1. Build the cube and its symmetry group T_d
2. Coarse scan (50 x 50) with the multistart search, or load from scan_data/
3. Reduce to distinct configurations
4. Fine scan (250 x 250), or load from scan_data/
5. Classify phases by support and residual symmetry
6. Cut at beta = 0.6 (500 points in gamma)
7. Overlay the stability thresholds and assemble the figure
```

### 10.4 `bipartite.ipynb` (Fig. 5; Supplementary Table 10)

```text
1. Build K_{4,4}
2. Coarse scan (50 x 50) with continuation in gamma, or load from scan_data/
3. Reduce to one representative per support class and add block seeds
4. Fine scan (200 x 200), or load from scan_data/
5. Classify phases, select representative networks, overlay thresholds
6. Export the data for Supplementary Table 10
```

### 10.5 `energy_grids.ipynb` (Fig. 6)

```text
1. Load and normalize the renewables.ninja time series
2. Compute the empirical second moments
3. Scan (400 x 400) in (zeta, gamma), or load from scan_data/
4. Plot the time series and the phase diagram
```

### 10.6 `multilayer.ipynb` (Supplementary Note 4)

```text
1. Build two 5 x 5 lattices with interlayer edges and four sources; symmetry group D_2d
2. Parallel coarse scan (50 x 50) with continuation, or load from scan_data/
3. Reduce to one representative per support class and add the block seed
4. Parallel fine scan (120 x 120), or load from scan_data/
5. Classify phases, compute layer measures and residual symmetries
6. Select representative networks and assemble the figure
```

## 11. Mapping between files and functionality

- `core/corson_algorithm.py`
  - self-consistent iteration (absolute convergence test)
  - multistart search for the cube
- `core/rings_calc.py`
  - ring geometry and second moments
  - candidate branches and their dissipation
  - dissipation landscape for $N = 3$
  - reactivation capacity
  - closed-form transition points and scaling with $N$
- `core/cube_calc.py`
  - cube geometry and second moments
  - coarse and fine scans
  - symmetry group, residual symmetry and phase classification
  - stability thresholds
- `core/bipartite_calc.py`
  - self-consistent iteration (relative convergence test)
  - seed bank, continuation in $\gamma$, coarse and fine scans
  - canonical supports and configuration bank
  - phase classification and stability thresholds
- `core/multilayer_calc.py`
  - multilayer geometry and second moments
  - parallel, checkpointed coarse and fine scans
  - canonical supports, residual symmetry and layer measures
- `core/*_plot.py`, `core/figstyle.py`
  - figure assembly and common style

## 12. Output of the code

Depending on the workflow, the code returns:
- optimal edge capacities $\boldsymbol{k}^*$ and the expected dissipation $\bar D$ at every parameter point,
- the number of active edges and the smallest (relative) edge capacity,
- phase labels and residual symmetry groups $\Gamma^*$,
- local stability thresholds and transition points,
- cached scans in `scan_data/` (`.npz`),
- and the figures of the manuscript as PDF files.
