This repository contains code and data used for obtaining numerical results for the publication
# Symmetry breaking in minimum dissipation networks
by Aarathi Parameswaran, Pezhman Ebrahimzadeh, Andrea Benigni, Dirk Witthaut, and Iva Bačić.

We consider optimal transport networks under stochastic load fluctuations, including ring networks, multilayer networks, and an application to renewable energy systems.

# Notebooks
rings.ipynb reproduces all results for ring networks (phase diagrams, transitions, scaling).

energy_grids.ipynb applies the framework to wind and solar energy data and computes the resulting phase diagram.

multilayer_1.ipynb computes and visualizes optimal structures in simple multilayer (cube-like) networks.

multilayer_2.ipynb computes and visualizes optimal structures in extended multilayer networks.

multilayer_3.ipynb computes and visualizes optimal structures in extended multilayer networks. 

The multilayer results are sensitive to initialization and branch selection in multistable regions, especially the extended ones.

The provided pipeline reproduces the results shown in the associated work.

# Usage
To use, run the notebooks to reproduce figures. All required functions are in core/.

# Data
The energy_data/ folder contains preprocessed time series for wind and solar generation used in the energy grid analysis, sourced from renewables.ninja.

# Requirements
numpy 1.26.4

scipy 1.15.3

matplotlib 3.9.1

pandas 2.2.2

networkx 3.3

# Citation
If you use this code or build upon it in your research, please cite the associated publication.
