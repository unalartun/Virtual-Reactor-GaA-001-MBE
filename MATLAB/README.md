# MATLAB KMC and predictor

These scripts are the MATLAB research lineage associated with the Python GaAs(001) Virtual Reactor in the repository root.

- `KMC_GaAsAlGaAs_MBE.m` runs the GaAs/AlGaAs MBE kinetic Monte Carlo model and plots RHEED intensity, roughness, an FFT spectrum, and layer coverage.
- `Final.m` generates KMC signal features, trains a neural network to predict the logarithms of Ga and As diffusion rates, and plots prediction comparisons.

`KMC_GaAsAlGaAs_MBE.m` received only basic run-blocking repairs while being added: the height-map initialization now uses the declared lattice size, an unfinished conditional was removed, and section labels that MATLAB would interpret as commands were commented out. The model equations and parameter values were left as supplied.

`Final.m` is kept as supplied. It is an experimental prototype: it currently generates only 10 samples, and its displayed R-squared values are computed on the same samples used for training, validation, and testing. Those values should not be treated as independent model validation.

The scripts have not been executed in MATLAB in this environment. `Final.m` uses `parfor` and `fitnet`, so the corresponding MATLAB toolboxes must be available.