# Virtual Reactor — GaAs(001)

A Python desktop application for kinetic Monte Carlo simulation of GaAs(001) growth using a solid-on-solid model. The interface is built with Qt, while OVITO provides the embedded atomistic view.

## Features

- Seeded KMC simulations with Ga adsorption, surface diffusion, desorption, and As adsorption attempts.
- Live controls for simulation parameters, including queued updates during a run.
- Live atomistic visualization, RHEED signal, RMS roughness, and layer coverage plots.
- Per-run trajectory, parameter-change journal, metrics CSV, and PNG plots written under `runs/`.

## Requirements

- Python 3.12 (the application is developed and packaged against Python 3.12).
- A platform supported by the OVITO Python package. The embedded viewport may require a working desktop graphics environment.

Install the dependencies from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe Script\main.py
```

On Windows, you can also launch `Script\Launch Virtual Reactor.bat` after installing dependencies in the root `.venv`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Project layout

```text
Script/     Application and simulation implementation
tests/      Unit tests
runs/       Generated simulation output (ignored by Git)
```

Key modules in `Script/`:

- `kmc_model.py` — KMC state, event selection, rates, metrics, snapshots, and trajectory export.
- `kmc_worker.py` — Qt worker and controller for run, pause, step, stop, reset, and queued parameter changes.
- `ovito_view.py` — zinc-blende plane coordinates, bonds, atom legend, and embedded OVITO viewport.
- `plots.py` — RHEED, RMS roughness, and layer-coverage curves.
- `controls.py`, `main_window.py` — simulation controls and desktop interface.
- `models.py` — parameter, metric, snapshot, and batch data structures.

## Model notes

Each outer cycle follows the supplied MATLAB operation order: sample a Ga arrival interval and advance time, attempt Ga adsorption, process diffusion and desorption events for that interval, perform the configured As adsorption attempts, then update diagnostics and trajectory output at their configured intervals.

The implementation uses NumPy's seeded random generator. A run is reproducible from its recorded seed, but it will not reproduce MATLAB's exact event-by-event random sequence. Changes to kinetic-rate parameters rebuild the rate arrays at a worker batch boundary; other supported parameter updates take effect at that boundary without rebuilding rates.

