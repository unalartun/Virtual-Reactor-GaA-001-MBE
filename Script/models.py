"""Data objects shared by the KMC engine and the user interface."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class KMCParameters:
    lattice_size: int = 40
    max_height: int = 40
    max_time_sec: float = 10.0
    group_iii_flux: float = 1.0
    as_excess: int = 100
    al_fraction: float = 0.0
    attempt_frequency: float = 1.0e13
    temperature_k: float = 850.0
    boltzmann_ev_k: float = 8.617e-5
    ediff_ga: float = 1.1
    ediff_al: float = 1.5
    ediff_as: float = 2.0
    e_lat_g3: float = 0.24
    e_lat_g5: float = 0.15
    e_es: float = 0.20
    e_des_ga: float = 2.8
    e_des_as: float = 2.4
    max_jumps_per_interval: int = 400
    rheed_every: int = 5
    ovito_every: int = 10
    lattice_constant_angstrom: float = 5.6533
    substrate_layers: int = 4
    seed: int | None = None


@dataclass(slots=True)
class MetricsSnapshot:
    time_sec: NDArray[np.float64]
    rheed_intensity: NDArray[np.float64]
    roughness: NDArray[np.float64]
    layer_coverage: NDArray[np.float64]
    current_time_sec: float
    outer_cycles: int


@dataclass(slots=True)
class ParticleSnapshot:
    coordinates: NDArray[np.float64]
    particle_ids: NDArray[np.int64]
    particle_types: NDArray[np.int32]
    box_size_angstrom: float
    time_sec: float
    outer_cycle: int


@dataclass(slots=True)
class BatchResult:
    finished: bool
    current_time_sec: float
    outer_cycles: int
    metrics: MetricsSnapshot
    particles: ParticleSnapshot | None = None
    error: str | None = None


@dataclass(slots=True)
class ParameterChange:
    values: dict[str, float | int] = field(default_factory=dict)
    applied_at_sec: float = 0.0
    outer_cycle: int = 0

