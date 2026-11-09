"""Python translation of the supplied GaAs(001) lattice KMC script.

The outer-cycle event order intentionally matches the MATLAB source:
Ga arrival, surface events over dt_ads, As attempts, then diagnostics/export.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from models import KMCParameters, MetricsSnapshot, ParticleSnapshot


RATE_PARAMETER_NAMES = frozenset(
    {
        "attempt_frequency",
        "temperature_k",
        "boltzmann_ev_k",
        "ediff_ga",
        "ediff_al",
        "ediff_as",
        "e_lat_g3",
        "e_lat_g5",
        "e_es",
        "e_des_ga",
        "e_des_as",
    }
)
STRUCTURAL_PARAMETER_NAMES = frozenset(
    {"lattice_size", "max_height", "seed"}
)


class GaAsKMC:
    """Stateful solid-on-solid kinetic Monte Carlo model."""

    def __init__(self, parameters: KMCParameters | None = None) -> None:
        self.parameters = parameters or KMCParameters()
        self._validate_parameters(self.parameters)
        if self.parameters.seed is None:
            self.parameters.seed = int(np.random.SeedSequence().entropy)
        self._rng = np.random.default_rng(self.parameters.seed)
        self._initialize_state()

    def _initialize_state(self) -> None:
        p = self.parameters
        shape = (p.lattice_size, p.lattice_size)
        self.height = np.zeros(shape, dtype=np.int32)
        self.type_map = np.zeros((*shape, p.max_height), dtype=np.int8)
        self.atom_id_map = np.zeros((*shape, p.max_height), dtype=np.int64)
        self.next_atom_id = 1

        self.rates = np.zeros((*shape, 5), dtype=np.float64)
        self.node_rates = np.zeros(shape, dtype=np.float64)
        self.total_rate = 0.0

        self.current_time_sec = 0.0
        self.outer_cycles = 0
        self.finished = False
        self._metric_times: list[float] = []
        self._rheed: list[float] = []
        self._roughness: list[float] = []
        self._coverage: list[NDArray[np.float64]] = []
        self.rebuild_all_rates()

    @staticmethod
    def _validate_parameters(parameters: KMCParameters) -> None:
        if parameters.lattice_size < 3:
            raise ValueError("lattice_size должен быть не меньше 3")
        if parameters.max_height < 1:
            raise ValueError("max_height должен быть положительным")
        if parameters.max_time_sec <= 0:
            raise ValueError("max_time_sec должен быть положительным")
        if parameters.group_iii_flux <= 0:
            raise ValueError("group_iii_flux должен быть положительным")
        if parameters.as_excess < 0:
            raise ValueError("as_excess не может быть отрицательным")
        if not 0.0 <= parameters.al_fraction <= 1.0:
            raise ValueError("al_fraction должен находиться в диапазоне [0, 1]")
        if parameters.attempt_frequency <= 0 or parameters.temperature_k <= 0:
            raise ValueError("attempt_frequency и temperature_k должны быть > 0")
        if parameters.boltzmann_ev_k <= 0:
            raise ValueError("boltzmann_ev_k должен быть положительным")
        energy_parameters = (
            parameters.ediff_ga,
            parameters.ediff_al,
            parameters.ediff_as,
            parameters.e_lat_g3,
            parameters.e_lat_g5,
            parameters.e_es,
            parameters.e_des_ga,
            parameters.e_des_as,
        )
        if any(value < 0 for value in energy_parameters):
            raise ValueError("Энергетические барьеры не могут быть отрицательными")
        if parameters.max_jumps_per_interval < 1:
            raise ValueError("max_jumps_per_interval должен быть >= 1")
        if parameters.rheed_every < 1 or parameters.ovito_every < 1:
            raise ValueError("частоты записи должны быть >= 1")
        if parameters.substrate_layers < 0:
            raise ValueError("substrate_layers не может быть отрицательным")

    def reset(self, parameters: KMCParameters | None = None) -> None:
        """Start a new run and reset the random generator from the seed."""
        if parameters is not None:
            self.parameters = parameters
        self._validate_parameters(self.parameters)
        self._rng = np.random.default_rng(self.parameters.seed)
        self._initialize_state()

    def apply_parameters(self, **changes: float | int) -> bool:
        """Apply changes between batches; return whether rates were rebuilt."""
        allowed = {item.name for item in fields(KMCParameters)}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Неизвестные параметры: {', '.join(sorted(unknown))}")
        structural = STRUCTURAL_PARAMETER_NAMES.intersection(changes)
        if structural:
            raise ValueError(
                "Нельзя менять размер решётки, высоту массива или seed в ходе роста; "
                "выполните reset с новыми параметрами"
            )
        if "max_time_sec" in changes and float(changes["max_time_sec"]) < self.current_time_sec:
            raise ValueError("max_time_sec не может быть меньше уже достигнутого времени")

        candidate = KMCParameters(**{
            item.name: getattr(self.parameters, item.name) for item in fields(KMCParameters)
        })
        for name, value in changes.items():
            setattr(candidate, name, value)
        self._validate_parameters(candidate)
        self.parameters = candidate

        rates_changed = bool(RATE_PARAMETER_NAMES.intersection(changes))
        if rates_changed:
            self.rebuild_all_rates()
        return rates_changed

    def rebuild_all_rates(self) -> None:
        """Recalculate every node after a global kinetic parameter change."""
        self.rates.fill(0.0)
        self.node_rates.fill(0.0)
        p = self.parameters
        for r in range(p.lattice_size):
            for c in range(p.lattice_size):
                self._calculate_single_node(r, c)
        self.total_rate = float(np.sum(self.node_rates, dtype=np.float64))

    def advance_batch(self, max_outer_cycles: int = 1) -> bool:
        """Run whole outer cycles without changing their original operation order."""
        if max_outer_cycles < 1:
            raise ValueError("max_outer_cycles должен быть >= 1")
        for _ in range(max_outer_cycles):
            if self.finished:
                break
            self._advance_outer_cycle()
        return self.finished

    def _advance_outer_cycle(self) -> None:
        p = self.parameters
        if self.current_time_sec >= p.max_time_sec:
            self.finished = True
            return
        dt_ads = -np.log(self._rng.random()) / (p.group_iii_flux * p.lattice_size**2)
        if self.current_time_sec + dt_ads > p.max_time_sec:
            self.current_time_sec = p.max_time_sec
            self.finished = True
            return

        self.current_time_sec += float(dt_ads)
        self.outer_cycles += 1

        # 1. Ga adsorption (same order as the source script).
        rx = int(self._rng.integers(p.lattice_size))
        ry = int(self._rng.integers(p.lattice_size))
        if self.height[rx, ry] % 2 == 0:
            new_h = int(self.height[rx, ry]) + 1
            self._require_capacity(new_h)
            self.height[rx, ry] = new_h
            self.type_map[rx, ry, new_h - 1] = 2 if self._rng.random() < p.al_fraction else 1
            self.atom_id_map[rx, ry, new_h - 1] = self.next_atom_id
            self.next_atom_id += 1
            self._update_patch(rx, ry)

        # 2. Surface diffusion/desorption during this interval.
        t_int = 0.0
        jump_count = 0
        while (
            t_int < dt_ads
            and jump_count < p.max_jumps_per_interval
            and self.total_rate > 0.0
        ):
            dt_jump = -np.log(self._rng.random()) / self.total_rate
            if t_int + dt_jump > dt_ads:
                break
            t_int += float(dt_jump)
            jump_count += 1
            r, c, event = self._choose_event()
            old_h = int(self.height[r, c])

            if event < 4:
                self._diffuse(r, c, event, old_h)
            elif old_h % 2 == 1:
                self._desorb_ga(r, c, old_h)
            else:
                self._desorb_as_pair(r, c, old_h)

        # 3. As adsorption attempts, after surface events as in the source.
        for _ in range(p.as_excess):
            ax = int(self._rng.integers(p.lattice_size))
            ay = int(self._rng.integers(p.lattice_size))
            if self.height[ax, ay] % 2 == 1:
                new_h = int(self.height[ax, ay]) + 1
                self._require_capacity(new_h)
                self.height[ax, ay] = new_h
                self.type_map[ax, ay, new_h - 1] = 0
                self.atom_id_map[ax, ay, new_h - 1] = self.next_atom_id
                self.next_atom_id += 1
                self._update_patch(ax, ay)

        # 4. Measurements and trajectory export, after both adsorption stages.
        if self.outer_cycles % p.rheed_every == 0:
            self._sample_metrics()

    def _require_capacity(self, new_height: int) -> None:
        if new_height > self.parameters.max_height:
            raise OverflowError(
                f"Рост вышел за MaxH={self.parameters.max_height}; "
                "увеличьте max_height перед запуском"
            )

    def _choose_event(self) -> tuple[int, int, int]:
        flat_node = int(
            np.searchsorted(np.cumsum(self.node_rates.ravel()), self._rng.random() * self.total_rate)
        )
        flat_node = min(flat_node, self.node_rates.size - 1)
        r, c = np.unravel_index(flat_node, self.node_rates.shape)
        node_rate = self.node_rates[r, c]
        event = int(np.searchsorted(np.cumsum(self.rates[r, c]), self._rng.random() * node_rate))
        return int(r), int(c), min(event, 4)

    def _diffuse(self, r: int, c: int, event: int, old_h: int) -> None:
        p = self.parameters
        dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[event]
        nr = (r + dr) % p.lattice_size
        nc = (c + dc) % p.lattice_size
        new_h = int(self.height[nr, nc]) + 1
        self._require_capacity(new_h)

        self.type_map[nr, nc, new_h - 1] = self.type_map[r, c, old_h - 1]
        self.atom_id_map[nr, nc, new_h - 1] = self.atom_id_map[r, c, old_h - 1]
        self.type_map[r, c, old_h - 1] = 0
        self.atom_id_map[r, c, old_h - 1] = 0
        self.height[r, c] = old_h - 1
        self.height[nr, nc] = new_h
        self._update_patches((r, c), (nr, nc))

    def _desorb_ga(self, r: int, c: int, old_h: int) -> None:
        self.type_map[r, c, old_h - 1] = 0
        self.atom_id_map[r, c, old_h - 1] = 0
        self.height[r, c] = old_h - 1
        self._update_patch(r, c)

    def _desorb_as_pair(self, r: int, c: int, old_h: int) -> None:
        p = self.parameters
        neighbors = [
            ((r + 1) % p.lattice_size, c),
            ((r - 1) % p.lattice_size, c),
            (r, (c + 1) % p.lattice_size),
            (r, (c - 1) % p.lattice_size),
        ]
        valid = [(nr, nc) for nr, nc in neighbors if self.height[nr, nc] == old_h]
        if not valid:
            return
        nr, nc = valid[int(self._rng.integers(len(valid)))]
        self.type_map[r, c, old_h - 1] = 0
        self.atom_id_map[r, c, old_h - 1] = 0
        self.height[r, c] = old_h - 1
        self.type_map[nr, nc, old_h - 1] = 0
        self.atom_id_map[nr, nc, old_h - 1] = 0
        self.height[nr, nc] = old_h - 1
        self._update_patches((r, c), (nr, nc))

    def _update_patch(self, x: int, y: int) -> None:
        self._update_patches((x, y))

    def _update_patches(self, *centers: tuple[int, int]) -> None:
        p = self.parameters
        offsets = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2))
        patch: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for x, y in centers:
            for dx, dy in offsets:
                r = (x + dx) % p.lattice_size
                c = (y + dy) % p.lattice_size
                key = (r, c)
                if key not in seen:
                    seen.add(key)
                    patch.append(key)
        for r, c in patch:
            self.total_rate -= self.node_rates[r, c]
        for r, c in patch:
            self._calculate_single_node(r, c)
        for r, c in patch:
            self.total_rate += self.node_rates[r, c]
        if self.total_rate < 1.0e-18:
            self.total_rate = 0.0

    def _calculate_single_node(self, r: int, c: int) -> None:
        p = self.parameters
        self.rates[r, c, :] = 0.0
        h_rc = int(self.height[r, c])
        if h_rc == 0:
            self.node_rates[r, c] = 0.0
            return

        current_type = h_rc % 2
        my_id = int(self.type_map[r, c, h_rc - 1])
        if current_type == 1:
            axis_strong = "x"
            ediff_base = p.ediff_al if my_id == 2 else p.ediff_ga
        else:
            axis_strong = "y"
            ediff_base = p.ediff_as

        nb_x = int(self.height[r, (c + 1) % p.lattice_size] >= h_rc) + int(
            self.height[r, (c - 1) % p.lattice_size] >= h_rc
        )
        nb_y = int(self.height[(r + 1) % p.lattice_size, c] >= h_rc) + int(
            self.height[(r - 1) % p.lattice_size, c] >= h_rc
        )
        e_lat = p.e_lat_g3 if current_type == 1 else p.e_lat_g5
        if axis_strong == "x":
            e_binding = nb_x * e_lat * 1.2 + nb_y * e_lat
        else:
            e_binding = nb_x * e_lat + nb_y * e_lat * 1.2

        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for event, (dr, dc) in enumerate(directions):
            nr = (r + dr) % p.lattice_size
            nc = (c + dc) % p.lattice_size
            destination_h = int(self.height[nr, nc])
            if destination_h < h_rc and current_type != destination_h % 2:
                cur_e = ediff_base + e_binding
                if destination_h < h_rc - 1:
                    cur_e += p.e_es
                self.rates[r, c, event] = self._arrhenius(cur_e)

        if current_type == 1:
            self.rates[r, c, 4] = self._arrhenius(p.e_des_ga + e_binding)
        else:
            n_as_neighbors = sum(
                int(self.height[nr, nc] == h_rc)
                for nr, nc in (
                    ((r + 1) % p.lattice_size, c),
                    ((r - 1) % p.lattice_size, c),
                    (r, (c + 1) % p.lattice_size),
                    (r, (c - 1) % p.lattice_size),
                )
            )
            if n_as_neighbors > 0:
                self.rates[r, c, 4] = (
                    0.5 * n_as_neighbors * self._arrhenius(p.e_des_as + e_binding)
                )

        self.node_rates[r, c] = float(np.sum(self.rates[r, c]))

    def _arrhenius(self, barrier_ev: float) -> float:
        p = self.parameters
        return p.attempt_frequency * np.exp(-barrier_ev / (p.boltzmann_ev_k * p.temperature_k))

    def _sample_metrics(self) -> None:
        p = self.parameters
        phase_sum = np.sum(np.exp(1j * np.pi * self.height.ravel() / 2.0))
        self._metric_times.append(self.current_time_sec)
        self._rheed.append(float(np.abs(phase_sum / p.lattice_size**2) ** 2))
        self._roughness.append(float(np.std(self.height, ddof=0)))
        coverage = np.fromiter(
            (np.count_nonzero(self.height >= h) / p.lattice_size**2 for h in range(1, p.max_height + 1)),
            dtype=np.float64,
            count=p.max_height,
        )
        self._coverage.append(coverage)

    def get_metrics(self) -> MetricsSnapshot:
        p = self.parameters
        coverage = (
            np.stack(self._coverage, axis=1)
            if self._coverage
            else np.empty((p.max_height, 0), dtype=np.float64)
        )
        return MetricsSnapshot(
            time_sec=np.asarray(self._metric_times, dtype=np.float64),
            rheed_intensity=np.asarray(self._rheed, dtype=np.float64),
            roughness=np.asarray(self._roughness, dtype=np.float64),
            layer_coverage=coverage,
            current_time_sec=float(self.current_time_sec),
            outer_cycles=int(self.outer_cycles),
        )

    def get_particle_snapshot(self, include_substrate: bool = True) -> ParticleSnapshot:
        p = self.parameters
        d = p.lattice_constant_angstrom / np.sqrt(2.0)
        dz = p.lattice_constant_angstrom / 4.0
        box_x = p.lattice_size * d
        atom_coords: list[tuple[float, float, float]] = []
        atom_ids: list[int] = []
        atom_types: list[int] = []

        if include_substrate:
            for s in range(1, p.substrate_layers + 1):
                h = 1 - s
                is_as = h % 2 == 0
                z = (h - 1) * dz
                xshift, yshift = zinc_blende_shift(h, d)
                particle_type = 2 if is_as else 1
                for ix in range(p.lattice_size):
                    x = ((ix * d + xshift) % box_x)
                    for iy in range(p.lattice_size):
                        y = (iy * d + yshift) % box_x
                        atom_coords.append((x, y, z))
                        atom_ids.append(-((s - 1) * p.lattice_size**2 + ix * p.lattice_size + iy + 1))
                        atom_types.append(particle_type)

        for ix in range(p.lattice_size):
            for iy in range(p.lattice_size):
                for h in range(1, int(self.height[ix, iy]) + 1):
                    atom_id = int(self.atom_id_map[ix, iy, h - 1])
                    if atom_id <= 0:
                        raise RuntimeError(f"Пропущен или некорректен AtomID в ({ix}, {iy}, {h})")
                    xshift, yshift = zinc_blende_shift(h, d)
                    if h % 2 == 0:
                        particle_type = 2
                    else:
                        particle_type = 3 if self.type_map[ix, iy, h - 1] == 2 else 1
                    x = ((ix * d + xshift) % box_x)
                    y = (iy * d + yshift) % box_x
                    atom_coords.append((x, y, (h - 1) * dz))
                    atom_ids.append(atom_id)
                    atom_types.append(particle_type)

        return ParticleSnapshot(
            coordinates=np.asarray(atom_coords, dtype=np.float64).reshape((-1, 3)),
            particle_ids=np.asarray(atom_ids, dtype=np.int64),
            particle_types=np.asarray(atom_types, dtype=np.int32),
            box_size_angstrom=float(box_x),
            time_sec=float(self.current_time_sec),
            outer_cycle=int(self.outer_cycles),
        )

    def export_dump(self, filename: str | Path, timestep: int | None = None) -> None:
        """Append the current state as one LAMMPS-style OVITO trajectory frame."""
        p = self.parameters
        snapshot = self.get_particle_snapshot(include_substrate=True)
        d = p.lattice_constant_angstrom / np.sqrt(2.0)
        dz = p.lattice_constant_angstrom / 4.0
        zlo = -p.substrate_layers * dz - dz
        zhi = (max(1, int(np.max(self.height))) + 2) * dz
        frame = self.outer_cycles if timestep is None else int(timestep)
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="ascii", newline="\n") as stream:
            stream.write(f"ITEM: TIMESTEP\n{frame}\n")
            stream.write(f"ITEM: NUMBER OF ATOMS\n{snapshot.particle_ids.size}\n")
            stream.write("ITEM: BOX BOUNDS pp pp ff\n")
            stream.write(f"0 {p.lattice_size * d:.12g}\n0 {p.lattice_size * d:.12g}\n{zlo:.12g} {zhi:.12g}\n")
            stream.write("ITEM: ATOMS id type x y z\n")
            for atom_id, atom_type, (x, y, z) in zip(
                snapshot.particle_ids,
                snapshot.particle_types,
                snapshot.coordinates,
                strict=True,
            ):
                stream.write(f"{atom_id} {atom_type} {x:.12g} {y:.12g} {z:.12g}\n")

    def check_invariants(self) -> None:
        """Raise when particle IDs, heights, or rate totals are inconsistent."""
        if np.any(self.height < 0) or np.any(self.height > self.parameters.max_height):
            raise AssertionError("height вышел за допустимый диапазон")
        occupied = int(np.sum(self.height, dtype=np.int64))
        ids = self.atom_id_map[self.atom_id_map > 0]
        if ids.size != occupied or np.unique(ids).size != ids.size:
            raise AssertionError("AtomIDMap потерял или продублировал ID")
        expected_total = float(np.sum(self.node_rates, dtype=np.float64))
        if not np.isclose(self.total_rate, expected_total, rtol=1e-10, atol=1e-12):
            raise AssertionError("R_total_system не совпадает с суммой узловых скоростей")


def zinc_blende_shift(layer: int, lattice_step: float) -> tuple[float, float]:
    """Return the four-plane zinc-blende (001) basis shift in the rotated XY plane."""
    half_step = lattice_step / 2.0
    # X=[110], Y=[1-10]; layers 1..4 shift by (0,0), (d/2,0),
    # (d/2,d/2), (0,d/2), then repeat every four atomic planes.
    shifts = (
        (0.0, half_step),
        (0.0, 0.0),
        (half_step, 0.0),
        (half_step, half_step),
    )
    return shifts[layer % 4]

