import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from kmc_model import GaAsKMC
from models import KMCParameters


def small_model() -> GaAsKMC:
    return GaAsKMC(
        KMCParameters(
            lattice_size=4,
            max_height=8,
            max_time_sec=1.0,
            group_iii_flux=0.1,
            as_excess=0,
            seed=17,
        )
    )


class GaAsKMCTests(unittest.TestCase):
    def test_zinc_blende_plane_shifts_repeat_every_four_planes(self) -> None:
        model = small_model()
        model.height[0, 0] = 4
        model.atom_id_map[0, 0, :4] = (1, 2, 3, 4)
        model.type_map[0, 0, 0] = 1
        model.type_map[0, 0, 2] = 1
        model.next_atom_id = 5
        snapshot = model.get_particle_snapshot(include_substrate=False)
        d = model.parameters.lattice_constant_angstrom / np.sqrt(2.0)
        dz = model.parameters.lattice_constant_angstrom / 4.0

        np.testing.assert_allclose(snapshot.coordinates[:4, 0], (0, d / 2, d / 2, 0))
        np.testing.assert_allclose(snapshot.coordinates[:4, 1], (0, 0, d / 2, d / 2))
        np.testing.assert_allclose(snapshot.coordinates[:4, 2], (0, dz, 2 * dz, 3 * dz))

    def test_empty_state_has_no_kinetic_events(self) -> None:
        model = small_model()
        self.assertEqual(model.total_rate, 0.0)
        model.check_invariants()

    def test_temperature_change_rebuilds_all_rates(self) -> None:
        model = small_model()
        model.height[1, 1] = 1
        model.type_map[1, 1, 0] = 1
        model.atom_id_map[1, 1, 0] = 1
        model.next_atom_id = 2
        model.rebuild_all_rates()
        rates_before = model.rates.copy()
        total_before = model.total_rate

        rebuilt = model.apply_parameters(temperature_k=1000.0)

        self.assertTrue(rebuilt)
        self.assertGreater(model.total_rate, total_before)
        self.assertTrue(np.all(model.rates[rates_before > 0] > rates_before[rates_before > 0]))
        model.check_invariants()

    def test_flux_change_does_not_rebuild_surface_rates(self) -> None:
        model = small_model()
        model.height[1, 1] = 1
        model.atom_id_map[1, 1, 0] = 1
        model.next_atom_id = 2
        model.rebuild_all_rates()
        old_rates = model.rates.copy()

        rebuilt = model.apply_parameters(group_iii_flux=0.5)

        self.assertFalse(rebuilt)
        np.testing.assert_array_equal(model.rates, old_rates)

    def test_seed_repeats_python_trajectory(self) -> None:
        parameters = KMCParameters(
            lattice_size=4,
            max_height=8,
            max_time_sec=0.02,
            group_iii_flux=10.0,
            as_excess=2,
            max_jumps_per_interval=5,
            seed=91,
        )
        first = GaAsKMC(parameters)
        second = GaAsKMC(parameters)
        first.advance_batch(2)
        second.advance_batch(2)

        np.testing.assert_array_equal(first.height, second.height)
        np.testing.assert_array_equal(first.atom_id_map, second.atom_id_map)
        self.assertEqual(first.current_time_sec, second.current_time_sec)

    def test_export_dump_atom_count_matches_snapshot(self) -> None:
        model = small_model()
        model.height[1, 1] = 1
        model.atom_id_map[1, 1, 0] = 1
        model.next_atom_id = 2
        snapshot = model.get_particle_snapshot()

        with TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.dump"
            model.export_dump(path, timestep=0)
            lines = path.read_text(encoding="ascii").splitlines()

        count_row = lines.index("ITEM: NUMBER OF ATOMS") + 1
        self.assertEqual(int(lines[count_row]), snapshot.particle_ids.size)
        self.assertEqual(lines.count("ITEM: ATOMS id type x y z"), 1)

    def test_local_rate_update_matches_full_rebuild(self) -> None:
        model = small_model()
        model.height[1, 1] = 1
        model.atom_id_map[1, 1, 0] = 1
        model.next_atom_id = 2
        model.rebuild_all_rates()
        model._update_patch(1, 1)
        local_rates = model.rates.copy()
        local_nodes = model.node_rates.copy()
        local_total = model.total_rate

        model.rebuild_all_rates()

        np.testing.assert_allclose(model.rates, local_rates, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(model.node_rates, local_nodes, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(model.total_rate, local_total, places=8)

    def test_snapshot_preserves_ids_and_ga_as_al_types(self) -> None:
        model = small_model()
        model.height[0, 0] = 2
        model.type_map[0, 0, 0] = 2
        model.atom_id_map[0, 0, 0] = 11
        model.atom_id_map[0, 0, 1] = 12
        model.next_atom_id = 13
        model.check_invariants()

        snapshot = model.get_particle_snapshot(include_substrate=False)

        np.testing.assert_array_equal(snapshot.particle_ids, [11, 12])
        np.testing.assert_array_equal(snapshot.particle_types, [3, 2])
        self.assertEqual(snapshot.coordinates.shape, (2, 3))

    def test_metric_history_shapes_match_layer_count(self) -> None:
        model = small_model()
        model.height[0, 0] = 1
        model._sample_metrics()

        metrics = model.get_metrics()

        self.assertEqual(metrics.layer_coverage.shape, (8, 1))
        self.assertEqual(metrics.rheed_intensity.size, 1)
        self.assertEqual(metrics.roughness.size, 1)


if __name__ == "__main__":
    unittest.main()

