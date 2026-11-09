import unittest

import numpy as np
from ovito.modifiers import CreateBondsModifier
from ovito.pipeline import Pipeline, StaticSource

from kmc_model import GaAsKMC
from models import KMCParameters
from ovito_view import build_ovito_data


class OvitoDataTests(unittest.TestCase):
    def test_memory_snapshot_builds_particle_pipeline_and_preserves_ids(self) -> None:
        model = GaAsKMC(
            KMCParameters(lattice_size=4, max_height=6, as_excess=0, seed=8)
        )
        model.height[0, 0] = 2
        model.type_map[0, 0, 0] = 2
        model.atom_id_map[0, 0, :2] = (41, 42)
        model.next_atom_id = 43
        snapshot = model.get_particle_snapshot()

        data = build_ovito_data(snapshot)
        pipeline = Pipeline(source=StaticSource(data=data))
        result = pipeline.compute()

        self.assertEqual(result.particles.count, snapshot.particle_ids.size)
        np.testing.assert_array_equal(
            result.particles["Particle Identifier"], snapshot.particle_ids
        )
        np.testing.assert_array_equal(result.particles["Particle Type"], snapshot.particle_types)

    def test_substrate_can_be_omitted_from_snapshot_view(self) -> None:
        model = GaAsKMC(
            KMCParameters(lattice_size=3, max_height=6, substrate_layers=2, as_excess=0)
        )
        model.height[1, 1] = 1
        model.atom_id_map[1, 1, 0] = 1
        model.next_atom_id = 2
        snapshot = model.get_particle_snapshot()

        data = build_ovito_data(snapshot, include_substrate=False)
        result = Pipeline(source=StaticSource(data=data)).compute()

        self.assertEqual(result.particles.count, 1)
        np.testing.assert_array_equal(result.particles["Particle Identifier"], [1])

    def test_pipeline_source_can_be_replaced_without_rebuilding_pipeline(self) -> None:
        model = GaAsKMC(
            KMCParameters(lattice_size=3, max_height=6, substrate_layers=0, as_excess=0)
        )
        model.height[1, 1] = 1
        model.atom_id_map[1, 1, 0] = 51
        model.next_atom_id = 52
        pipeline = Pipeline(
            source=StaticSource(data=build_ovito_data(model.get_particle_snapshot(False)))
        )
        first = pipeline.compute()
        self.assertEqual(first.particles.count, 1)

        model.atom_id_map[1, 1, 0] = 51
        model.height[2, 2] = 1
        model.atom_id_map[2, 2, 0] = 52
        model.next_atom_id = 53
        updated = build_ovito_data(model.get_particle_snapshot(False))
        pipeline.source = StaticSource(data=updated)
        second = pipeline.compute()

        self.assertEqual(second.particles.count, 2)
        np.testing.assert_array_equal(second.particles["Particle Identifier"], [51, 52])

    def test_zinc_blende_bond_cutoff_builds_nearest_neighbor_network(self) -> None:
        model = GaAsKMC(
            KMCParameters(lattice_size=4, max_height=8, substrate_layers=4, as_excess=0)
        )
        model.height[:, :] = 4
        atom_id = 1
        for ix in range(4):
            for iy in range(4):
                model.atom_id_map[ix, iy, :4] = np.arange(atom_id, atom_id + 4)
                atom_id += 4
        model.next_atom_id = atom_id
        pipeline = Pipeline(source=StaticSource(data=build_ovito_data(model.get_particle_snapshot())))
        pipeline.modifiers.append(CreateBondsModifier(cutoff=2.50))

        result = pipeline.compute()

        self.assertGreater(result.particles.bonds.count, 0)
        self.assertEqual(result.particles.count, 128)


if __name__ == "__main__":
    unittest.main()

