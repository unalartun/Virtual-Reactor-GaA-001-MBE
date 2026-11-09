"""Embedded OVITO viewport fed directly from in-memory KMC snapshots."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from ovito.data import DataCollection
from ovito.gui import create_qwidget
from ovito.modifiers import CreateBondsModifier
from ovito.pipeline import Pipeline, StaticSource
from ovito.vis import Viewport

from models import ParticleSnapshot

ATOM_COLORS = {
    "Ga": "#f26b5b",
    "As": "#4aa3ff",
    "Al": "#a7d85c",
}


def build_ovito_data(snapshot: ParticleSnapshot, include_substrate: bool = True) -> DataCollection:
    """Create OVITO particle arrays from one packed in-memory snapshot."""
    data = DataCollection()
    keep = np.ones(snapshot.particle_ids.size, dtype=bool)
    if not include_substrate:
        keep = snapshot.particle_ids > 0
    coordinates = snapshot.coordinates[keep]
    particle_ids = snapshot.particle_ids[keep]
    particle_types = snapshot.particle_types[keep]

    if coordinates.size:
        min_z = float(np.min(coordinates[:, 2]))
        max_z = float(np.max(coordinates[:, 2]))
    else:
        min_z, max_z = -1.0, 1.0
    zlo = min_z - 2.0
    zspan = max(max_z - zlo + 2.0, 4.0)
    box_xy = max(snapshot.box_size_angstrom, 1.0)
    data.create_cell(np.diag([box_xy, box_xy, zspan]), pbc=(True, True, False))
    data.cell_[:, 3] = np.asarray((0.0, 0.0, zlo), dtype=np.float64)

    particles = data.create_particles(count=int(particle_ids.size))
    particles.create_property("Position", data=coordinates)
    particles.create_property("Particle Identifier", data=particle_ids)
    type_property = particles.create_property("Particle Type")
    types = (
        ("Ga", (0.95, 0.42, 0.36), 1.22),
        ("As", (0.29, 0.64, 1.00), 1.18),
        ("Al", (0.65, 0.85, 0.36), 1.16),
    )
    for name, color, radius in types:
        particle_type = type_property.add_type_name(name, particles)
        particle_type.color = color
        particle_type.radius = radius
    type_property[:] = particle_types
    return data


class OvitoView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewport = Viewport(type=Viewport.Type.Perspective, camera_dir=(2.0, 2.0, -1.4))
        self._viewport_window = create_qwidget(self.viewport, parent=self)
        self.viewport_widget = self._viewport_window.widget
        self.pipeline: Pipeline | None = None
        self._bond_modifier: CreateBondsModifier | None = None
        self._camera_initialized = False
        self._latest_snapshot: ParticleSnapshot | None = None
        self._include_substrate = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._make_legend())
        layout.addWidget(self.viewport_widget)

    @staticmethod
    def _make_legend() -> QWidget:
        legend = QFrame()
        legend.setObjectName("AtomLegend")
        row = QHBoxLayout(legend)
        row.setContentsMargins(12, 6, 12, 6)
        row.setSpacing(16)
        title = QLabel("ATOMS")
        title.setObjectName("LegendTitle")
        row.addWidget(title)
        for element, color in ATOM_COLORS.items():
            item = QWidget(legend)
            item_row = QHBoxLayout(item)
            item_row.setContentsMargins(0, 0, 0, 0)
            item_row.setSpacing(6)
            swatch = QLabel()
            swatch.setObjectName("LegendSwatch")
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(f"background-color: {color}; border-radius: 6px;")
            name = QLabel(element)
            name.setObjectName("LegendLabel")
            item_row.addWidget(swatch)
            item_row.addWidget(name)
            row.addWidget(item)
        subtitle = QLabel("Zinc-blende GaAs(001) · nearest-neighbor bonds")
        subtitle.setObjectName("LegendSubtitle")
        row.addStretch(1)
        row.addWidget(subtitle)
        return legend

    def set_snapshot(self, snapshot: ParticleSnapshot, include_substrate: bool = True) -> None:
        self._latest_snapshot = snapshot
        self._include_substrate = include_substrate
        self._refresh()

    def _refresh(self) -> None:
        if self._latest_snapshot is None:
            return
        data = self._build_data(self._latest_snapshot, self._include_substrate)
        if self.pipeline is None:
            self.pipeline = Pipeline(source=StaticSource(data=data))
            self._bond_modifier = CreateBondsModifier(cutoff=2.50)
            self._bond_modifier.vis.color = (0.58, 0.68, 0.79)
            self._bond_modifier.vis.width = 0.35
            self.pipeline.modifiers.append(self._bond_modifier)
            self.pipeline.add_to_scene()
        else:
            self.pipeline.source = StaticSource(data=data)
        self.pipeline.compute()
        self.viewport_widget.update()
        if not self._camera_initialized:
            self.viewport.zoom_all((self.viewport_widget.width(), self.viewport_widget.height()))
            self._camera_initialized = True

    def _build_data(self, snapshot: ParticleSnapshot, include_substrate: bool) -> DataCollection:
        return build_ovito_data(snapshot, include_substrate)

    def set_substrate_visible(self, visible: bool) -> None:
        self._include_substrate = visible
        self._refresh()


