"""Live diagnostic plots for the KMC model."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6.QtWidgets import QComboBox, QTabWidget, QVBoxLayout, QWidget
from pathlib import Path

from models import MetricsSnapshot


class MetricsPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tabs = QTabWidget(self)
        self.rheed_plot = pg.PlotWidget(title="RHEED intensity")
        self.rheed_plot.setLabel("bottom", "Physical time", units="s")
        self.rheed_plot.setLabel("left", "Intensity")
        self.rheed_plot.showGrid(x=True, y=True, alpha=0.25)
        self.rheed_curve = self.rheed_plot.plot(pen=pg.mkPen("#4c9aff", width=2))
        self.tabs.addTab(self.rheed_plot, "RHEED")

        self.roughness_plot = pg.PlotWidget(title="RMS roughness")
        self.roughness_plot.setLabel("bottom", "Physical time", units="s")
        self.roughness_plot.setLabel("left", "RMS", units="layers")
        self.roughness_plot.showGrid(x=True, y=True, alpha=0.25)
        self.roughness_curve = self.roughness_plot.plot(pen=pg.mkPen("#ff647c", width=2))
        self.tabs.addTab(self.roughness_plot, "RMS")

        coverage_widget = QWidget(self)
        coverage_layout = QVBoxLayout(coverage_widget)
        self.layer_select = QComboBox(coverage_widget)
        self.layer_select.currentIndexChanged.connect(self._update_selected_layer)
        coverage_layout.addWidget(self.layer_select)
        self.coverage_plot = pg.PlotWidget(title="Layer coverage")
        self.coverage_plot.setLabel("bottom", "Physical time", units="s")
        self.coverage_plot.setLabel("left", "Coverage θ")
        self.coverage_plot.setYRange(0, 1)
        self.coverage_plot.showGrid(x=True, y=True, alpha=0.25)
        self.coverage_curve = self.coverage_plot.plot(pen=pg.mkPen("#f5b942", width=2))
        coverage_layout.addWidget(self.coverage_plot, 1)
        self.tabs.addTab(coverage_widget, "Layer curve")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self._metrics: MetricsSnapshot | None = None

    def set_metrics(self, metrics: MetricsSnapshot) -> None:
        self._metrics = metrics
        self.rheed_curve.setData(metrics.time_sec, metrics.rheed_intensity)
        self.roughness_curve.setData(metrics.time_sec, metrics.roughness)
        if metrics.layer_coverage.size:
            current_names = [self.layer_select.itemText(i) for i in range(self.layer_select.count())]
            names = [f"Layer {h + 1}" for h in range(metrics.layer_coverage.shape[0])]
            if current_names != names:
                self.layer_select.blockSignals(True)
                self.layer_select.clear()
                self.layer_select.addItems(names)
                self.layer_select.blockSignals(False)
            self._update_selected_layer()

    def _update_selected_layer(self, _index: int = -1) -> None:
        if self._metrics is None or not self._metrics.layer_coverage.size:
            return
        layer = max(0, self.layer_select.currentIndex())
        coverage = self._metrics.layer_coverage[layer]
        self.coverage_curve.setData(self._metrics.time_sec, coverage)
        self.coverage_plot.setTitle(f"Layer coverage — layer {layer + 1}")

    def save_charts(self, directory: str | Path) -> list[Path]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        charts = (
            (self.rheed_plot, "rheed.png"),
            (self.roughness_plot, "rms_roughness.png"),
            (self.coverage_plot, "layer_coverage.png"),
        )
        saved: list[Path] = []
        for widget, filename in charts:
            path = target / filename
            ImageExporter(widget.plotItem).export(str(path))
            saved.append(path)
        return saved

