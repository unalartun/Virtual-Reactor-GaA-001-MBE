"""Main PySide6 application window."""

from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFrame,
    QMainWindow,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QWidget,
)

from controls import ControlsPanel, SimulationToolbar
from kmc_worker import KMCController
from models import KMCParameters, MetricsSnapshot, ParticleSnapshot
from ovito_view import OvitoView
from plots import MetricsPanel


APP_STYLESHEET = """
QMainWindow, QWidget {
    background: #0b1118;
    color: #e6edf5;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QFrame#AppHeader {
    background: #111923;
    border: 1px solid #263445;
    border-radius: 12px;
}
QLabel#BrandTitle { color: #f4f8fc; font-size: 19pt; font-weight: 700; letter-spacing: 1px; }
QLabel#BrandCaption { color: #8fa2b7; font-size: 9pt; }
QLabel#EngineState {
    background: #142b2a;
    color: #77e0b6;
    border: 1px solid #24554a;
    border-radius: 9px;
    padding: 7px 12px;
    font-size: 9pt;
    font-weight: 700;
}
QWidget#SimulationToolbar, QFrame#AtomLegend {
    background: #111923;
    border: 1px solid #263445;
    border-radius: 10px;
}
QLabel#ToolbarState { color: #79c9ef; font-weight: 700; padding: 0 8px; }
QLabel#ToolbarProgress { color: #91a5b9; font-size: 9pt; padding: 0 10px; }
QLabel#LegendTitle { color: #8295aa; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }
QLabel#LegendLabel { color: #e4ebf3; font-weight: 600; }
QLabel#LegendSubtitle { color: #8698ab; font-size: 9pt; }
QTabWidget::pane {
    background: #101720;
    border: 1px solid #263445;
    border-radius: 10px;
    top: -1px;
}
QTabBar::tab {
    background: #111923;
    color: #91a2b5;
    padding: 10px 16px;
    margin-right: 4px;
    border: 1px solid #263445;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}
QTabBar::tab:selected { background: #1b2a38; color: #f4f8fc; border-bottom-color: #1b2a38; }
QGroupBox {
    background: #111923;
    border: 1px solid #263445;
    border-radius: 10px;
    margin-top: 13px;
    padding: 13px 10px 10px 10px;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #a9bacb; }
QPushButton {
    background: #1a2531;
    color: #dce6ef;
    border: 1px solid #334357;
    border-radius: 7px;
    padding: 8px 10px;
    font-weight: 700;
}
QPushButton:hover { background: #243447; border-color: #4d6a83; }
QPushButton:pressed { background: #14202c; }
QPushButton:disabled { color: #607080; background: #151c24; border-color: #202b36; }
QPushButton#RunButton { background: #12658d; border-color: #2387b2; color: #f4fbff; }
QPushButton#RunButton:hover { background: #167aa8; }
QDoubleSpinBox, QSpinBox {
    background: #0c131b;
    color: #e6edf5;
    border: 1px solid #2d3c4e;
    border-radius: 6px;
    padding: 5px 7px;
    min-height: 24px;
    selection-background-color: #207ba4;
}
QCheckBox { spacing: 8px; color: #c9d5e1; }
QCheckBox::indicator { width: 16px; height: 16px; }
QPlainTextEdit {
    background: #0d141c;
    color: #9fb2c5;
    border: 1px solid #263445;
    border-radius: 8px;
    padding: 6px;
    font-family: "Cascadia Mono";
    font-size: 9pt;
}
QSplitter::handle { background: #0b1118; width: 5px; }
QStatusBar { background: #0b1118; color: #91a2b5; border-top: 1px solid #202b36; }
"""


class MainWindow(QMainWindow):
    def __init__(self, parameters: KMCParameters | None = None) -> None:
        super().__init__()
        self.parameters = parameters or KMCParameters()
        self.setWindowTitle("Virtual Reactor — GaAs(001) KMC")
        self.resize(1500, 920)
        self.setMinimumSize(1120, 720)
        self.setStyleSheet(APP_STYLESHEET)

        self.ovito_view = OvitoView(self)
        self.simulation_toolbar = SimulationToolbar(self)
        self.metrics_panel = MetricsPanel(self)
        self.controls_panel = ControlsPanel(self.parameters, self)
        self.event_log = QPlainTextEdit(self)
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(1000)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left_layout.addWidget(self.ovito_view, 5)
        left_layout.addWidget(self.simulation_toolbar)
        self.event_log.setMaximumHeight(145)
        left_layout.addWidget(self.event_log)

        right_tabs = QTabWidget(self)
        right_tabs.addTab(self.metrics_panel, "Diagnostics")
        right_tabs.addTab(self.controls_panel, "Parameters")

        header = QFrame(self)
        header.setObjectName("AppHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 10, 18, 10)
        brand = QVBoxLayout()
        title = QLabel("VIRTUAL REACTOR")
        title.setObjectName("BrandTitle")
        caption = QLabel("GaAs (001)  ·  zinc-blende epitaxial growth")
        caption.setObjectName("BrandCaption")
        brand.addWidget(title)
        brand.addWidget(caption)
        header_layout.addLayout(brand)
        header_layout.addStretch(1)
        self.engine_badge = QLabel("STARTING ENGINE")
        self.engine_badge.setObjectName("EngineState")
        header_layout.addWidget(self.engine_badge)

        page = QWidget(self)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(14, 12, 14, 8)
        page_layout.setSpacing(12)
        page_layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(left)
        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([930, 540])
        page_layout.addWidget(splitter, 1)
        self.setCentralWidget(page)
        self._current_cycle = 0
        self._current_atoms = 0

        self.controller = KMCController(self.parameters)
        self._wire_signals()
        self._set_controls_enabled(False)
        self.statusBar().showMessage("Starting Python KMC engine…")
        self.controller.start()

    def _wire_signals(self) -> None:
        panel = self.controls_panel
        toolbar = self.simulation_toolbar
        toolbar.run_requested.connect(self.controller.run_requested)
        toolbar.pause_requested.connect(self.controller.pause_requested)
        toolbar.step_requested.connect(self.controller.step_requested)
        toolbar.stop_requested.connect(self.controller.stop_requested)
        toolbar.reset_requested.connect(self.controller.reset_requested)
        panel.parameters_submitted.connect(self.controller.parameters_requested)
        panel.substrate_visibility_changed.connect(self.ovito_view.set_substrate_visible)

        self.controller.initialized.connect(self._on_initialized)
        self.controller.state_changed.connect(self._on_state_changed)
        self.controller.metrics_ready.connect(self._on_metrics)
        self.controller.particles_ready.connect(self._on_particles)
        self.controller.parameters_applied.connect(self._on_parameters_applied)
        self.controller.trajectory_ready.connect(self._on_trajectory)
        self.controller.error_occurred.connect(self._on_error)
        self._trajectory_path: Path | None = None

    def _set_controls_enabled(self, enabled: bool) -> None:
        for button in (
            self.simulation_toolbar.run_button,
            self.simulation_toolbar.pause_button,
            self.simulation_toolbar.step_button,
            self.simulation_toolbar.stop_button,
            self.simulation_toolbar.reset_button,
            self.controls_panel.apply_button,
        ):
            button.setEnabled(enabled)

    def _on_initialized(
        self,
        metrics: MetricsSnapshot,
        particles: ParticleSnapshot,
        applied_parameters: dict[str, object],
    ) -> None:
        self.parameters.seed = applied_parameters.get("seed")
        self.metrics_panel.set_metrics(metrics)
        self.ovito_view.set_snapshot(particles, self.controls_panel.substrate_check.isChecked())
        self._set_controls_enabled(True)
        self._append_log("Python KMC ready. Seed: " + str(self.parameters.seed))

    def _on_state_changed(self, state: str) -> None:
        self.simulation_toolbar.set_state(state)
        self.engine_badge.setText(state.upper())
        self.statusBar().showMessage(state)
        self._append_log(state)
        if state == "Finished" or state.startswith("Stopped"):
            self._save_charts()

    def _on_metrics(self, metrics: MetricsSnapshot) -> None:
        self.metrics_panel.set_metrics(metrics)
        self._current_cycle = metrics.outer_cycles
        self.simulation_toolbar.set_progress(
            metrics.current_time_sec, self._current_cycle, self._current_atoms
        )
        self.statusBar().showMessage(
            f"t = {metrics.current_time_sec:.6g} s   |   outer cycles = {metrics.outer_cycles}"
        )

    def _on_particles(self, particles: ParticleSnapshot) -> None:
        self.ovito_view.set_snapshot(
            particles,
            self.controls_panel.substrate_check.isChecked(),
        )
        self._current_atoms = int(particles.particle_ids.size)
        self._current_cycle = particles.outer_cycle
        self.simulation_toolbar.set_progress(
            particles.time_sec, self._current_cycle, self._current_atoms
        )
        self.statusBar().showMessage(
            f"t = {particles.time_sec:.6g} s   |   atoms = {particles.particle_ids.size}"
        )

    def _on_parameters_applied(self, entry: dict[str, object]) -> None:
        self.controls_panel.show_applied(entry)
        changes = entry.get("new", {})
        rebuild = entry.get("full_rate_rebuild", False)
        self._append_log(
            f"Parameters applied at t={entry.get('simulation_time_sec', 0):.6g}s, "
            f"cycle {entry.get('outer_cycle', 0)}: {changes}; full rate rebuild={rebuild}"
        )

    def _on_trajectory(self, filename: str) -> None:
        self._trajectory_path = Path(filename)
        self.statusBar().showMessage(f"Trajectory: {filename}")

    def _save_charts(self) -> None:
        if self._trajectory_path is None:
            return
        try:
            paths = self.metrics_panel.save_charts(self._trajectory_path.parent)
            self._append_log("Saved diagnostic charts: " + ", ".join(path.name for path in paths))
        except Exception as exc:
            self._append_log("Chart export error: " + str(exc))

    def _on_error(self, message: str) -> None:
        self.statusBar().showMessage("KMC error")
        self.simulation_toolbar.set_state("Error")
        self.engine_badge.setText("ERROR")
        self._append_log("ERROR: " + message)
        self._set_controls_enabled(True)

    def _append_log(self, message: str) -> None:
        self.event_log.appendPlainText(message)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.controller.shutdown()
        event.accept()

