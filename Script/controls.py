"""Simulation controls and editable KMC parameters."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from models import KMCParameters


class SimulationToolbar(QWidget):
    run_requested = Signal()
    pause_requested = Signal()
    step_requested = Signal()
    stop_requested = Signal()
    reset_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SimulationToolbar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        self.run_button = QPushButton("▶  RUN")
        self.run_button.setObjectName("RunButton")
        self.pause_button = QPushButton("Ⅱ  PAUSE")
        self.step_button = QPushButton("→|  STEP")
        self.stop_button = QPushButton("■  STOP")
        self.reset_button = QPushButton("↺  RESET")
        self.progress_label = QLabel("t = 0 s  ·  cycle 0  ·  atoms 0")
        self.progress_label.setObjectName("ToolbarProgress")
        self.state_label = QLabel("READY")
        self.state_label.setObjectName("ToolbarState")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        for button in (self.run_button, self.pause_button, self.step_button, self.stop_button, self.reset_button):
            button.setMinimumHeight(38)
            layout.addWidget(button)
        layout.addStretch(1)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.state_label)

        self.run_button.clicked.connect(self.run_requested)
        self.pause_button.clicked.connect(self.pause_requested)
        self.step_button.clicked.connect(self.step_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.reset_button.clicked.connect(self.reset_requested)

    def set_state(self, state: str) -> None:
        self.state_label.setText(state.upper())

    def set_progress(self, time_sec: float, outer_cycles: int, atom_count: int) -> None:
        self.progress_label.setText(
            f"t = {time_sec:.5g} s  ·  cycle {outer_cycles}  ·  atoms {atom_count}"
        )


class ControlsPanel(QWidget):
    parameters_submitted = Signal(dict)
    substrate_visibility_changed = Signal(bool)

    def __init__(self, parameters: KMCParameters, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_applied = self._parameter_values(parameters)
        self._fields: dict[str, QDoubleSpinBox | QSpinBox] = {}

        parameters_group = QGroupBox("Parameters")
        self.form = QFormLayout(parameters_group)
        self._add_float("max_time_sec", "Maximum time (s)", parameters.max_time_sec, 0.001, 10000, 0.1)
        self._add_float("temperature_k", "Temperature T (K)", parameters.temperature_k, 1, 3000, 1)
        self._add_float("group_iii_flux", "Ga flux (ML/s)", parameters.group_iii_flux, 0.0001, 1000, 0.1)
        self._add_int("as_excess", "As excess attempts", parameters.as_excess, 0, 100000)
        self._add_float("e_des_ga", "E des Ga (eV)", parameters.e_des_ga, 0, 10, 0.01)
        self._add_float("e_des_as", "E des As (eV)", parameters.e_des_as, 0, 10, 0.01)
        self._add_float("ediff_ga", "E diff Ga (eV)", parameters.ediff_ga, 0, 10, 0.01)
        self._add_float("ediff_al", "E diff Al (eV)", parameters.ediff_al, 0, 10, 0.01)
        self._add_float("ediff_as", "E diff As (eV)", parameters.ediff_as, 0, 10, 0.01)
        self._add_float("e_lat_g3", "Lateral E group III", parameters.e_lat_g3, 0, 10, 0.01)
        self._add_float("e_lat_g5", "Lateral E As", parameters.e_lat_g5, 0, 10, 0.01)
        self._add_float("e_es", "Ehrlich–Schwoebel E", parameters.e_es, 0, 10, 0.01)
        self._add_int("ovito_every", "Snapshot every N cycles", parameters.ovito_every, 1, 100000)

        self.apply_button = QPushButton("Apply edited parameters")
        self.apply_button.clicked.connect(self._submit_parameters)
        self.form.addRow(self.apply_button)

        self.applied_label = QLabel(self._format_applied_values(self._last_applied))
        self.applied_label.setWordWrap(True)
        self.substrate_check = QCheckBox("Show substrate")
        self.substrate_check.setChecked(True)
        self.substrate_check.toggled.connect(self.substrate_visibility_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(parameters_group)
        layout.addWidget(self.applied_label)
        layout.addWidget(self.substrate_check)
        layout.addStretch(1)

    @staticmethod
    def _parameter_values(parameters: KMCParameters) -> dict[str, float | int]:
        return {
            "max_time_sec": parameters.max_time_sec,
            "temperature_k": parameters.temperature_k,
            "group_iii_flux": parameters.group_iii_flux,
            "as_excess": parameters.as_excess,
            "e_des_ga": parameters.e_des_ga,
            "e_des_as": parameters.e_des_as,
            "ediff_ga": parameters.ediff_ga,
            "ediff_al": parameters.ediff_al,
            "ediff_as": parameters.ediff_as,
            "e_lat_g3": parameters.e_lat_g3,
            "e_lat_g5": parameters.e_lat_g5,
            "e_es": parameters.e_es,
            "ovito_every": parameters.ovito_every,
        }

    def _add_float(self, name: str, label: str, value: float, minimum: float, maximum: float, step: float) -> None:
        field = QDoubleSpinBox(self)
        field.setRange(minimum, maximum)
        field.setDecimals(4)
        field.setSingleStep(step)
        field.setValue(value)
        self._fields[name] = field
        self.form.addRow(label, field)

    def _add_int(self, name: str, label: str, value: int, minimum: int, maximum: int) -> None:
        field = QSpinBox(self)
        field.setRange(minimum, maximum)
        field.setValue(value)
        self._fields[name] = field
        self.form.addRow(label, field)

    def _submit_parameters(self) -> None:
        changes = {
            name: field.value()
            for name, field in self._fields.items()
            if field.value() != self._last_applied[name]
        }
        if changes:
            self.parameters_submitted.emit(changes)

    def show_applied(self, entry: dict[str, object]) -> None:
        new_values = entry.get("new", {})
        if not isinstance(new_values, dict):
            return
        self._last_applied.update(new_values)
        if entry.get("event") == "reset_parameters_applied":
            for name, value in new_values.items():
                if name in self._fields:
                    self._fields[name].setValue(value)
        self.applied_label.setText(self._format_applied_values(self._last_applied))

    @staticmethod
    def _format_applied_values(values: dict[str, float | int]) -> str:
        summary = ", ".join(
            f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}"
            for name, value in values.items()
        )
        return f"Applied values: {summary}"

