"""Qt worker that keeps KMC computation off the GUI thread."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot

from kmc_model import GaAsKMC
from models import KMCParameters, MetricsSnapshot, ParticleSnapshot


class KMCWorker(QObject):
    initialized = Signal(object, object, object)
    state_changed = Signal(str)
    metrics_ready = Signal(object)
    particles_ready = Signal(object)
    parameters_applied = Signal(object)
    trajectory_ready = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, parameters: KMCParameters) -> None:
        super().__init__()
        self._initial_parameters = replace(parameters)
        self.model: GaAsKMC | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self._advance_once)
        self._running = False
        self._stopped = False
        self._run_directory: Path | None = None
        self._trajectory_path: Path | None = None
        self._parameter_log_path: Path | None = None
        self._metrics_path: Path | None = None
        self._last_metric_count = 0
        self._last_particles_cycle = -1

    @Slot()
    def initialize(self) -> None:
        try:
            self.model = GaAsKMC(self._initial_parameters)
            self._create_run_files()
            metrics = self.model.get_metrics()
            particles = self.model.get_particle_snapshot()
            self.initialized.emit(metrics, particles, self._parameters_as_dict(self.model.parameters))
            self.state_changed.emit("Ready")
        except Exception as exc:  # Report worker startup errors in the UI.
            self.error_occurred.emit(str(exc))

    def _create_run_files(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._run_directory = Path.cwd() / "runs" / f"run_{stamp}"
        self._run_directory.mkdir(parents=True, exist_ok=False)
        self._trajectory_path = self._run_directory / "gaas_kmc.dump"
        self._parameter_log_path = self._run_directory / "parameter_changes.jsonl"
        self._metrics_path = self._run_directory / "metrics.csv"
        with self._metrics_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["time_sec", "rheed_intensity", "roughness", *
                 (f"theta_layer_{h}" for h in range(1, self._initial_parameters.max_height + 1))]
            )
        self._last_metric_count = 0
        self._write_parameter_log(
            {
                "event": "initialized",
                "timestamp": datetime.now().astimezone().isoformat(),
                "simulation_time_sec": 0.0,
                "outer_cycle": 0,
                "parameters": self._parameters_as_dict(self._initial_parameters),
            }
        )

    @staticmethod
    def _parameters_as_dict(parameters: KMCParameters) -> dict[str, object]:
        return {name: getattr(parameters, name) for name in parameters.__dataclass_fields__}

    def _write_parameter_log(self, entry: dict[str, object]) -> None:
        if self._parameter_log_path is None:
            return
        with self._parameter_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @Slot()
    def run(self) -> None:
        if self.model is None:
            return
        if self.model.finished or self._stopped:
            self.state_changed.emit("Finished — reset to start a new run")
            return
        self._running = True
        self.timer.start()
        self.state_changed.emit("Running")

    @Slot()
    def pause(self) -> None:
        self.timer.stop()
        self._running = False
        self.state_changed.emit("Paused")

    @Slot()
    def step(self) -> None:
        if self.model is None or self._running or self._stopped:
            return
        self._advance_one(force_publish=True)

    @Slot()
    def stop(self) -> None:
        self.timer.stop()
        self._running = False
        self._stopped = True
        self._publish(force=True)
        self.state_changed.emit("Stopped — reset to start a new run")

    @Slot()
    def shutdown(self) -> None:
        self.timer.stop()
        self._running = False

    @Slot()
    def reset(self) -> None:
        self.timer.stop()
        self._running = False
        self._stopped = False
        try:
            self.model = GaAsKMC(self._initial_parameters)
            self._create_run_files()
            self._last_particles_cycle = -1
            self.initialized.emit(
                self.model.get_metrics(),
                self.model.get_particle_snapshot(),
                self._parameters_as_dict(self.model.parameters),
            )
            self.parameters_applied.emit(
                {
                    "event": "reset_parameters_applied",
                    "new": self._parameters_as_dict(self._initial_parameters),
                    "full_rate_rebuild": True,
                }
            )
            self.state_changed.emit("Ready")
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    @Slot(dict)
    def apply_parameters(self, changes: dict[str, float | int]) -> None:
        if self.model is None or not changes:
            return
        try:
            old_values = {name: getattr(self.model.parameters, name) for name in changes}
            rates_rebuilt = self.model.apply_parameters(**changes)
            for name, value in changes.items():
                setattr(self._initial_parameters, name, value)
            entry = {
                "event": "parameters_applied",
                "timestamp": datetime.now().astimezone().isoformat(),
                "simulation_time_sec": self.model.current_time_sec,
                "outer_cycle": self.model.outer_cycles,
                "old": old_values,
                "new": changes,
                "full_rate_rebuild": rates_rebuilt,
            }
            self._write_parameter_log(entry)
            self.parameters_applied.emit(entry)
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    @Slot()
    def _advance_once(self) -> None:
        if self.model is None:
            return
        self._advance_one()

    def _advance_one(self, force_publish: bool = False) -> None:
        assert self.model is not None
        try:
            finished = self.model.advance_batch(1)
            if self.model.outer_cycles % self.model.parameters.rheed_every == 0 or finished or force_publish:
                metrics = self.model.get_metrics()
                self._persist_new_metrics(metrics)
                self.metrics_ready.emit(metrics)
            if (
                self.model.outer_cycles % self.model.parameters.ovito_every == 0
                or finished
                or force_publish
                or self._last_particles_cycle < 0
            ):
                snapshot = self.model.get_particle_snapshot()
                self.particles_ready.emit(snapshot)
                self._last_particles_cycle = self.model.outer_cycles
                if self._trajectory_path is not None:
                    self.model.export_dump(self._trajectory_path, self.model.outer_cycles)
                    self.trajectory_ready.emit(str(self._trajectory_path))
            if finished:
                self.timer.stop()
                self._running = False
                self._stopped = True
                self.state_changed.emit("Finished")
                self.finished.emit()
        except Exception as exc:
            self.timer.stop()
            self._running = False
            self.error_occurred.emit(str(exc))
            self.state_changed.emit("Error")

    def _publish(self, force: bool = False) -> None:
        if self.model is None:
            return
        metrics = self.model.get_metrics()
        self._persist_new_metrics(metrics)
        self.metrics_ready.emit(metrics)
        snapshot = self.model.get_particle_snapshot()
        self.particles_ready.emit(snapshot)
        if force and self._trajectory_path is not None:
            if self.model.outer_cycles != self._last_particles_cycle:
                self.model.export_dump(self._trajectory_path, self.model.outer_cycles)
                self._last_particles_cycle = self.model.outer_cycles
            self.trajectory_ready.emit(str(self._trajectory_path))

    def _persist_new_metrics(self, metrics: MetricsSnapshot) -> None:
        if self._metrics_path is None:
            return
        start = self._last_metric_count
        if metrics.time_sec.size <= start:
            return
        with self._metrics_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            for index in range(start, metrics.time_sec.size):
                writer.writerow(
                    [
                        float(metrics.time_sec[index]),
                        float(metrics.rheed_intensity[index]),
                        float(metrics.roughness[index]),
                        *metrics.layer_coverage[:, index].tolist(),
                    ]
                )
        self._last_metric_count = int(metrics.time_sec.size)


class KMCController(QObject):
    """Thread-owning signal facade used by the main window."""

    run_requested = Signal()
    pause_requested = Signal()
    step_requested = Signal()
    stop_requested = Signal()
    shutdown_requested = Signal()
    reset_requested = Signal()
    parameters_requested = Signal(dict)

    initialized = Signal(object, object, object)
    state_changed = Signal(str)
    metrics_ready = Signal(object)
    particles_ready = Signal(object)
    parameters_applied = Signal(object)
    trajectory_ready = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, parameters: KMCParameters) -> None:
        super().__init__()
        self.thread = QThread(self)
        self.worker = KMCWorker(parameters)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.initialize)
        self.run_requested.connect(self.worker.run)
        self.pause_requested.connect(self.worker.pause)
        self.step_requested.connect(self.worker.step)
        self.stop_requested.connect(self.worker.stop)
        self.shutdown_requested.connect(
            self.worker.shutdown,
            Qt.ConnectionType.BlockingQueuedConnection,
        )
        self.reset_requested.connect(self.worker.reset)
        self.parameters_requested.connect(self.worker.apply_parameters)
        self.worker.initialized.connect(self.initialized)
        self.worker.state_changed.connect(self.state_changed)
        self.worker.metrics_ready.connect(self.metrics_ready)
        self.worker.particles_ready.connect(self.particles_ready)
        self.worker.parameters_applied.connect(self.parameters_applied)
        self.worker.trajectory_ready.connect(self.trajectory_ready)
        self.worker.error_occurred.connect(self.error_occurred)
        self.worker.finished.connect(self.finished)
        self.thread.finished.connect(self.worker.deleteLater)

    def start(self) -> None:
        if not self.thread.isRunning():
            self.thread.start()

    def shutdown(self) -> None:
        if not self.thread.isRunning():
            return
        self.shutdown_requested.emit()
        self.thread.quit()
        self.thread.wait(5000)

