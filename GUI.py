import sys
import os
import json
import warnings
from copy import deepcopy
import pyqtgraph as pg
import serial
import serial.tools.list_ports
import numpy as np
import control as ctrl
from datetime import datetime
from collections import deque
from time import monotonic, sleep
from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QButtonGroup, QDialog, QMessageBox
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import cont2discrete
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from control_runtime import (
    controller_to_zinv, deployment_coefficients, deployment_period_ms, simulate_pwm_loop,
    continuous_controller_coefficients, discretize_continuous_controller,
)
from system_identification import resample_held_input, simulate_zoh_transfer
from coefficient_edit import CoefficientEdit
from pid_tuning import tune_pid

KNOWN_VIDPID = {
    (0x16C0, 0x0483),  # Teensy 4.x USB-Serial (PJRC)
    (0x2341, 0x006D),  # Arduino UNO R4 WiFi
    (0x2341, 0x006C),  # Arduino UNO R4 Minima
}
PREF_DESCR = ("Teensy", "UNO R4")  # optional: bias by description text
LINE_EDIT_CONFIG_FILENAME = ".gui_config.json"


def centered_moving_average(time_s, values, window_s=0.03):
    """Display-only centered mean, with symmetric shorter windows at the edges.

    The nominal window is 30 ms (at least three samples). Symmetric sample
    weights avoid a time shift; the newest edge is revised as samples arrive.
    """
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if time_s.ndim != 1 or values.ndim != 1 or len(time_s) != len(values):
        raise ValueError("Time and signal must be equally sized one-dimensional arrays.")
    if len(values) < 3:
        return values.copy()
    intervals = np.diff(time_s)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if not len(intervals):
        return values.copy()
    half_window = max(1, int(round(window_s / (2 * np.median(intervals)))))
    index = np.arange(len(values))
    radius = np.minimum(half_window, np.minimum(index, len(values) - 1 - index))
    left, right = index - radius, index + radius + 1
    finite = np.isfinite(values)
    total = np.r_[0.0, np.cumsum(np.where(finite, values, 0.0))]
    count = np.r_[0, np.cumsum(finite)]
    counts = count[right] - count[left]
    result = np.full(len(values), np.nan)
    np.divide(total[right] - total[left], counts, out=result, where=counts > 0)
    return result


def _load_designer_form(window, filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    uic.loadUi(path, window)


def _fit_designer_window(window):
    available = window.screen().availableGeometry()
    window.setMinimumSize(min(window.minimumWidth(), available.width()),
                          min(window.minimumHeight(), available.height()))
    window.resize(min(window.width(), available.width()), min(window.height(), available.height()))


def auto_find_port():
    ports = list(serial.tools.list_ports.comports())
    # Prefer matches by VID/PID, then by description, else give up
    for p in ports:
        if (p.vid, p.pid) in KNOWN_VIDPID:
            return p.device
    for p in ports:
        if any(k in (p.description or "") for k in PREF_DESCR):
            return p.device
    return None

def plot_root_locus_clean(sys, ax):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="axis already exists; grid keyword ignored",
            category=UserWarning,
        )
        return ctrl.root_locus_plot(sys, ax=ax, grid=False, interactive=False)


def tf_num_den(sys):
    """Return numerator and denominator coefficients (descending powers)."""
    num = np.asarray(sys.num[0][0], dtype=float).ravel()
    den = np.asarray(sys.den[0][0], dtype=float).ravel()
    return num, den


def closed_loop_poles(sys, K):
    """
    Closed-loop poles of 1 + K*sys = 0 (unity negative feedback).

    Evaluated straight from the characteristic polynomial (den + K*num) instead
    of building a feedback() system every time, which is much faster and works
    the same for continuous (s) and discrete (z) transfer functions.
    """
    num, den = tf_num_den(sys)
    n = max(len(num), len(den))
    num = np.pad(num, (n - len(num), 0))
    den = np.pad(den, (n - len(den), 0))
    poly = den + float(K) * num

    nonzero = np.nonzero(np.abs(poly) > 1e-14)[0]
    if nonzero.size == 0:
        return np.array([], dtype=complex)
    poly = poly[nonzero[0]:]
    if poly.size <= 1:
        return np.array([], dtype=complex)
    return np.roots(poly)


def locus_gain_from_point(sys, point, k_seed=None):
    """
    Gain K whose closed-loop pole is closest to `point` in the s/z plane.

    Keep the gain implied by the characteristic equation as a candidate, then
    refine local minima near it and the sampled locus. Distance to the nearest
    pole is not unimodal over a wide gain range, especially on curved branches.
    """
    point = complex(point)
    num, den = tf_num_den(sys)

    def distance(K):
        poles = closed_loop_poles(sys, K)
        if poles.size == 0:
            return np.inf
        return float(np.min(np.abs(poles - point)))

    seeds = []
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = -np.polyval(den, point) / np.polyval(num, point)
        if np.isfinite(gain):
            seeds.extend([float(gain.real), float(abs(gain))])
    except Exception:
        pass

    if k_seed is not None:
        try:
            k_seed = float(k_seed)
            if np.isfinite(k_seed) and k_seed > 0:
                seeds.append(k_seed)
        except (TypeError, ValueError):
            pass

    seeds = sorted(set(seed for seed in seeds if np.isfinite(seed) and seed > 0)) or [1.0]
    best_k, best_distance = 0.0, distance(0.0)
    for seed in seeds:
        candidate_distance = distance(seed)
        if candidate_distance < best_distance:
            best_k, best_distance = seed, candidate_distance
    # A click exactly on a branch already satisfies D(point) + K*N(point) = 0.
    if best_distance <= 1e-12 * max(1.0, abs(point)):
        return float(best_k)

    log_grid = np.unique(np.concatenate([
        np.log(seed) + np.linspace(-np.log(4.0), np.log(4.0), 9) for seed in seeds
    ]))
    distances = np.asarray([distance(np.exp(log_k)) for log_k in log_grid])
    for i, candidate_distance in enumerate(distances):
        if candidate_distance < best_distance:
            best_k, best_distance = float(np.exp(log_grid[i])), candidate_distance
        if (0 < i < len(log_grid) - 1 and np.isfinite(candidate_distance)
                and candidate_distance <= distances[i - 1] and candidate_distance <= distances[i + 1]):
            result = minimize_scalar(
                lambda log_k: distance(np.exp(log_k)),
                bounds=(log_grid[i - 1], log_grid[i + 1]), method="bounded",
                options={"xatol": 1e-9, "maxiter": 60},
            )
            if result.fun < best_distance:
                best_k, best_distance = float(np.exp(result.x)), float(result.fun)
    if not np.isfinite(best_distance):
        raise ValueError("No finite closed-loop pole could be selected.")
    return float(best_k)


def k_slider_pos(K, kmin, kmax, steps=1000):
    """Map a gain to a slider position in [0, steps] with logarithmic spacing."""
    try:
        K = float(K)
    except (TypeError, ValueError):
        return 0
    kmin = max(float(kmin), 1e-15)
    kmax = max(float(kmax), kmin * 10.0)
    if not np.isfinite(K) or K <= 0:
        return 0
    frac = np.log(K / kmin) / np.log(kmax / kmin)
    return int(round(float(np.clip(frac, 0.0, 1.0)) * steps))


def k_from_slider_pos(pos, kmin, kmax, steps=1000):
    """Inverse of k_slider_pos: slider position -> gain (0 at position 0)."""
    if pos <= 0:
        return 0.0
    kmin = max(float(kmin), 1e-15)
    kmax = max(float(kmax), kmin * 10.0)
    frac = float(np.clip(pos, 0, steps)) / steps
    return float(kmin * (kmax / kmin) ** frac)


def design_state_key(prefix, sys, extra=()):
    """
    Key that identifies a plant/design tab, so a controller design can be
    restored only while the underlying transfer function is unchanged.
    """
    num, den = tf_num_den(sys)
    return (
        prefix,
        len(num),
        len(den),
        *np.round(num, 12).tolist(),
        *np.round(den, 12).tolist(),
        *tuple(extra),
    )


class SiglentDMMWorker(QtCore.QObject):
    sample_ready = QtCore.pyqtSignal(float, float, int)
    status_changed = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, resource_name=None, poll_interval_s=0.02, parent=None):
        super().__init__(parent)
        self.resource_name = resource_name
        self.poll_interval_s = poll_interval_s
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    @staticmethod
    def _configure_resource(inst):
        inst.timeout = 1500
        inst.write_termination = "\n"
        inst.read_termination = "\n"

    def _open_sdm3045x(self, rm):
        if self.resource_name:
            candidates = [self.resource_name]
        else:
            candidates = [
                resource
                for resource in rm.list_resources()
                if str(resource).upper().startswith("USB")
            ]

        last_error = None
        for resource in candidates:
            inst = None
            try:
                inst = rm.open_resource(resource)
                self._configure_resource(inst)
                idn = inst.query("*IDN?").strip()
                if "SIGLENT" in idn.upper() and "SDM3045X" in idn.upper():
                    return inst, resource, idn
                inst.close()
            except Exception as exc:
                last_error = exc
                if inst is not None:
                    try:
                        inst.close()
                    except Exception:
                        pass

        if last_error is not None:
            raise RuntimeError(f"No se encontro el SDM3045X por VISA: {last_error}")
        raise RuntimeError("No se encontro ningun recurso USB VISA para el SDM3045X.")

    @QtCore.pyqtSlot()
    def run(self):
        rm = None
        inst = None
        try:
            import pyvisa

            rm = pyvisa.ResourceManager()
            inst, _resource, _idn = self._open_sdm3045x(rm)
            self.status_changed.emit("DMM connected")

            inst.write("*CLS")
            inst.write("CONF:CURR:DC 10")
            inst.write("CURR:DC:RANG 10")
            inst.write("CURR:DC:RANG:AUTO OFF")
            inst.write("CURR:DC:NPLC 0.3")
            inst.write("SAMP:COUN 1")
            inst.write("TRIG:SOUR IMM")
            self.status_changed.emit("DMM ready")

            sample_id = 0
            while not self._stop_requested:
                loop_start = monotonic()
                raw = inst.query("READ?").strip().split(",")[0]
                value = float(raw)
                sample_id += 1
                self.sample_ready.emit(value, monotonic(), sample_id)

                remaining = self.poll_interval_s - (monotonic() - loop_start)
                while remaining > 0 and not self._stop_requested:
                    nap = min(remaining, 0.01)
                    sleep(nap)
                    remaining -= nap
        except Exception as exc:
            if not self._stop_requested:
                self.error.emit(str(exc))
        finally:
            if inst is not None:
                try:
                    inst.close()
                except Exception:
                    pass
            if rm is not None:
                try:
                    rm.close()
                except Exception:
                    pass
            self.finished.emit()

class ControlPlotDialog(QDialog):
    """
    Root locus big window:
      - Left: Root locus + Step response (stacked)
      - Right: Text output + compensator builder
      - Click on locus => updates BOTH info + step plot
    """

    def __init__(self, parent, sys, state=None):
        super().__init__(parent)
        _load_designer_form(self, "ui/RootLocusDialog.ui")

        flags = self.windowFlags()
        self.setWindowFlags(
            flags
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        self.base_sys = sys
        self.sys = sys
        self.comp_zeros = []
        self.comp_poles = []
        self.comp_integrators = 0
        self.comp_origin_zeros = 0
        self.comp_history = []
        self.comp_czero_pairs = []
        self.comp_cpole_pairs = []

        # Design state carried over from a previous time this window was open
        # (compensator + selected gain), so closing it does not throw the tuning away.
        state = state or {}
        self.comp_zeros = list(state.get("comp_zeros", []))
        self.comp_poles = list(state.get("comp_poles", []))
        self.comp_czero_pairs = [tuple(v) for v in state.get("comp_czero_pairs", [])]
        self.comp_cpole_pairs = [tuple(v) for v in state.get("comp_cpole_pairs", [])]
        self.comp_integrators = int(state.get("comp_integrators", 0))
        self.comp_origin_zeros = int(state.get("comp_origin_zeros", 0))
        self.comp_history = list(state.get("comp_history", []))

        self.selected_k = float(state.get("selected_k", 0.0) or 0.0)
        self.selected_pole = None
        self._sel_artists = []
        self._view_lims = None

        # ---- Figure: 2 rows (RL on top, Step on bottom) ----
        self.fig = self.canvas.figure
        self.fig.set_layout_engine("constrained")
        self.ax_rl = self.fig.add_subplot(211)
        self.ax_step = self.fig.add_subplot(212)
        self.toolbar.bind_canvas(self.canvas)

        # Grid refresh state: the damping grid depends on the visible limits.
        self._grid_artists = []
        self._suppress_grid_events = False
        self._grid_timer = QTimer(self)
        self._grid_timer.setSingleShot(True)
        self._grid_timer.setInterval(40)
        self._grid_timer.timeout.connect(self._redraw_grid)
        self.ax_rl.callbacks.connect("xlim_changed", self._on_view_limits_changed)
        self.ax_rl.callbacks.connect("ylim_changed", self._on_view_limits_changed)

        _fit_designer_window(self)

        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)

        # Root locus picking state
        self._rlist = None
        self._klist = None

        # Connect buttons
        self.comp_add_btn.clicked.connect(self._add_compensator_element)
        self.comp_undo_btn.clicked.connect(self._undo_compensator_element)
        self.comp_clear_btn.clicked.connect(self._clear_compensator)
        self.comp_type.currentIndexChanged.connect(self._update_comp_input_visibility)
        self.k_slider.valueChanged.connect(self._on_k_slider_changed)
        self.k_value.editingFinished.connect(self._on_k_edit_finished)
        self.export_btn.clicked.connect(self._close_and_export_to_continuous_tab)

        # Initial plots
        self._rebuild_compensated_system()
        self._plot_root_locus_initial()
        self._update_comp_display()
        self._update_comp_input_visibility()
        self._set_k_display(self.selected_k)
        self._apply_gain(self.selected_k)

        # Connect click picking and zooming
        self.canvas.mpl_connect("button_press_event", self._on_click_rlocus)
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)

    def _close_and_export_to_continuous_tab(self):
        try:
            controller = float(self.selected_k or 0) * self.comp_sys
            if self.parent()._import_continuous_controller(controller):
                self.parent()._remember_controller_design("continuous", self)
                self.accept()
        except (ValueError, TypeError, OverflowError) as exc:
            self.info.append(f"\nExport error: {exc}")

    # ---------- Compensator builder ----------
    def _build_compensator_tf(self):
        s = ctrl.TransferFunction.s
        C = ctrl.TransferFunction([1], [1])

        for z in self.comp_zeros:
            C *= (s + z)

        for p in self.comp_poles:
            C *= 1 / (s + p)

        for zr, zi in self.comp_czero_pairs:
            C *= ((s + zr) ** 2 + zi ** 2)

        for pr, pi in self.comp_cpole_pairs:
            C *= 1 / (((s + pr) ** 2 + pi ** 2))

        for _ in range(self.comp_integrators):
            C *= 1 / s

        for _ in range(self.comp_origin_zeros):
            C *= s

        return ctrl.minreal(C, verbose=False)

    def _update_comp_input_visibility(self):
        kind = self.comp_type.currentText()

        need_imag = kind in ("Complex zero pair", "Complex pole pair")
        need_location = kind in ("Real zero", "Real pole", "Complex zero pair", "Complex pole pair")

        self.comp_value.setVisible(need_location)
        self.comp_imag.setVisible(need_imag)

        form = getattr(self, "comp_form", None)
        if form is not None:
            try:
                form.labelForField(self.comp_value).setVisible(need_location)
                form.labelForField(self.comp_imag).setVisible(need_imag)
            except Exception:
                pass

        if kind == "Real zero":
            self.comp_value.setPlaceholderText("e.g. 2  ->  s = -2")
        elif kind == "Real pole":
            self.comp_value.setPlaceholderText("e.g. 2  ->  s = -2")
        elif kind in ("Complex zero pair", "Complex pole pair"):
            self.comp_value.setPlaceholderText("e.g. 2  ->  real part = -2")
            self.comp_imag.setPlaceholderText("e.g. 3  ->  ±j3")

    def _rebuild_compensated_system(self):
        C = self._build_compensator_tf()
        self.comp_sys = C
        self.sys = ctrl.minreal(C * self.base_sys, verbose=False)

    def _update_comp_display(self):
        try:
            C = self._build_compensator_tf()
            txt = "C(s):\n"
            txt += self.parent().tf_to_pretty_str(C)

            if self.comp_zeros:
                txt += "\n\nReal zeros at:\n" + ", ".join([f"-{z:g}" for z in self.comp_zeros])

            if self.comp_poles:
                txt += "\n\nReal poles at:\n" + ", ".join([f"-{p:g}" for p in self.comp_poles])

            if self.comp_czero_pairs:
                txt += "\n\nComplex zero pairs at:\n"
                txt += "\n".join([f"-{zr:g} ± j{zi:g}" for zr, zi in self.comp_czero_pairs])

            if self.comp_cpole_pairs:
                txt += "\n\nComplex pole pairs at:\n"
                txt += "\n".join([f"-{pr:g} ± j{pi:g}" for pr, pi in self.comp_cpole_pairs])

            if self.comp_integrators:
                txt += f"\n\nIntegrators:\n{self.comp_integrators} × 1/s"

            if self.comp_origin_zeros:
                txt += f"\n\nZeros at origin:\n{self.comp_origin_zeros} × s"

            self.comp_display.setText(txt)

        except Exception as e:
            self.comp_display.setText(f"Compensator error: {e}")

    def _add_compensator_element(self):
        try:
            kind = self.comp_type.currentText()

            if kind == "Real zero":
                txt = self.comp_value.text().strip()
                if not txt:
                    self.comp_display.setText("Enter a positive value for the real zero.")
                    return
                val = float(txt)
                if val <= 0:
                    self.comp_display.setText("Please enter a positive value.")
                    return
                self.comp_zeros.append(val)
                self.comp_history.append(("real_zero", val))

            elif kind == "Real pole":
                txt = self.comp_value.text().strip()
                if not txt:
                    self.comp_display.setText("Enter a positive value for the real pole.")
                    return
                val = float(txt)
                if val <= 0:
                    self.comp_display.setText("Please enter a positive value.")
                    return
                self.comp_poles.append(val)
                self.comp_history.append(("real_pole", val))

            elif kind == "Complex zero pair":
                txt_r = self.comp_value.text().strip()
                txt_i = self.comp_imag.text().strip()

                if not txt_r or not txt_i:
                    self.comp_display.setText("Enter positive real and imaginary parts.")
                    return

                zr = float(txt_r)
                zi = float(txt_i)

                if zr <= 0 or zi <= 0:
                    self.comp_display.setText("Please enter positive real and imaginary parts.")
                    return

                self.comp_czero_pairs.append((zr, zi))
                self.comp_history.append(("complex_zero_pair", (zr, zi)))

            elif kind == "Complex pole pair":
                txt_r = self.comp_value.text().strip()
                txt_i = self.comp_imag.text().strip()

                if not txt_r or not txt_i:
                    self.comp_display.setText("Enter positive real and imaginary parts.")
                    return

                pr = float(txt_r)
                pi = float(txt_i)

                if pr <= 0 or pi <= 0:
                    self.comp_display.setText("Please enter positive real and imaginary parts.")
                    return

                self.comp_cpole_pairs.append((pr, pi))
                self.comp_history.append(("complex_pole_pair", (pr, pi)))

            elif kind == "Integrator (1/s)":
                self.comp_integrators += 1
                self.comp_history.append(("integrator", None))

            elif kind == "Zero at origin (s)":
                self.comp_origin_zeros += 1
                self.comp_history.append(("origin_zero", None))

            self.comp_value.clear()
            self.comp_imag.clear()

            self._rebuild_compensated_system()
            self._plot_root_locus_initial()
            self._update_comp_display()
            self._set_k_display(self.selected_k)
            self._apply_gain(self.selected_k)

        except Exception as e:
            self.comp_display.setText(f"Add error: {e}")

    def _undo_compensator_element(self):
        if not self.comp_history:
            return

        kind, val = self.comp_history.pop()

        if kind == "real_zero" and self.comp_zeros:
            self.comp_zeros.pop()
        elif kind == "real_pole" and self.comp_poles:
            self.comp_poles.pop()
        elif kind == "complex_zero_pair" and self.comp_czero_pairs:
            self.comp_czero_pairs.pop()
        elif kind == "complex_pole_pair" and self.comp_cpole_pairs:
            self.comp_cpole_pairs.pop()
        elif kind == "integrator" and self.comp_integrators > 0:
            self.comp_integrators -= 1
        elif kind == "origin_zero" and self.comp_origin_zeros > 0:
            self.comp_origin_zeros -= 1

        self._rebuild_compensated_system()
        self._plot_root_locus_initial()
        self._update_comp_display()
        self._set_k_display(self.selected_k)
        self._apply_gain(self.selected_k)

    def _clear_compensator(self):
        self.comp_zeros = []
        self.comp_poles = []
        self.comp_czero_pairs = []
        self.comp_cpole_pairs = []
        self.comp_integrators = 0
        self.comp_origin_zeros = 0
        self.comp_history = []

        self._rebuild_compensated_system()
        self._plot_root_locus_initial()
        self._update_comp_display()
        self._set_k_display(self.selected_k)
        self._apply_gain(self.selected_k)

    # ---------- Root locus ----------
    def _plot_root_locus_initial(self):
        self._suppress_grid_events = True
        try:
            self.ax_rl.clear()
            self._grid_artists = []
            self._view_lims = None
            self._sel_artists = []

            rldata = ctrl.root_locus_map(self.sys)
            self._rlist = np.asarray(rldata.loci)
            self._klist = np.asarray(rldata.gains)

            plot_root_locus_clean(self.sys, self.ax_rl)

            xleft, _ = self.ax_rl.get_xlim()
            self.ax_rl.set_xlim(xleft, 0.0)

            ylow, yhigh = self.ax_rl.get_ylim()
            ymax = max(abs(ylow), abs(yhigh), 1.0)
            self.ax_rl.set_ylim(-ymax, ymax)

            self.ax_rl.set_title("")
            self.ax_rl.set_xlabel("")
            self.ax_rl.set_ylabel("")
        finally:
            self._suppress_grid_events = False

        self._redraw_grid()
        self.canvas.draw_idle()

    def _on_view_limits_changed(self, _ax=None):
        if getattr(self, "_suppress_grid_events", False):
            return
        self._grid_timer.start()

    def _redraw_grid(self):
        """Remove and redraw the damping grid for the current visible limits."""
        self._suppress_grid_events = True
        try:
            for artist in self._grid_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            self._grid_artists = []

            parent = self.parent()
            if hasattr(parent, "_overlay_sgrid"):
                self._grid_artists = list(
                    parent._overlay_sgrid(self.ax_rl, show_labels=True)
                )
        finally:
            self._suppress_grid_events = False
        self.canvas.draw_idle()

    def _nearest_locus_seed(self, point):
        """Gain of the closest sample in the coarse locus data (search seed)."""
        if self._rlist is None or self._klist is None:
            return None
        try:
            diffs = np.abs(self._rlist - point)
            idx_flat = int(np.argmin(diffs))
            ik, _ = np.unravel_index(idx_flat, diffs.shape)
            return float(self._klist[ik])
        except Exception:
            return None

    def _pixel_distance(self, p1, p2):
        try:
            x1, y1 = self.ax_rl.transData.transform((p1.real, p1.imag))
            x2, y2 = self.ax_rl.transData.transform((p2.real, p2.imag))
            return float(np.hypot(x1 - x2, y1 - y2))
        except Exception:
            return 0.0

    def _on_click_rlocus(self, event):
        if getattr(self.toolbar, "mode", ""):
            return   # pan/zoom tool active: let the toolbar handle the click
        if getattr(event, "button", 1) != 1:
            return
        if event.inaxes != self.ax_rl:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._rlist is None or self._klist is None:
            return

        s_click = complex(event.xdata, event.ydata)
        seed = self._nearest_locus_seed(s_click)
        try:
            K = locus_gain_from_point(self.sys, s_click, k_seed=seed)
        except Exception:
            return

        poles = closed_loop_poles(self.sys, K)
        if poles.size == 0:
            return
        pole = poles[int(np.argmin(np.abs(poles - s_click)))]

        # Ignore clicks that are not near any branch of the locus.
        if self._pixel_distance(s_click, pole) > 30:
            return

        self._apply_gain(K, pole=pole)

    def _apply_gain(self, K, pole=None):
        """Set the loop gain, refresh the markers, the step response and info."""
        try:
            K = float(K)
        except (TypeError, ValueError):
            return
        if not np.isfinite(K) or K < 0:
            return
        if K < 1e-9:
            K = 0.0

        self.selected_k = K
        self._set_k_display(K)

        if K > 0:
            poles = closed_loop_poles(self.sys, K)
        else:
            poles = np.asarray(ctrl.poles(self.sys), dtype=complex).ravel()

        if poles.size:
            if pole is None or not np.isfinite(pole):
                pole = poles[int(np.argmax(np.asarray(poles).real))]
            else:
                pole = poles[int(np.argmin(np.abs(poles - complex(pole))))]
        else:
            pole = None
        self.selected_pole = pole

        for artist in self._sel_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._sel_artists = []

        for p in np.atleast_1d(poles):
            if np.isfinite(p.real) and np.isfinite(p.imag):
                self._sel_artists.append(
                    self.ax_rl.plot(
                        [p.real], [p.imag],
                        marker="s", markersize=5,
                        markerfacecolor="0.55", markeredgecolor="0.35",
                        linestyle="None", zorder=5,
                    )[0]
                )

        if pole is not None and np.isfinite(pole.real) and np.isfinite(pole.imag):
            self._sel_artists.append(
                self.ax_rl.plot(
                    [pole.real], [pole.imag],
                    marker="o", markersize=9,
                    markerfacecolor="red", markeredgecolor="red",
                    linestyle="None", zorder=6,
                )[0]
            )

        self.canvas.draw_idle()
        self._plot_step_for_K(K)
        if pole is not None and K > 0:
            self._update_info_for_K(K, pole)
        else:
            self._set_info_idle()

    def _k_slider_range(self):
        karr = np.asarray(self._klist if self._klist is not None else [], dtype=float).ravel()
        positive = karr[np.isfinite(karr) & (karr > 0)]
        if positive.size == 0:
            return 1e-3, 1.0
        kmax = float(np.max(positive))
        return kmax * 1e-6, kmax

    def _set_k_display(self, K):
        self.k_value.blockSignals(True)
        try:
            self.k_value.setText(f"{float(K):.6g}")
        except (TypeError, ValueError):
            self.k_value.setText("")
        self.k_value.blockSignals(False)

        kmin, kmax = self._k_slider_range()
        self.k_slider.blockSignals(True)
        self.k_slider.setValue(k_slider_pos(K, kmin, kmax))
        self.k_slider.blockSignals(False)

    def _on_k_slider_changed(self, value):
        kmin, kmax = self._k_slider_range()
        self._apply_gain(k_from_slider_pos(value, kmin, kmax))

    def _on_k_edit_finished(self):
        text = self.k_value.text().strip()
        try:
            K = float(text)
        except ValueError:
            self._set_k_display(self.selected_k)
            return
        if not np.isfinite(K) or K < 0:
            self._set_k_display(self.selected_k)
            return
        self._apply_gain(K)

    def _on_scroll_zoom(self, event):
        if event.inaxes != self.ax_rl:
            return
        if event.xdata is None or event.ydata is None:
            return
        button = getattr(event, "button", None)
        if button == "up":
            factor = 0.80
        elif button == "down":
            factor = 1.25
        else:
            return

        x0, x1 = self.ax_rl.get_xlim()
        y0, y1 = self.ax_rl.get_ylim()
        cx, cy = event.xdata, event.ydata
        self._suppress_grid_events = True
        try:
            self.ax_rl.set_xlim(cx + (x0 - cx) * factor, cx + (x1 - cx) * factor)
            self.ax_rl.set_ylim(cy + (y0 - cy) * factor, cy + (y1 - cy) * factor)
        finally:
            self._suppress_grid_events = False
        self._view_lims = (self.ax_rl.get_xlim(), self.ax_rl.get_ylim())
        self._redraw_grid()

    def get_state(self):
        """Snapshot of the design so it can be restored on the next open."""
        return {
            "comp_zeros": list(self.comp_zeros),
            "comp_poles": list(self.comp_poles),
            "comp_czero_pairs": [tuple(v) for v in self.comp_czero_pairs],
            "comp_cpole_pairs": [tuple(v) for v in self.comp_cpole_pairs],
            "comp_integrators": int(self.comp_integrators),
            "comp_origin_zeros": int(self.comp_origin_zeros),
            "comp_history": list(self.comp_history),
            "selected_k": float(self.selected_k),
        }

    # ---------- Step response ----------
    def _plot_step_for_K(self, K: float):
        self.ax_step.clear()

        L = K * self.sys
        T = ctrl.feedback(L, 1, sign=-1)

        try:
            t = np.linspace(0, 0.5, 800)
            t, y = ctrl.step_response(T, T=t)

            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    info = ctrl.step_info(T)
                ts = float(info.get("SettlingTime", np.nan))
                if np.isfinite(ts) and ts > 0:
                    t_final = max(0.12, 2.5 * ts)
                    t = np.linspace(0, t_final, 1000)
                    t, y = ctrl.step_response(T, T=t)
            except Exception:
                pass

            self.parent()._plot_time_mpl(self.ax_step, t, y)
            self.ax_step.set_xlim(float(t[0]), float(t[-1]))

            ymin = float(np.min(y))
            ymax = float(np.max(y))
            if abs(ymax - ymin) < 1e-9:
                ymin -= 0.05
                ymax += 0.05
            else:
                margin = 0.1 * (ymax - ymin)
                ymin -= margin
                ymax += margin

            self.ax_step.set_ylim(ymin, ymax)

        except Exception as e:
            self.ax_step.text(
                0.05, 0.5,
                f"step_response error: {e}",
                transform=self.ax_step.transAxes
            )

        self.ax_step.grid(True, which="both")
        self.ax_step.set_title("")
        self.ax_step.set_xlabel("")
        self.ax_step.set_ylabel("")
        self.canvas.draw_idle()

    # ---------- Info panel ----------
    def _set_info_idle(self):
        self.info.setText(
            "Root locus + step response\n\n"
            "No gain selected yet (K = 0)."
        )

    def _update_info_for_K(self, K: float, pole):
        L = K * self.sys
        T = ctrl.feedback(L, 1, sign=-1)

        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                si = ctrl.step_info(T)
            step_txt = "\n".join([f"{k}: {si[k]}" for k in si])
        except Exception as e:
            step_txt = f"step_info error: {e}"

        try:
            bw = ctrl.bandwidth(T)
            bw_txt = f"{bw} rad/s"
        except Exception as e:
            bw_txt = f"bandwidth error: {e}"

        try:
            gm, pm, wg, wp = ctrl.margin(L)
            margin_txt = (
                f"gm: {gm}\n"
                f"pm: {pm} deg\n"
                f"w_g: {wg} rad/s\n"
                f"w_p: {wp} rad/s"
            )
        except Exception as e:
            margin_txt = f"margin error: {e}"

        try:
            Ctxt = self.parent().tf_to_pretty_str(self.comp_sys)
        except Exception:
            Ctxt = str(self.comp_sys)

        self.info.setText(
            f"Selected point\n"
            f"K = {K}\n"
            f"Pole = {pole}\n\n"
            f"C(s):\n{Ctxt}\n\n"
            f"Closed-loop step_info(T):\n{step_txt}\n\n"
            f"Closed-loop bandwidth(T) (-3 dB): {bw_txt}\n\n"
            f"Open-loop margins (L = K*C*G):\n{margin_txt}"
        )

class DiscreteControlPlotDialog(QDialog):
    """
    Discrete root locus big window:
      - Left: Root locus in z-plane + step response
      - Right: text info + gain slider + compensator builder
      - Click on locus => updates info + step plot + slider
      - Export K*C(z) to digital controller tab (A..I)
    """

    def __init__(self, parent, sysz, Ts, state=None):
        super().__init__(parent)
        _load_designer_form(self, "ui/DiscreteRootLocusDialog.ui")

        flags = self.windowFlags()
        self.setWindowFlags(
            flags
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        self.base_sys = sysz
        self.sys = sysz
        self.Ts = float(Ts)

        # ---- compensator storage in z-plane ----
        self.comp_zeros = []
        self.comp_poles = []
        self.comp_czero_pairs = []
        self.comp_cpole_pairs = []
        self.comp_origin_zeros = 0
        self.comp_delays = 0
        self.comp_integrators = 0
        self.comp_history = []

        # Design state carried over from a previous open (compensator + gain).
        state = state or {}
        self.comp_zeros = list(state.get("comp_zeros", []))
        self.comp_poles = list(state.get("comp_poles", []))
        self.comp_czero_pairs = [tuple(v) for v in state.get("comp_czero_pairs", [])]
        self.comp_cpole_pairs = [tuple(v) for v in state.get("comp_cpole_pairs", [])]
        self.comp_origin_zeros = int(state.get("comp_origin_zeros", 0))
        self.comp_delays = int(state.get("comp_delays", 0))
        self.comp_integrators = int(state.get("comp_integrators", 0))
        self.comp_history = list(state.get("comp_history", []))

        # ---- selected locus point ----
        self.selected_k = float(state.get("selected_k", 0.0) or 0.0)
        self.selected_pole = None
        self.selected_index = 0
        self._base_drawn = False
        self._view_lims = None
        self._sel_artists = []

        # ---- figure ----
        self.fig = self.canvas.figure
        self.fig.set_layout_engine("constrained")
        self.ax_rl = self.fig.add_subplot(211)
        self.ax_step = self.fig.add_subplot(212)
        self.toolbar.bind_canvas(self.canvas)

        # Grid refresh state: the z-grid depends on the visible limits.
        self._grid_artists = []
        self._suppress_grid_events = False
        self._grid_timer = QTimer(self)
        self._grid_timer.setSingleShot(True)
        self._grid_timer.setInterval(40)
        self._grid_timer.timeout.connect(self._redraw_grid)
        self.ax_rl.callbacks.connect("xlim_changed", self._on_view_limits_changed)
        self.ax_rl.callbacks.connect("ylim_changed", self._on_view_limits_changed)

        self.pwm_mode.setItemData(0, 2)
        self.pwm_mode.setItemData(1, 3)
        self.pwm_plant_input.setItemData(0, "applied")
        self.pwm_plant_input.setItemData(1, "effective")
        self.pwm_mode.setCurrentIndex(1 if "Position" in parent.modooperacion.currentText() else 0)
        self.pwm_deadzone.setValue(parent._line_edit_int(parent.deadzone, minimum=0, maximum=255))
        self.pwm_bits.setValue(parent.firmware_pwm_bits or (12 if parent.gui_only else 8))
        pwm_state = state.get("pwm", {})
        if pwm_state:
            self.pwm_compare.setChecked(bool(pwm_state.get("compare", False)))
            self.pwm_reference.setValue(int(pwm_state.get("reference", 1)))
            for combo, key in ((self.pwm_mode, "mode"), (self.pwm_plant_input, "plant_input")):
                index = combo.findData(pwm_state.get(key))
                if index >= 0:
                    combo.setCurrentIndex(index)
            self.pwm_deadzone.setValue(int(pwm_state.get("deadzone", self.pwm_deadzone.value())))
            self.pwm_bits.setValue(int(pwm_state.get("bits", self.pwm_bits.value())))
        _fit_designer_window(self)

        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)

        # ---- root locus state ----
        self._rlist = None
        self._klist = None

        # ---- signals ----
        self.comp_add_btn.clicked.connect(self._add_compensator_element)
        self.comp_undo_btn.clicked.connect(self._undo_compensator_element)
        self.comp_clear_btn.clicked.connect(self._clear_compensator)
        self.comp_type.currentIndexChanged.connect(self._update_comp_input_visibility)

        self.k_slider.valueChanged.connect(self._on_k_slider_changed)
        self.k_value.editingFinished.connect(self._on_k_edit_finished)
        self.export_btn.clicked.connect(self._close_and_export_to_digital_tab)
        self.pwm_compare.toggled.connect(lambda: self._plot_step_for_K(self.selected_k))
        self.pwm_reference.valueChanged.connect(lambda: self._plot_step_for_K(self.selected_k))
        self.pwm_mode.currentIndexChanged.connect(lambda: self._plot_step_for_K(self.selected_k))
        self.pwm_deadzone.valueChanged.connect(lambda: self._plot_step_for_K(self.selected_k))
        self.pwm_bits.valueChanged.connect(lambda: self._plot_step_for_K(self.selected_k))
        self.pwm_plant_input.currentIndexChanged.connect(lambda: self._plot_step_for_K(self.selected_k))

        self.canvas.mpl_connect("button_press_event", self._on_click_rlocus)
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)

        # ---- init ----
        self._rebuild_compensated_system()
        self._compute_root_locus_data()
        self._update_comp_input_visibility()
        self._update_comp_display()

        if self._klist is not None and len(self._klist) > 0:
            self._select_exact_k(self.selected_k)
        else:
            self._draw_root_locus(selected_poles=None)
            self._plot_step_for_K(0.0)
            self._set_info_idle()
            self._set_k_display(0.0)

    # =========================================================
    # Compensator builder
    # =========================================================
    def _build_compensator_tf(self):
        z = ctrl.TransferFunction([1, 0], [1], self.Ts)
        C = ctrl.TransferFunction([1], [1], self.Ts)

        for zz in self.comp_zeros:
            C *= (z - zz)

        for pp in self.comp_poles:
            C *= 1 / (z - pp)

        for zr, zi in self.comp_czero_pairs:
            C *= (z**2 - 2*zr*z + (zr**2 + zi**2))

        for pr, pi in self.comp_cpole_pairs:
            C *= 1 / (z**2 - 2*pr*z + (pr**2 + pi**2))

        for _ in range(self.comp_origin_zeros):
            C *= z

        for _ in range(self.comp_delays):
            C *= 1 / z

        for _ in range(self.comp_integrators):
            C *= self.Ts / (z - 1)

        return C

    def _rebuild_compensated_system(self):
        C = self._build_compensator_tf()
        self.comp_sys = C
        self.sys = ctrl.minreal(C * self.base_sys, verbose=False)

    def _update_comp_input_visibility(self):
        kind = self.comp_type.currentText()

        need_imag = kind in ("Complex zero pair", "Complex pole pair")
        need_location = kind in (
            "Real zero",
            "Real pole",
            "Complex zero pair",
            "Complex pole pair"
        )

        self.comp_value.setVisible(need_location)
        self.comp_imag.setVisible(need_imag)

        form = getattr(self, "comp_form", None)
        if form is not None:
            try:
                form.labelForField(self.comp_value).setVisible(need_location)
                form.labelForField(self.comp_imag).setVisible(need_imag)
            except Exception:
                pass

        if kind in ("Real zero", "Real pole"):
            self.comp_value.setPlaceholderText("e.g. 0.6")
        elif kind in ("Complex zero pair", "Complex pole pair"):
            self.comp_value.setPlaceholderText("e.g. 0.7")
            self.comp_imag.setPlaceholderText("e.g. 0.2")

    def _update_comp_display(self):
        try:
            C = self._build_compensator_tf()
            txt = "C(z):\n"
            txt += self.parent().tf_to_pretty_str(C, var='z')

            if self.comp_zeros:
                txt += "\n\nReal zeros at:\n" + ", ".join([f"{zv:g}" for zv in self.comp_zeros])

            if self.comp_poles:
                txt += "\n\nReal poles at:\n" + ", ".join([f"{pv:g}" for pv in self.comp_poles])

            if self.comp_czero_pairs:
                txt += "\n\nComplex zero pairs at:\n"
                txt += "\n".join([f"{zr:g} ± j{zi:g}" for zr, zi in self.comp_czero_pairs])

            if self.comp_cpole_pairs:
                txt += "\n\nComplex pole pairs at:\n"
                txt += "\n".join([f"{pr:g} ± j{pi:g}" for pr, pi in self.comp_cpole_pairs])

            if self.comp_origin_zeros:
                txt += f"\n\nZeros at origin:\n{self.comp_origin_zeros} × z"

            if self.comp_delays:
                txt += f"\n\nSample delays:\n{self.comp_delays} × z^-1"

            if self.comp_integrators:
                txt += f"\n\nIntegrators:\n{self.comp_integrators} × {self.Ts:g}/(z-1)"

            self.comp_display.setText(txt)

        except Exception as e:
            self.comp_display.setText(f"Compensator error: {e}")

    def _add_compensator_element(self):
        try:
            kind = self.comp_type.currentText()

            if kind == "Real zero":
                txt = self.comp_value.text().strip()
                if not txt:
                    self.comp_display.setText("Enter the zero location in the z-plane.")
                    return
                self.comp_zeros.append(float(txt))
                self.comp_history.append(("real_zero", None))

            elif kind == "Real pole":
                txt = self.comp_value.text().strip()
                if not txt:
                    self.comp_display.setText("Enter the pole location in the z-plane.")
                    return
                self.comp_poles.append(float(txt))
                self.comp_history.append(("real_pole", None))

            elif kind == "Complex zero pair":
                txt_r = self.comp_value.text().strip()
                txt_i = self.comp_imag.text().strip()
                if not txt_r or not txt_i:
                    self.comp_display.setText("Enter real and imaginary parts.")
                    return
                zr = float(txt_r)
                zi = float(txt_i)
                if zi <= 0:
                    self.comp_display.setText("Imaginary part must be positive.")
                    return
                self.comp_czero_pairs.append((zr, zi))
                self.comp_history.append(("complex_zero_pair", None))

            elif kind == "Complex pole pair":
                txt_r = self.comp_value.text().strip()
                txt_i = self.comp_imag.text().strip()
                if not txt_r or not txt_i:
                    self.comp_display.setText("Enter real and imaginary parts.")
                    return
                pr = float(txt_r)
                pi = float(txt_i)
                if pi <= 0:
                    self.comp_display.setText("Imaginary part must be positive.")
                    return
                self.comp_cpole_pairs.append((pr, pi))
                self.comp_history.append(("complex_pole_pair", None))

            elif kind == "Zero at origin":
                self.comp_origin_zeros += 1
                self.comp_history.append(("origin_zero", None))

            elif kind == "Sample delay (z^-1)":
                self.comp_delays += 1
                self.comp_history.append(("delay", None))

            elif kind == "Integrator":
                self.comp_integrators += 1
                self.comp_history.append(("integrator", None))

            self.comp_value.clear()
            self.comp_imag.clear()

            self._rebuild_compensated_system()
            self._compute_root_locus_data()
            self._update_comp_display()
            self._view_lims = None
            self._base_drawn = False

            if self._klist is not None and len(self._klist) > 0:
                self._select_exact_k(self.selected_k)
            else:
                self._draw_root_locus(selected_poles=None)
                self._plot_step_for_K(0.0)
                self._set_info_idle()

        except Exception as e:
            self.comp_display.setText(f"Add error: {e}")

    def _undo_compensator_element(self):
        if not self.comp_history:
            return

        kind, _ = self.comp_history.pop()

        if kind == "real_zero" and self.comp_zeros:
            self.comp_zeros.pop()
        elif kind == "real_pole" and self.comp_poles:
            self.comp_poles.pop()
        elif kind == "complex_zero_pair" and self.comp_czero_pairs:
            self.comp_czero_pairs.pop()
        elif kind == "complex_pole_pair" and self.comp_cpole_pairs:
            self.comp_cpole_pairs.pop()
        elif kind == "origin_zero" and self.comp_origin_zeros > 0:
            self.comp_origin_zeros -= 1
        elif kind == "delay" and self.comp_delays > 0:
            self.comp_delays -= 1
        elif kind == "integrator" and self.comp_integrators > 0:
            self.comp_integrators -= 1

        self._rebuild_compensated_system()
        self._compute_root_locus_data()
        self._update_comp_display()
        self._view_lims = None
        self._base_drawn = False

        if self._klist is not None and len(self._klist) > 0:
            self._select_exact_k(self.selected_k)
        else:
            self._draw_root_locus(selected_poles=None)
            self._plot_step_for_K(0.0)
            self._set_info_idle()

    def _clear_compensator(self):
        self.comp_zeros = []
        self.comp_poles = []
        self.comp_czero_pairs = []
        self.comp_cpole_pairs = []
        self.comp_origin_zeros = 0
        self.comp_delays = 0
        self.comp_integrators = 0
        self.comp_history = []

        self.selected_k = 0.0
        self._rebuild_compensated_system()
        self._compute_root_locus_data()
        self._update_comp_display()
        self._view_lims = None
        self._base_drawn = False

        if self._klist is not None and len(self._klist) > 0:
            self._select_exact_k(0.0)
        else:
            self._draw_root_locus(selected_poles=None)
            self._plot_step_for_K(0.0)
            self._set_info_idle()

    # =========================================================
    # Root locus and selection
    # =========================================================
    def _compute_root_locus_data(self):
        rldata = ctrl.root_locus_map(self.sys)
        self._rlist = np.asarray(rldata.loci)
        self._klist = np.asarray(rldata.gains)

    def _draw_root_locus(self, selected_poles=None, redraw_base=False):
        redraw = bool(redraw_base or not self._base_drawn)

        for artist in self._sel_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._sel_artists = []

        if redraw:
            self._suppress_grid_events = True
            try:
                self.ax_rl.clear()
                self._grid_artists = []

                plot_root_locus_clean(self.sys, self.ax_rl)

                parent = self.parent()
                if hasattr(parent, "_set_zplane_limits"):
                    points = []
                    points.append(selected_poles)
                    try:
                        points.append(ctrl.poles(self.sys))
                        points.append(ctrl.zeros(self.sys))
                    except Exception:
                        pass
                    parent._set_zplane_limits(self.ax_rl, points, aspect="auto")
                else:
                    self.ax_rl.set_aspect('auto')

                self.ax_rl.set_title("")
                self.ax_rl.set_xlabel("")
                self.ax_rl.set_ylabel("")
                self._base_drawn = True

                # Keep the zoom level chosen by the user, if any.
                if self._view_lims is not None:
                    try:
                        self.ax_rl.set_xlim(*self._view_lims[0])
                        self.ax_rl.set_ylim(*self._view_lims[1])
                    except Exception:
                        pass
            finally:
                self._suppress_grid_events = False

            self._redraw_grid()

        if selected_poles is not None:
            for p in np.atleast_1d(selected_poles):
                p = complex(p)
                if np.isfinite(p.real) and np.isfinite(p.imag):
                    self._sel_artists.append(
                        self.ax_rl.plot(
                            [p.real], [p.imag],
                            marker="s",
                            markersize=7,
                            markerfacecolor="black",
                            markeredgecolor="black",
                            linestyle="None",
                            zorder=6
                        )[0]
                    )

        self.canvas.draw_idle()

    def _on_view_limits_changed(self, _ax=None):
        if getattr(self, "_suppress_grid_events", False):
            return
        self._grid_timer.start()

    def _redraw_grid(self):
        """Remove and redraw the z-grid for the current visible limits."""
        self._suppress_grid_events = True
        try:
            for artist in self._grid_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            self._grid_artists = []

            parent = self.parent()
            if hasattr(parent, "_overlay_zgrid"):
                self._grid_artists = list(
                    parent._overlay_zgrid(self.ax_rl, show_labels=True, Ts=self.Ts)
                )
        finally:
            self._suppress_grid_events = False
        self.canvas.draw_idle()

    def _select_point_from_k_index(self, ik):
        if self._rlist is None or self._klist is None:
            return
        if ik < 0 or ik >= len(self._klist):
            return
        self._select_exact_k(float(self._klist[ik]))

    def _select_exact_k(self, K):
        try:
            K = float(K)
        except (TypeError, ValueError):
            return
        if not np.isfinite(K) or K < 0:
            return
        if K < 1e-9:
            K = 0.0

        if K > 0:
            poles = closed_loop_poles(self.sys, K)
        else:
            poles = np.asarray(ctrl.poles(self.sys), dtype=complex).ravel()

        if len(poles) == 0:
            return

        mags = np.abs(poles)
        inside = np.where(mags < 1.0)[0]

        if len(inside) > 0:
            idx_main = inside[np.argmax(mags[inside])]
        else:
            idx_main = np.argmax(mags)

        pole_main = poles[idx_main]

        self.selected_k = K
        self.selected_pole = pole_main

        if self._klist is not None and len(self._klist) > 0:
            karr = np.asarray(self._klist, dtype=float).ravel()
            self.selected_index = int(np.argmin(np.abs(karr - K)))

        self._set_k_display(K)
        self._draw_root_locus(selected_poles=poles)
        self._plot_step_for_K(K)
        self._update_info_for_K(K, pole_main)

    def _k_slider_range(self):
        karr = np.asarray(self._klist if self._klist is not None else [], dtype=float).ravel()
        positive = karr[np.isfinite(karr) & (karr > 0)]
        if positive.size == 0:
            return 1e-3, 1.0
        kmax = float(np.max(positive))
        return kmax * 1e-6, kmax

    def _set_k_display(self, K):
        self.k_value.blockSignals(True)
        try:
            self.k_value.setText(f"{float(K):.6g}")
        except (TypeError, ValueError):
            self.k_value.setText("")
        self.k_value.blockSignals(False)

        kmin, kmax = self._k_slider_range()
        self.k_slider.blockSignals(True)
        self.k_slider.setValue(k_slider_pos(K, kmin, kmax))
        self.k_slider.blockSignals(False)

    def _on_k_slider_changed(self, value):
        kmin, kmax = self._k_slider_range()
        self._select_exact_k(k_from_slider_pos(value, kmin, kmax))

    def _set_k_from_value(self, K_user):
        self._select_exact_k(K_user)

    def _on_k_edit_finished(self):
        txt = self.k_value.text().strip()
        try:
            K_user = float(txt)
        except (TypeError, ValueError):
            self._set_k_display(self.selected_k)
            return

        if not np.isfinite(K_user) or K_user < 0:
            self._set_k_display(self.selected_k)
            return

        self._select_exact_k(K_user)

    def _on_click_rlocus(self, event):
        if getattr(self.toolbar, "mode", ""):
            return   # pan/zoom tool active: let the toolbar handle the click
        if getattr(event, "button", 1) != 1:
            return
        if event.inaxes != self.ax_rl:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._rlist is None or self._klist is None:
            return

        z_click = complex(event.xdata, event.ydata)

        seed = None
        try:
            diffs = np.abs(self._rlist - z_click)
            idx_flat = int(np.argmin(diffs))
            ik, _ = np.unravel_index(idx_flat, diffs.shape)
            seed = float(self._klist[ik])
        except Exception:
            seed = None

        try:
            K = locus_gain_from_point(self.sys, z_click, k_seed=seed)
        except Exception:
            return

        poles = closed_loop_poles(self.sys, K)
        if poles.size == 0:
            return
        pole = poles[int(np.argmin(np.abs(poles - z_click)))]

        try:
            x1, y1 = self.ax_rl.transData.transform((z_click.real, z_click.imag))
            x2, y2 = self.ax_rl.transData.transform((pole.real, pole.imag))
            if float(np.hypot(x1 - x2, y1 - y2)) > 30:
                return   # click is not close to any branch
        except Exception:
            pass

        self._select_exact_k(K)

    def _on_scroll_zoom(self, event):
        if event.inaxes != self.ax_rl:
            return
        if event.xdata is None or event.ydata is None:
            return
        button = getattr(event, "button", None)
        if button == "up":
            factor = 0.80
        elif button == "down":
            factor = 1.25
        else:
            return

        x0, x1 = self.ax_rl.get_xlim()
        y0, y1 = self.ax_rl.get_ylim()
        cx, cy = event.xdata, event.ydata
        self._suppress_grid_events = True
        try:
            self.ax_rl.set_xlim(cx + (x0 - cx) * factor, cx + (x1 - cx) * factor)
            self.ax_rl.set_ylim(cy + (y0 - cy) * factor, cy + (y1 - cy) * factor)
        finally:
            self._suppress_grid_events = False
        self._view_lims = (self.ax_rl.get_xlim(), self.ax_rl.get_ylim())
        self._redraw_grid()

    def get_state(self):
        """Snapshot of the design so it can be restored on the next open."""
        return {
            "comp_zeros": list(self.comp_zeros),
            "comp_poles": list(self.comp_poles),
            "comp_czero_pairs": [tuple(v) for v in self.comp_czero_pairs],
            "comp_cpole_pairs": [tuple(v) for v in self.comp_cpole_pairs],
            "comp_origin_zeros": int(self.comp_origin_zeros),
            "comp_delays": int(self.comp_delays),
            "comp_integrators": int(self.comp_integrators),
            "comp_history": list(self.comp_history),
            "selected_k": float(self.selected_k),
            "pwm": {
                "compare": self.pwm_compare.isChecked(),
                "reference": self.pwm_reference.value(),
                "mode": self.pwm_mode.currentData(),
                "deadzone": self.pwm_deadzone.value(),
                "bits": self.pwm_bits.value(),
                "plant_input": self.pwm_plant_input.currentData(),
            },
        }

    # =========================================================
    # Step response and info
    # =========================================================
    def _plot_step_for_K(self, K: float):
        self.ax_step.clear()

        L = K * self.sys
        T = ctrl.feedback(L, 1, sign=-1)

        try:
            # First pass
            N = 200
            t = np.arange(N, dtype=float) * self.Ts
            t, y = ctrl.step_response(T, T=t)

            # Refine x-axis using actual settling time
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    info = ctrl.step_info(T)
                ts = float(info.get("SettlingTime", np.nan))
                if np.isfinite(ts) and ts > 0:
                    t_final = max(8 * self.Ts, 2.5 * ts)
                    N = max(80, int(np.ceil(t_final / self.Ts)))
                    t = np.arange(N, dtype=float) * self.Ts
                    t, y = ctrl.step_response(T, T=t)
            except Exception:
                pass

            reference = self.pwm_reference.value()
            y = np.asarray(y) * reference
            self.parent()._plot_time_mpl(self.ax_step, t, y, label="Ideal (linear)")
            self.ax_step.axhline(reference, linestyle="--", linewidth=1)
            plotted = [y]
            self.pwm_status.setText("")
            self.last_pwm_simulation = None
            if self.pwm_compare.isChecked():
                try:
                    self.last_pwm_simulation = simulate_pwm_loop(
                        self.base_sys, K * self.comp_sys, np.full(len(t), reference),
                        mode=self.pwm_mode.currentData(), deadzone=self.pwm_deadzone.value(),
                        plant_input=self.pwm_plant_input.currentData(),
                        pwm_bits=self.pwm_bits.value(),
                    )
                    y_pwm = self.last_pwm_simulation["output"]
                    self.parent()._plot_time_mpl(self.ax_step, t, y_pwm, label="Integer PWM + saturation")
                    plotted.append(y_pwm)
                    self.pwm_status.setText(
                        f"{self.pwm_bits.value()}-bit PWM model: fixed Ts, continuous sensor, no RPM filter or friction model."
                    )
                except Exception as exc:
                    self.pwm_status.setText(f"PWM comparison unavailable: {exc}")
            self.ax_step.grid(True)
            self.ax_step.set_xlim(float(t[0]), float(t[-1]))
            self.ax_step.legend(loc="best")

            ymin = float(min(np.min(values) for values in plotted))
            ymax = float(max(np.max(values) for values in plotted))
            if abs(ymax - ymin) < 1e-9:
                ymin -= 0.05
                ymax += 0.05
            else:
                margin = 0.1 * (ymax - ymin)
                ymin -= margin
                ymax += margin
            self.ax_step.set_ylim(ymin, ymax)

        except Exception as e:
            self.ax_step.text(
                0.05, 0.5,
                f"step_response error: {e}",
                transform=self.ax_step.transAxes
            )

        self.ax_step.set_title("")
        self.ax_step.set_xlabel("")
        self.ax_step.set_ylabel("")
        self.canvas.draw_idle()

    def _set_info_idle(self):
        self.info.setText(
            "Discrete root locus + step response\n\n"
            "No gain selected yet (K = 0)."
        )

    def _poly_to_zinv_str(self, coeffs, var='z', digits=5, eps=1e-12):
        """
        Format [c0, c1, c2, ...] as:
            c0 + c1 z^-1 + c2 z^-2 + ...
        """
        coeffs = np.asarray(coeffs, dtype=float).ravel()
        terms = []

        for k, a in enumerate(coeffs):
            if abs(a) < eps:
                continue

            a_str = f"{a:.{digits}g}"

            if k == 0:
                terms.append(f"{a_str}")
            elif k == 1:
                if abs(a - 1.0) < eps:
                    terms.append(f"{var}^-1")
                elif abs(a + 1.0) < eps:
                    terms.append(f"-{var}^-1")
                else:
                    terms.append(f"{a_str} {var}^-1")
            else:
                if abs(a - 1.0) < eps:
                    terms.append(f"{var}^-{k}")
                elif abs(a + 1.0) < eps:
                    terms.append(f"-{var}^-{k}")
                else:
                    terms.append(f"{a_str} {var}^-{k}")

        if not terms:
            return "0"

        s = " + ".join(terms)
        return s.replace("+ -", "- ")

    def _implemented_controller_pretty_str(self, K):
        """
        Return K*C(z) in the implementation-friendly z^-1 form:
            (A + B z^-1 + ...)/(1 - E z^-1 - ...)
        """
        Ck = float(K) * self.comp_sys
        num, den = self._controller_to_zinv_coeffs(Ck)

        # normalize
        if abs(den[0]) > 1e-12:
            num = num / den[0]
            den = den / den[0]

        num_s = self._poly_to_zinv_str(num, var='z', digits=5)
        den_s = self._poly_to_zinv_str(den, var='z', digits=5)

        bar = "-" * max(len(num_s), len(den_s), 12)
        return f"{num_s}\n{bar}\n{den_s}"

    def _update_info_for_K(self, K: float, pole):
        L = K * self.sys
        T = ctrl.feedback(L, 1, sign=-1)

        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                si = ctrl.step_info(T)
            step_txt = "\n".join([f"{k}: {si[k]}" for k in si])
        except Exception as e:
            step_txt = f"step_info error: {e}"

        try:
            pole_mag = abs(pole)
            stable_txt = "Yes" if pole_mag < 1.0 else "No"
        except Exception:
            pole_mag = np.nan
            stable_txt = "Unknown"

        try:
            implemented_txt = self._implemented_controller_pretty_str(K)
        except Exception as e:
            implemented_txt = f"implementation form error: {e}"

        self.info.setText(
            f"Selected point\n"
            f"K = {K}\n"
            f"Pole = {pole}\n"
            f"|z| = {pole_mag}\n"
            f"Inside unit circle: {stable_txt}\n"
            f"Ts = {self.Ts} s\n\n"
            f"Controller to implement K*C(z) in z^-1 form:\n"
            f"{implemented_txt}\n\n"
            f"Closed-loop step_info(T):\n{step_txt}"
        )

    # =========================================================
    # Export
    # =========================================================
    def _controller_to_zinv_coeffs(self, sys):
        return controller_to_zinv(sys)

    def _close_and_export_to_digital_tab(self):
        try:
            K = float(self.selected_k) if self.selected_k is not None else 0.0
            Ck = K * self.comp_sys
            num, feedback = deployment_coefficients(Ck)
            period_ms = deployment_period_ms(self.Ts)
            p = self.parent()
            if num[4] != 0 and not p.gui_only and not p._supports_b4:
                raise ValueError("Firmware has not confirmed B4 support. Update/connect it before exporting I != 0.")
            # Nine significant digits round-trip every firmware float32 coefficient.
            for name, value in zip("ABCDI", num):
                getattr(p, name).setText(f"{value:.9g}")
            for name, value in zip("EFGH", feedback):
                getattr(p, name).setText(f"{value:.9g}")
            p.delay.setText(str(period_ms))

            # jump to "Discrete Contr"
            p.tabWidget.setCurrentWidget(p.Digital)

            p.toggleupdate_parameters()
            p._remember_controller_design("discrete", self)
            self.accept()

        except Exception as e:
            self.info.append(f"\nExport error: {e}")

class PIDTunerPlotDialog(QDialog):
    """
    Large interactive PID tuner window:
      - Left: large closed-loop step response
      - Right: controls + text info
    """

    def __init__(self, parent, G, ctype, speed_value, damping_value):
        super().__init__(parent)
        self.parent_dialog = parent
        self.G = G

        _load_designer_form(self, "ui/PIDTunerDialog.ui")

        flags = self.windowFlags()
        self.setWindowFlags(
            flags
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        # ---------- Figure ----------
        self.fig = self.canvas.figure
        self.fig.set_layout_engine("constrained")
        self.ax = self.fig.add_subplot(111)
        self.toolbar.bind_canvas(self.canvas)

        idx = self.controller_type_combo.findText(ctype.upper())
        if idx >= 0:
            self.controller_type_combo.setCurrentIndex(idx)
        self.speed_slider.setValue(speed_value)
        self.speed_value_label.setText(str(speed_value))
        self.damping_slider.setValue(damping_value)
        self.damping_value_label.setText(str(damping_value))
        _fit_designer_window(self)

        # ---------- Debounced update ----------
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._update_view)

        def _schedule():
            self._timer.start(120)

        self.controller_type_combo.currentIndexChanged.connect(_schedule)
        self.speed_slider.valueChanged.connect(_schedule)
        self.damping_slider.valueChanged.connect(_schedule)

        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_value_label.setText(str(v))
        )
        self.damping_slider.valueChanged.connect(
            lambda v: self.damping_value_label.setText(str(v))
        )
        self.export_btn.clicked.connect(self._close_and_export_to_pid_tab)
        self.current_kp = 0.0
        self.current_ki = 0.0
        self.current_kd = 0.0
        self.current_ctype = ctype
        self._update_view()

    def _close_and_export_to_pid_tab(self):
        if not self.export_btn.isEnabled():
            return
        try:
            p = self.parent_dialog

            # Export continuous gains to the deployment/PID tab
            p.Kp.setText(f"{self.current_kp:.6g}")
            p.Ki.setText(f"{self.current_ki:.6g}")
            p.Kd.setText(f"{self.current_kd:.6g}")

            p.tabWidget.setCurrentWidget(p.tab_3)

            # Push values through your existing flow
            p.toggleupdate_parameters()

            self.accept()

        except Exception as e:
            self.info.append(f"\nExport error: {e}")

    def _update_view(self):
        ctype = self.controller_type_combo.currentText().strip().upper()

        try:
            kp, ki, kd, C, target_ts, target_os, _ = self.parent_dialog._tune_pid_from_ui(
                self.G, ctype, self.speed_slider.value(), self.damping_slider.value())
            self.export_btn.setEnabled(True)

            self.current_kp = kp
            self.current_ki = ki
            self.current_kd = kd
            self.current_ctype = ctype

            self.kp_edit.setText(f"{kp:.6g}")
            self.ki_edit.setText(f"{ki:.6g}")
            self.kd_edit.setText(f"{kd:.6g}")

            T = ctrl.feedback(C * self.G, 1)

            # First pass
            t_final = max(0.12, 4.0 * target_ts)
            t = np.linspace(0, t_final, 800)
            t_y, y = ctrl.step_response(T, T=t)

            # Refine using actual settling time
            try:
                info = ctrl.step_info(T)
                ts_actual = float(info.get("SettlingTime", np.nan))
                if np.isfinite(ts_actual) and ts_actual > 0:
                    t_final = max(0.12, 2.5 * ts_actual)
                    t = np.linspace(0, t_final, 1200)
                    t_y, y = ctrl.step_response(T, T=t)
            except Exception:
                pass

            self.ax.clear()
            self.parent()._plot_time_mpl(self.ax, t_y, y, label="y(t)")
            self.ax.set_xlim(float(t_y[0]), float(t_y[-1]))
            self.ax.axhline(1.0, linestyle="--", linewidth=1)
            self.ax.grid(True)
            self.ax.set_title("Closed-loop step response")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude")
            self.ax.legend(loc="best")
            self.canvas.draw_idle()

            try:
                si = ctrl.step_info(T)
                step_txt = "\n".join([f"{k}: {si[k]}" for k in si])
            except Exception as e:
                step_txt = f"step_info error: {e}"

            try:
                bw = ctrl.bandwidth(T)
                bw_txt = f"{bw} rad/s"
            except Exception as e:
                bw_txt = f"bandwidth error: {e}"

            txt = (
                f"Controller type: {ctype}\n\n"
                f"Kp = {kp}\n"
                f"Ki = {ki}\n"
                f"Kd = {kd}\n\n"

                f"Target overshoot: {target_os:.4g} %\n\n"
                f"Closed-loop step_info(T):\n{step_txt}\n\n"
                f"Closed-loop bandwidth(T) (-3 dB): {bw_txt}"
            )

            self.info.setText(txt)

        except Exception as e:
            self.export_btn.setEnabled(False)
            self.current_kp = self.current_ki = self.current_kd = 0.0
            for field in (self.kp_edit, self.ki_edit, self.kd_edit):
                field.clear()
            self.ax.clear()
            self.ax.text(
                0.05, 0.5,
                f"PID tuner error: {e}",
                transform=self.ax.transAxes
            )
            self.ax.grid(True)
            self.canvas.draw_idle()
            self.info.setText(f"PID tuner error: {e}")

class PortSelectDialog(QDialog):
    def __init__(self, initial_port=None, parent=None):
        super().__init__(parent)
        _load_designer_form(self, "ui/PortSelectDialog.ui")
        self.initial_port = initial_port
        self._ports = []
        self.gui_only = False
        # Signals
        self.btn_refresh.clicked.connect(self.populate)
        self.btn_gui_only.clicked.connect(self._choose_gui_only)
        self.buttons.accepted.connect(self.validate_and_accept)
        self.buttons.rejected.connect(self.reject)

        # Initial population
        self.populate()

    def populate(self):
        self.combo.clear()
        self._ports = list(serial.tools.list_ports.comports())
        for p in self._ports:
            self.combo.addItem(f"{p.device} — {p.description}", userData=p.device)
        # preselect previous choice if still present
        if self.initial_port is not None:
            idx = self.combo.findData(self.initial_port)
            if idx != -1:
                self.combo.setCurrentIndex(idx)

        if self._ports:
            self.label.setText("Choose the serial (COM) port to use:")
        else:
            self.label.setText(
                "No serial ports detected. Connect the Teensy and press Refresh, "
                "or choose 'GUI only' to work without hardware."
            )

    def _choose_gui_only(self):
        self.gui_only = True
        self.accept()

    def selected_port(self):
        if self.gui_only:
            return None
        idx = self.combo.currentIndex()
        if idx < 0:
            return None
        return self.combo.itemData(idx)

    def validate_and_accept(self):
        port = self.selected_port()
        if not port:
            QMessageBox.warning(
                self, "No port selected",
                "Please select a COM port, choose 'GUI only' to work without hardware, "
                "or click Cancel to exit.",
            )
            return
        self.gui_only = False
        self.accept()

class MyDialog(QtWidgets.QDialog):
    def __init__(self, port=None, gui_only=False, parent=None):
        super(MyDialog, self).__init__(parent)
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "QtDesignerGUI.ui")
        uic.loadUi(ui_path, self)  # load the UI first
        self._controller_designs = {}
        self._design_states = {}
        self.continuous_num_fields = [getattr(self, f"continuous_num{i}") for i in range(4, -1, -1)]
        self.continuous_den_fields = [getattr(self, f"continuous_den{i}") for i in range(4, -1, -1)]
        self.discrete_back_to_design.clicked.connect(lambda: self._return_to_controller_design("discrete"))
        self.continuous_back_to_design.clicked.connect(lambda: self._return_to_controller_design("continuous"))
        self._line_edit_config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            LINE_EDIT_CONFIG_FILENAME,
        )
        self._line_edit_config_loading = False
        self._line_edit_default_values = {
            "A": "0",
            "B": "0",
            "C": "0",
            "D": "0",
            "E": "0",
            "F": "0",
            "G": "0",
            "H": "0",
            "I": "0",
            "deadzone": "0",
            "offset": "0",
            "tiemporeferencia": "2000",
            "amplitude": "150",
            "reference": "150",
            "delay": "20",
            "Kp": "0",
            "Ki": "0",
            "Kd": "0",
            "denorder": "1",
            "numorder": "0",
            "datapoints": "200",
            "x_scale": "5",
            "time_constant": "0.2",
            "reset_time": "0.5",
            "sampling_time": "0.02",
            "lqr_sampling": "0.02",
            "PPR": "240",
            "PWM_min": "0",
            "PWM_max": "255",
            "RPM_filter": "1",
        }
        self._persistent_line_edit_names = self._collect_persistent_line_edit_names()
        self._load_line_edit_config()
        self._runtime_config_values = self._default_runtime_config_values()
        # ---- make the dialog look/behave like a normal resizable window ----
        flags = self.windowFlags()
        flags |= Qt.WindowType.WindowSystemMenuHint          # show system menu
        flags |= Qt.WindowType.WindowMinMaxButtonsHint       # add Min/Max buttons
        flags &= ~Qt.WindowType.WindowContextHelpButtonHint  # remove the "?" button
        self.setWindowFlags(flags)

        # allow resizing even if the .ui layout had "SetFixedSize"
        if self.layout() is not None:
            self.layout().setSizeConstraint(
                QtWidgets.QLayout.SizeConstraint.SetDefaultConstraint
            )

        # optional but useful
        _fit_designer_window(self)

        base_title = self.windowTitle()
        self._base_window_title = base_title
        self.gui_only = bool(gui_only)
        if self.gui_only:
            self.setWindowTitle(f"{base_title} — GUI only")
        elif port:
            self.setWindowTitle(f"{base_title} — {port}")
        else:
            self.setWindowTitle(f"{base_title} — no port")
        self.serial_port = None
        self.serial_port_name = port
        self.controller_connected = False
        self._supports_b4 = False
        self.firmware_pwm_bits = None
        self._last_reconnect_attempt = 0.0
        self._reconnect_interval_s = 1.0
        self._serial_status_message = ""
        if port and not self.gui_only:
            try:
                self.serial_port = serial.Serial(port, 115200, timeout=0.05, write_timeout=0.05)
                self.serial_port.write(b"CAPS?\n")
                self.controller_connected = True
            except Exception as e:
                QMessageBox.critical(self, "Serial error",
                                     f"Could not open port {port}:\n{e}")
                self.serial_port = None

        if os.environ.get("OPENMCT_DEBUG_TABS"):
            print("=== tabWidget order ===")
            for i in range(self.tabWidget.count()):
                print(i, self.tabWidget.tabText(i))

        # Initialize data lists
        #self.header_written = False
        datapoints = self._line_edit_int(self.datapoints, default=200, minimum=1)
        self.dataRPM_setpoint = deque(maxlen=datapoints)
        self.dataRPM_measured = deque(maxlen=datapoints)
        self.dataPWM = deque(maxlen=datapoints)
        self.dataDT = deque(maxlen=datapoints)   # ms per sample
        self.dataCurrent = deque(maxlen=datapoints)  # calibrated, filtered mA; NaN if unavailable
        self.tabWidget.currentChanged.connect(self.tabChanged)

        self.reference.textChanged.connect(self.update_slider_from_line_edit)
        self.slider.valueChanged.connect(self.update_line_edit_from_slider)

        # Define mode_map at the class level
        self.mode_map = {
            "Disabled": "0",
            "System Identification": "1",
            "Speed Control": "2",
            "Position Control": "3",
            "Speed - manual": "2",
            "Position - manual": "3",
        }

        self._sync_sampling_lineedits()

        # valor inicial compartido
        if self.sampling_time.text().strip() and not self.lqr_sampling.text().strip():
            self.lqr_sampling.setText(self.sampling_time.text())
        elif self.lqr_sampling.text().strip() and not self.sampling_time.text().strip():
            self.sampling_time.setText(self.lqr_sampling.text())


        self.datapoints.textChanged.connect(self.resize_deque)
        self.modooperacion.currentIndexChanged.connect(self.sendModeOperation)
        self.grupo_checkboxes = QButtonGroup(self)
        self.grupo_checkboxes.addButton(self.manualinput, 1)
        self.grupo_checkboxes.addButton(self.automaticinput, 0)
        self.grupo_checkboxes.setExclusive(True)
        self.automaticinput.setChecked(True)
        self.manualinput.toggled.connect(self._on_reference_source_changed)
        self.automaticinput.toggled.connect(self._on_reference_source_changed)

        self.saveValuesCheckBox.toggled.connect(self._on_save_values_toggled)
        self._dmm_thread = None
        self._dmm_worker = None
        self._dmm_stop_requested = False
        self._dmm_latest_current_a = None
        self._dmm_latest_time = None
        self._dmm_latest_sample_id = -1
        self._recording_include_dmm = False
        self._recording_start_monotonic = None
        self._dmmout_widget = self.findChild(QtWidgets.QLineEdit, "DMMout")
        if self._dmmout_widget is not None:
            self._dmmout_widget.setText("DMM off")
        if hasattr(self, "DMMSync"):
            self.DMMSync.toggled.connect(self._on_dmm_sync_toggled)

        self._test_duration_timer = QTimer(self)
        self._test_duration_timer.setSingleShot(True)
        self._test_duration_timer.timeout.connect(self._on_test_duration_elapsed)
        if hasattr(self, "time_delimited") and hasattr(self, "test_time"):
            self.time_delimited.toggled.connect(self._on_time_delimited_toggled)
            self._on_time_delimited_toggled(self.time_delimited.isChecked())


        # Define el botón de start/stop
        self.StartStop.clicked.connect(self.toggleStartStop)
        self.isRunning = False

        # Realiza la discretización
        self.discretize.clicked.connect(self.discretize_function)
        self.method.currentTextChanged.connect(self._update_prewarp_controls)
        self.prewarp_tustin.toggled.connect(self._update_prewarp_controls)
        self._update_prewarp_controls()

        # Define botón de update parameters
        self.update_parameters.clicked.connect(self._on_update_parameters_clicked)
        # Define botón de identificación de sistema
        self.identify.clicked.connect(self.identify_system)
        # Define botón de simulación
        self.simulation.clicked.connect(self.Simulate)
        # Define botón de cálculo tf equivalente
        self.reduce.clicked.connect(self.reduceTF)
        self.identified_sys_tf = None
        self.tuner_loaded_plant = None
        self.run_button.clicked.connect(self.run_control_command)
        self.clear_button.clicked.connect(self.clear_control_console)

        self._setup_analysis_plots()
        self._setup_discrete_rlocus_tab_plots()
        self._setup_pid_tuner()
        # Open big plots when user clicks the thumbnail plots
        self.canvas_rl.mpl_connect("button_press_event", self._on_rlocus_canvas_click)

        # Plot widgets are instantiated by uic from the Designer form.
        pi = self.graphWidgetRPM.getPlotItem()
        self.legendRPM = pi.addLegend()
        self.legendRPM.setParentItem(pi.vb)
        self.legendRPM.anchor(itemPos=(1,0), parentPos=(1,0), offset=(10, -10))
        pi.setLabel('left', 'Motor Input / Output', units='PWM / RPM')
        self.graphWidgetRPM.setYRange(0, 120)

        pi = self.graphWidgetPWM.getPlotItem()
        self.legendPWM = pi.addLegend()
        self.legendPWM.setParentItem(pi.vb)
        self.legendPWM.anchor(itemPos=(1,0), parentPos=(1,0), offset=(10, -10))
        pi.setLabel('left', 'uController output', units='PWM')
        pi.setLabel('bottom', 'Time', units='s')
        self.graphWidgetPWM.setYRange(0, 255)
        self.graphWidgetforced_response.setYRange(0, 1.2)
        self.graphWidgetsim_response.setYRange(0, 1.2)

        # Initialize curves for fast updates
        self.graphWidgetPWM.setXLink(self.graphWidgetRPM)
        self.curve_setpoint = self._plot_time_pg(self.graphWidgetRPM, name="Input", pen=pg.mkPen(color='b', width=3, style=QtCore.Qt.PenStyle.SolidLine))  # Blue line for setpoint
        self.curve_measured = self._plot_time_pg(self.graphWidgetRPM, name="Output", pen=pg.mkPen(color='r', width=3, style=QtCore.Qt.PenStyle.SolidLine))  # Red line for measured
        self.curve_pwm = self._plot_time_pg(self.graphWidgetPWM, name="PWM", pen=pg.mkPen(color='g', width=3, style=QtCore.Qt.PenStyle.SolidLine))       # Green line for PWM
        self._setup_telemetry_overlays()
        self.modo_plot.currentTextChanged.connect(self._on_plot_mode_changed)
        self._set_runtime_plot_interaction(enabled=True)

        # Disconnect the standard dialog accept/reject slots
        # Esto se hace para quitarle los valores por default que tienen los botones de
        # Ok y cancel (que es cerrar la ventana)

        # Setup the QTimer
        # Este es el timer para la rapidez de update de la hora
        self.timerHora = QTimer(self)
        self.timerHora.setInterval(1000)
        self.timerHora.timeout.connect(self.updateDateTime)

        # Este es el timer para la rapidez de update de la gráfica
        self.timer = QTimer(self)
        self.timer.setInterval(10)  # Adjust the interval as needed
        self.timer.timeout.connect(self.update_graph)
        self.pid_tuner_dialog_open = False

        # Keep line edits visually empty on a fresh install; runtime defaults are
        # read from _line_edit_default_values when a field is blank.
        self.modooperacion.setCurrentIndex(0)
        self.tabWidget.blockSignals(True)
        self.tabWidget.setCurrentWidget(self.tab)
        self.tabWidget.blockSignals(False)
        self._sync_reference_slider_to_line_edit()
        self._update_reference_controls_visibility()
        self._connect_line_edit_config_saves()
        self._connect_continuous_controller_signals()

        if self.gui_only:
            self.textBrowser.setText("GUI only — no hardware connected, commands are not sent")

        # Comienza todos los timers
        self.timer.start()
        self.timerHora.start()

    def _show_controller_disconnected(self, reason=None):
        self._supports_b4 = False
        self.firmware_pwm_bits = None
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
        self.serial_port = None
        self.controller_connected = False
        self._serial_status_message = "Controlador desconectado, esperando reconexión"
        self.textBrowser.setText(self._serial_status_message)
        if reason:
            print(f"Serial disconnected: {reason}")

    def _show_controller_connected(self):
        was_disconnected = not self.controller_connected
        self.controller_connected = True
        self._serial_status_message = ""
        self.textBrowser.setText("")
        if was_disconnected:
            self.toggleupdate_parameters()

    def _try_reconnect_serial(self):
        if getattr(self, "gui_only", False):
            return False
        now = monotonic()
        if now - self._last_reconnect_attempt < self._reconnect_interval_s:
            return False
        self._last_reconnect_attempt = now

        ports_to_try = []
        if self.serial_port_name:
            ports_to_try.append(self.serial_port_name)
        detected_port = auto_find_port()
        if detected_port and detected_port not in ports_to_try:
            ports_to_try.append(detected_port)

        if not ports_to_try:
            self.textBrowser.setText("Controlador desconectado, esperando reconexión")
            return False

        for port in ports_to_try:
            try:
                self.serial_port = serial.Serial(port, 115200, timeout=0.05, write_timeout=0.05)
                self.serial_port_name = port
                self.serial_port.reset_input_buffer()
                self._supports_b4 = False
                self.firmware_pwm_bits = None
                self.serial_port.write(b"CAPS?\n")
                self.setWindowTitle(f"{self._base_window_title} — {port}")
                self._show_controller_connected()
                return True
            except Exception:
                self.serial_port = None

        self.textBrowser.setText("Controlador desconectado, esperando reconexión")
        return False


    def _connect_continuous_controller_signals(self):
        for field in self.continuous_num_fields + self.continuous_den_fields:
            field.textChanged.connect(self._refresh_continuous_controller)
        self.delay.textChanged.connect(self._refresh_continuous_controller)
        self._refresh_continuous_controller()

    def _read_continuous_controller(self):
        def polynomial(fields):
            try:
                values = [float((self._coefficient_text(field).strip() or "0").replace(",", "."))
                          for field in fields]
            except ValueError:
                raise ValueError("Enter numeric controller coefficients.") from None
            if not np.all(np.isfinite(values)):
                raise ValueError("Controller coefficients must be finite.")
            while len(values) > 1 and values[0] == 0:
                values.pop(0)
            return values
        num = polynomial(self.continuous_num_fields)
        den = polynomial(self.continuous_den_fields)
        if not any(den):
            raise ValueError("Enter a nonzero controller denominator.")
        controller = ctrl.tf(num, den)
        continuous_controller_coefficients(controller)
        return controller

    def _continuous_controller_period(self):
        try:
            milliseconds = float(str(self.delay.text().strip() or self._line_edit_default(self.delay)).replace(",", "."))
        except (ValueError, TypeError):
            raise ValueError("Enter a sampling time in whole milliseconds.") from None
        period = milliseconds / 1000
        deployment_period_ms(period)
        return period

    def _prepare_continuous_controller(self, controller):
        period = self._continuous_controller_period()
        discrete = discretize_continuous_controller(controller, period)
        numerator, feedback = deployment_coefficients(discrete)
        if numerator[4] != 0 and not self.gui_only and not self._supports_b4:
            raise ValueError("Firmware has not confirmed B4 support. Update/connect it before using this controller.")
        return numerator, feedback

    def _refresh_continuous_controller(self, *_):
        self.continuous_period.setText(self.delay.text().strip() or str(self._line_edit_default(self.delay)))
        if not any(self._coefficient_text(field).strip()
                   for field in self.continuous_num_fields + self.continuous_den_fields):
            self.continuous_display.clear()
            return
        try:
            controller = self._read_continuous_controller()
            self.continuous_display.setPlainText(self.tf_to_pretty_str(controller, decimals=2))
        except (ValueError, TypeError, OverflowError) as exc:
            self.continuous_display.setPlainText(str(exc))

    def _import_continuous_controller(self, controller):
        num, den = continuous_controller_coefficients(controller)
        self._prepare_continuous_controller(controller)  # validate before changing any fields
        was_loading = self._line_edit_config_loading
        self._line_edit_config_loading = True
        try:
            for fields, values in ((self.continuous_num_fields, num), (self.continuous_den_fields, den)):
                for field, value in zip(fields, np.pad(values, (5 - len(values), 0))):
                    with QtCore.QSignalBlocker(field):
                        field.setText(f"{value:.17g}")
            with QtCore.QSignalBlocker(self.tabWidget):
                self.tabWidget.setCurrentWidget(self.Continuous)
        finally:
            self._line_edit_config_loading = was_loading
        self._refresh_continuous_controller()
        self._save_line_edit_config()
        return self.toggleupdate_parameters()

    def _update_design_return_buttons(self):
        for kind, button in (("continuous", self.continuous_back_to_design),
                             ("discrete", self.discrete_back_to_design)):
            available = kind in self._controller_designs
            button.setEnabled(available)
            button.setToolTip("Return to this controller's exported design, including its plant and gain."
                              if available else "Export a controller from a root locus design first.")

    def _remember_controller_design(self, kind, designer):
        num, den = tf_num_den(designer.base_sys)
        period = float(designer.Ts) if kind == "discrete" else None
        self._controller_designs[kind] = {
            "plant_num": num.tolist(), "plant_den": den.tolist(),
            "period": period, "state": deepcopy(designer.get_state()),
        }
        extra = (round(period, 12),) if period is not None else ()
        key = design_state_key(kind, designer.base_sys, extra=extra)
        self._design_states[key] = deepcopy(self._controller_designs[kind]["state"])
        self._update_design_return_buttons()
        self._save_line_edit_config()

    def _return_to_controller_design(self, kind):
        saved = self._controller_designs.get(kind)
        if saved is None:
            return
        dialog = None
        try:
            period = float(saved["period"]) if kind == "discrete" else None
            plant = ctrl.tf(saved["plant_num"], saved["plant_den"], period if period is not None else 0)
            num, den = tf_num_den(plant)
            if len(num) > 4 or len(den) > 4:
                raise ValueError("The source plant exceeds the design page's polynomial order.")
            if kind == "continuous":
                prefix, page = "rlocus", self.tab_6
            else:
                prefix, page = "rlocusz", self.tab_8
            was_loading = self._line_edit_config_loading
            self._line_edit_config_loading = True
            try:
                for suffix, values in (("num", num), ("den", den)):
                    for power, value in zip(range(3, -1, -1), np.pad(values, (4 - len(values), 0))):
                        field = getattr(self, f"{prefix}_{suffix}{power}")
                        with QtCore.QSignalBlocker(field):
                            field.setText(f"{value:.17g}")
                if period is not None:
                    for field in (self.lqr_sampling, self.sampling_time):
                        with QtCore.QSignalBlocker(field):
                            field.setText(f"{period:.17g}")
                # Navigation must not send a new active-tab command to the motor.
                with QtCore.QSignalBlocker(self.tabWidget):
                    self.tabWidget.setCurrentWidget(page)
            finally:
                self._line_edit_config_loading = was_loading
            if kind == "continuous":
                self.update_analysis_plots()
                dialog = ControlPlotDialog(self, plant, state=deepcopy(saved["state"]))
            else:
                self.update_discrete_rlocus_plots()
                dialog = DiscreteControlPlotDialog(self, plant, period, state=deepcopy(saved["state"]))
            dialog.exec()
            extra = (round(period, 12),) if period is not None else ()
            self._design_states[design_state_key(kind, plant, extra=extra)] = dialog.get_state()
            self._save_line_edit_config()
        except (ValueError, TypeError, KeyError) as exc:
            self.textBrowser.setText(f"Could not restore controller design: {exc}")
        finally:
            if dialog is not None:
                dialog.deleteLater()


    def _collect_persistent_line_edit_names(self):
        excluded = {
            "DMMout",
            "serial_in",
            "serial_out",
            "rmse",
            "mae",
            "r2",
            "nrmse",
            "tuner_kp",
            "tuner_ki",
            "tuner_kd",
        }
        names = []
        for line_edit in self.findChildren(QtWidgets.QLineEdit):
            name = line_edit.objectName()
            if name and name not in excluded and not line_edit.isReadOnly():
                names.append(name)
        return tuple(sorted(set(names)))

    def _load_line_edit_config(self):
        if not os.path.exists(self._line_edit_config_path):
            return

        try:
            with open(self._line_edit_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Could not load GUI config: {e}")
            return

        values = data.get("line_edits", {})
        if not isinstance(values, dict):
            return

        self._line_edit_config_loading = True
        try:
            for name in self._persistent_line_edit_names:
                if name not in values:
                    continue
                widget = getattr(self, name, None)
                if isinstance(widget, QtWidgets.QLineEdit):
                    value = values[name]
                    widget.setText("" if value is None else str(value))
            designs = data.get("controller_designs", {})
            if isinstance(designs, dict):
                for kind in ("continuous", "discrete"):
                    saved = designs.get(kind)
                    if (isinstance(saved, dict) and isinstance(saved.get("state"), dict)
                            and "plant_num" in saved and "plant_den" in saved
                            and (kind == "continuous" or saved.get("period") is not None)):
                        self._controller_designs[kind] = deepcopy(saved)
            self._update_design_return_buttons()
        finally:
            self._line_edit_config_loading = False

    def _connect_line_edit_config_saves(self):
        for name in self._persistent_line_edit_names:
            widget = getattr(self, name, None)
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.textChanged.connect(self._save_line_edit_config)

    def _save_line_edit_config(self, *_args):
        if self._line_edit_config_loading:
            return

        values = {}
        for name in self._persistent_line_edit_names:
            widget = getattr(self, name, None)
            if isinstance(widget, QtWidgets.QLineEdit):
                values[name] = self._coefficient_text(widget)

        data = {
            "version": 1,
            "line_edits": values,
            "controller_designs": self._controller_designs,
        }

        tmp_path = f"{self._line_edit_config_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_path, self._line_edit_config_path)
        except Exception as e:
            print(f"Could not save GUI config: {e}")

    @staticmethod
    def _coefficient_text(widget):
        return widget.rawText() if isinstance(widget, CoefficientEdit) else widget.text()

    def _line_edit_default(self, widget, fallback=0.0):
        name = widget.objectName() if hasattr(widget, "objectName") else ""
        return self._line_edit_default_values.get(name, fallback)

    def _line_edit_number(self, widget, default=None, minimum=None, maximum=None, integer=False):
        if default is None:
            default = self._line_edit_default(widget, 0.0)

        text = self._coefficient_text(widget).strip() if hasattr(widget, "text") else str(widget).strip()
        raw_value = text if text else default
        try:
            value = float(str(raw_value).replace(",", "."))
        except (TypeError, ValueError):
            value = float(default)

        if minimum is not None:
            value = max(float(minimum), value)
        if maximum is not None:
            value = min(float(maximum), value)

        if integer:
            return int(round(value))
        return value

    def _line_edit_int(self, widget, default=None, minimum=None, maximum=None):
        return self._line_edit_number(
            widget,
            default=default,
            minimum=minimum,
            maximum=maximum,
            integer=True,
        )

    def _line_edit_float(self, widget, default=None, minimum=None, maximum=None):
        return self._line_edit_number(
            widget,
            default=default,
            minimum=minimum,
            maximum=maximum,
            integer=False,
        )

    def _sync_reference_slider_to_line_edit(self):
        value = self._line_edit_int(self.reference, default=150)
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)

    def _default_runtime_config_values(self):
        return {
            "PPR": 240,
            "PWM_min": 0,
            "PWM_max": 255,
            "RPM_filter": 1,
        }

    def _read_runtime_config_values_from_ui(self):
        ppr = self._line_edit_int(self.PPR, default=240, minimum=1)
        pwm_min = self._line_edit_int(self.PWM_min, default=0, minimum=0, maximum=255)
        pwm_max = self._line_edit_int(self.PWM_max, default=255, minimum=0, maximum=255)
        if pwm_max < pwm_min:
            pwm_min, pwm_max = pwm_max, pwm_min

        rpm_filter = self._line_edit_int(
            self.RPM_filter,
            default=1,
            minimum=1,
            maximum=64,
        )

        return {
            "PPR": ppr,
            "PWM_min": pwm_min,
            "PWM_max": pwm_max,
            "RPM_filter": rpm_filter,
        }

    def _on_update_parameters_clicked(self, checked=False):
        self.toggleupdate_parameters(update_runtime_config=True)

    def _reset_recording_state(self):
        self.header_written = False
        self._recording_include_dmm = False
        self._recording_start_monotonic = None

    def _on_save_values_toggled(self, checked):
        self._reset_recording_state()

    def _on_time_delimited_toggled(self, checked):
        if hasattr(self, "test_time"):
            self.test_time.setEnabled(checked)
        if hasattr(self, "test_time_label"):
            self.test_time_label.setEnabled(checked)
        if not checked:
            self._test_duration_timer.stop()

    def _test_duration_ms(self):
        text = self.test_time.text().strip().replace(",", ".")
        try:
            duration_s = float(text)
        except ValueError:
            duration_s = float("nan")

        max_duration_s = (2**31 - 1) / 1000.0
        if not np.isfinite(duration_s) or duration_s <= 0 or duration_s > max_duration_s:
            QMessageBox.warning(
                self,
                "Invalid test time",
                "Test time must be a positive number of seconds.",
            )
            self.test_time.setFocus()
            return None

        return max(1, int(round(duration_s * 1000.0)))

    def _on_test_duration_elapsed(self):
        if self.isRunning:
            self.isRunning = False
            self.StartStop.setText("START")
            self.StopAction()
        if self.saveValuesCheckBox.isChecked():
            self.saveValuesCheckBox.setChecked(False)

    def _set_dmm_output(self, text):
        if getattr(self, "_dmmout_widget", None) is not None:
            self._dmmout_widget.setText(text)

    def _dmm_requested(self):
        return bool(hasattr(self, "DMMSync") and self.DMMSync.isChecked())

    def _on_dmm_sync_toggled(self, checked):
        if checked:
            self._start_dmm_worker()
        else:
            self._stop_dmm_worker(wait=False)
            self._set_dmm_output("DMM off")

    def _start_dmm_worker(self):
        if self._dmm_thread is not None:
            return

        self._dmm_latest_current_a = None
        self._dmm_latest_time = None
        self._dmm_latest_sample_id = -1
        self._dmm_stop_requested = False
        self._set_dmm_output("Connecting DMM...")

        resource_name = os.environ.get("OPENMCT_DMM_RESOURCE")
        self._dmm_thread = QtCore.QThread(self)
        self._dmm_worker = SiglentDMMWorker(resource_name=resource_name)
        self._dmm_worker.moveToThread(self._dmm_thread)

        self._dmm_thread.started.connect(self._dmm_worker.run)
        self._dmm_worker.sample_ready.connect(self._on_dmm_sample_ready)
        self._dmm_worker.status_changed.connect(self._on_dmm_status_changed)
        self._dmm_worker.error.connect(self._on_dmm_error)
        self._dmm_worker.finished.connect(self._dmm_thread.quit)
        self._dmm_worker.finished.connect(self._dmm_worker.deleteLater)
        self._dmm_thread.finished.connect(self._on_dmm_thread_finished)
        self._dmm_thread.finished.connect(self._dmm_thread.deleteLater)
        self._dmm_thread.start()

    def _stop_dmm_worker(self, wait=False):
        self._dmm_stop_requested = True
        if self._dmm_worker is not None:
            self._dmm_worker.stop()
        if wait and self._dmm_thread is not None:
            if not self._dmm_thread.wait(2500):
                print("DMM worker did not stop before timeout.")

    def _on_dmm_thread_finished(self):
        self._dmm_thread = None
        self._dmm_worker = None
        if not self._dmm_stop_requested and hasattr(self, "DMMSync"):
            self.DMMSync.blockSignals(True)
            self.DMMSync.setChecked(False)
            self.DMMSync.blockSignals(False)

    def _on_dmm_status_changed(self, message):
        self._set_dmm_output(message)

    def _on_dmm_error(self, message):
        self._set_dmm_output(f"DMM error: {message}")

    def _on_dmm_sample_ready(self, current_a, sample_time, sample_id):
        self._dmm_latest_current_a = current_a
        self._dmm_latest_time = sample_time
        self._dmm_latest_sample_id = sample_id
        self._set_dmm_output(f"{current_a:.9g} A")

    @staticmethod
    def _dmm_columns_header():
        return ",DMM_CURRENT_A,DMM_TIME_s,DMM_AGE_ms,DMM_SAMPLE_ID"

    def _dmm_row_suffix(self):
        if not self._recording_include_dmm:
            return ""
        if (
            self._recording_start_monotonic is None
            or not self._dmm_requested()
            or self._dmm_latest_current_a is None
            or self._dmm_latest_time is None
            or self._dmm_latest_time < self._recording_start_monotonic
        ):
            return ",nan,nan,nan,-1"

        dmm_time_s = self._dmm_latest_time - self._recording_start_monotonic
        dmm_age_ms = (monotonic() - self._dmm_latest_time) * 1000.0
        return (
            f",{self._dmm_latest_current_a:.9g}"
            f",{dmm_time_s:.6f}"
            f",{dmm_age_ms:.1f}"
            f",{self._dmm_latest_sample_id}"
        )

    def _runtime_plot_widgets(self):
        return (self.graphWidgetRPM, self.graphWidgetPWM)

    def _time_plot_is_zoh(self):
        return self.modo_plot.currentText().strip().upper() == "ZOH"

    def _pg_step_mode(self):
        # In PyQtGraph 0.13, 'right' holds y[k] through x[k+1], like stairs.
        return "right" if self._time_plot_is_zoh() else None

    def _mpl_drawstyle(self):
        return "steps-post" if self._time_plot_is_zoh() else "default"

    def _plot_time_pg(self, widget, *args, **kwargs):
        return widget.plot(*args, stepMode=self._pg_step_mode(), **kwargs)

    def _plot_time_mpl(self, axes, *args, **kwargs):
        lines = axes.plot(*args, drawstyle=self._mpl_drawstyle(), **kwargs)
        for line in lines:
            line.set_gid("time-series")
        return lines

    def _setup_telemetry_overlays(self):
        self.curve_rpm_average = self._plot_time_pg(
            self.graphWidgetRPM, pen=pg.mkPen('y', width=3), connect="finite")
        self.curve_rpm_average.setZValue(1)
        self.curve_measured.setZValue(2)
        self.curve_rpm_average.hide()
        self._rpm_average_is_visible = False

        # Keep the two time axes aligned even when their Y tick labels have
        # different lengths. The reservation scales with the existing UI font.
        tick_font = self.graphWidgetRPM.getPlotItem().getAxis('left').style['tickFont'] or QtWidgets.QApplication.font()
        tick_width = int(np.ceil(QtGui.QFontMetricsF(tick_font).horizontalAdvance('-99999')))
        for plot in self._runtime_plot_widgets():
            plot.getPlotItem().getAxis('left').setStyle(
                autoExpandTextSpace=False, tickTextWidth=tick_width)

        pwm_plot = self.graphWidgetPWM.getPlotItem()
        self.current_viewbox = pg.ViewBox(enableMenu=False)
        self.current_viewbox.setMouseEnabled(x=False, y=False)
        self.current_viewbox.setZValue(pwm_plot.vb.zValue() - 1)
        pwm_plot.scene().addItem(self.current_viewbox)
        pwm_plot.getAxis('right').linkToView(self.current_viewbox)
        pwm_plot.getAxis('right').setLabel('Motor current', units='mA', color='#00ffff')
        pwm_plot.getAxis('right').enableAutoSIPrefix(False)
        pwm_plot.getAxis('right').setStyle(autoExpandTextSpace=False, tickTextWidth=tick_width)
        self.current_viewbox.setXLink(pwm_plot.vb)
        self.curve_current = pg.PlotDataItem(pen=pg.mkPen('c', width=2), connect="finite",
                                            stepMode=self._pg_step_mode())
        self.current_viewbox.addItem(self.curve_current)
        self.curve_current.hide()
        self._current_is_visible = False
        # Reserve the same right margin above, so the linked plots keep their
        # full shared time span when the current axis becomes visible.
        rpm_right = self.graphWidgetRPM.getPlotItem().getAxis('right')
        rpm_right.setStyle(showValues=False, tickLength=0)
        rpm_right.setPen(pg.mkPen(None))
        pwm_plot.vb.sigResized.connect(self._sync_current_viewbox)
        pwm_plot.getAxis('right').geometryChanged.connect(self._sync_current_viewbox)
        self._sync_current_viewbox()

        self.rpm_average_checkbox.toggled.connect(self._update_telemetry_overlays)
        self.motor_current_checkbox.toggled.connect(self._update_telemetry_overlays)
        self.modooperacion.currentTextChanged.connect(self._update_telemetry_overlays)
        self._update_telemetry_overlays()

    def _sync_current_viewbox(self):
        pwm_view = self.graphWidgetPWM.getPlotItem().vb
        self.current_viewbox.setGeometry(pwm_view.sceneBoundingRect())
        self.current_viewbox.linkedViewChanged(pwm_view, pg.ViewBox.XAxis)
        if self._current_is_visible:
            self.graphWidgetRPM.getPlotItem().getAxis('right').setWidth(
                self.graphWidgetPWM.getPlotItem().getAxis('right').width())

    def _update_telemetry_overlays(self, *_args, time_s=None, rpm=None):
        rpm_visible = self.rpm_average_checkbox.isChecked() and "Position" not in self.modooperacion.currentText()
        current_visible = self.motor_current_checkbox.isChecked()
        for visible, previous, curve, legend, name in (
                (rpm_visible, self._rpm_average_is_visible, self.curve_rpm_average,
                 self.legendRPM, "RPM average"),
                (current_visible, self._current_is_visible, self.curve_current,
                 self.legendPWM, "Motor current")):
            if visible != previous:
                if visible:
                    legend.addItem(curve, name)
                else:
                    legend.removeItem(curve)
            curve.setVisible(visible)
        self._rpm_average_is_visible = rpm_visible
        self._current_is_visible = current_visible
        self.graphWidgetPWM.getPlotItem().showAxis('right', current_visible)
        self.graphWidgetRPM.getPlotItem().showAxis('right', current_visible)
        self.current_viewbox.setVisible(current_visible)
        self._sync_current_viewbox()

        if time_s is None:
            n = min(len(self.dataDT), len(self.dataRPM_measured), len(self.dataPWM),
                    len(self.dataRPM_setpoint))
            time_s = np.cumsum(np.asarray(self.dataDT, dtype=float)[-n:] * 1e-3) if n else np.array([])
            if n:
                time_s -= time_s[0]
            rpm = np.asarray(self.dataRPM_measured, dtype=float)[-n:] if n else np.array([])
        n = len(time_s)
        if rpm_visible and n:
            self.curve_rpm_average.setData(time_s, centered_moving_average(time_s, rpm),
                                           stepMode=self._pg_step_mode())
        else:
            self.curve_rpm_average.setData([], [])
        if current_visible and n:
            current = np.full(n, np.nan)
            available = min(n, len(self.dataCurrent))
            if available:
                current[-available:] = np.asarray(self.dataCurrent, dtype=float)[-available:]
            finite_current = current[np.isfinite(current)]
            if len(finite_current):
                self.curve_current.setData(time_s, current, stepMode=self._pg_step_mode())
                self.current_viewbox.setYRange(min(0.0, float(finite_current.min())),
                                               max(1.0, float(finite_current.max())), padding=0.08)
            else:
                self.curve_current.setData([], [])
        else:
            self.curve_current.setData([], [])

    def _on_plot_mode_changed(self, _text=None):
        # Restyle the original samples; no acquisition, fitting or simulation runs.
        for widget in (self.graphWidgetRPM, self.graphWidgetPWM,
                       self.graphWidgetforced_response, self.graphWidgetsim_response):
            for curve in widget.listDataItems():
                x, y = curve.getOriginalDataset()
                if x is None or y is None:
                    x, y = [], []
                curve.setData(x, y, stepMode=self._pg_step_mode())
        x, y = self.curve_current.getOriginalDataset()
        self.curve_current.setData(x if x is not None else [], y if y is not None else [],
                                   stepMode=self._pg_step_mode())
        # Includes response plots in enlarged dialogs parented to the main GUI.
        for canvas in self.findChildren(FigureCanvas):
            changed = False
            for axes in canvas.figure.axes:
                for line in axes.lines:
                    if line.get_gid() == "time-series":
                        line.set_drawstyle(self._mpl_drawstyle())
                        changed = True
            if changed:
                canvas.draw_idle()

    def _set_runtime_plot_interaction(self, enabled):
        for plot_widget in self._runtime_plot_widgets():
            plot_item = plot_widget.getPlotItem()
            view_box = plot_item.getViewBox()
            view_box.setMouseEnabled(x=enabled, y=enabled)
            if hasattr(view_box, "setMenuEnabled"):
                view_box.setMenuEnabled(enabled)
            if hasattr(plot_item, "setMenuEnabled"):
                plot_item.setMenuEnabled(enabled)
            plot_widget.setCursor(
                Qt.CursorShape.OpenHandCursor if enabled else Qt.CursorShape.ArrowCursor
            )

    def _refresh_runtime_plots(self):
        n = min(len(self.dataDT), len(self.dataRPM_setpoint), len(self.dataRPM_measured), len(self.dataPWM))
        if not n:
            return
        time_s = np.cumsum(np.asarray(self.dataDT, dtype=float)[-n:] * 1e-3)
        time_s -= time_s[0]
        if n > 1 and time_s[-1] > time_s[0]:
            self.graphWidgetRPM.setXRange(float(time_s[0]), float(time_s[-1]), padding=0)
            self.graphWidgetPWM.setXRange(float(time_s[0]), float(time_s[-1]), padding=0)
        setpoint = np.asarray(self.dataRPM_setpoint, dtype=float)[-n:]
        measured = np.asarray(self.dataRPM_measured, dtype=float)[-n:]
        pwm = np.asarray(self.dataPWM, dtype=float)[-n:]
        self.curve_setpoint.setData(time_s, setpoint, stepMode=self._pg_step_mode())
        self.curve_measured.setData(time_s, measured, stepMode=self._pg_step_mode())
        self.curve_pwm.setData(time_s, pwm, stepMode=self._pg_step_mode())
        if "Position" in self.modooperacion.currentText():
            ymin = float(min(setpoint.min(initial=0), measured.min(initial=0)))
            ymax = float(max(setpoint.max(initial=0), measured.max(initial=0)))
            self._set_plot_y_range(self.graphWidgetRPM, ymin, ymax)
        else:
            ymax = float(max(setpoint.max(initial=0), measured.max(initial=0), 1.0))
            self._set_plot_y_range(self.graphWidgetRPM, 0.0, ymax, include_zero=True)
        self._set_plot_y_range(self.graphWidgetPWM, 0.0, float(max(pwm.max(initial=0), 1.0)), include_zero=True)
        self._update_telemetry_overlays(time_s=time_s, rpm=measured)

    def _reset_runtime_plots(self):
        datapoints = self.dataRPM_setpoint.maxlen or 1000
        self.dataRPM_setpoint = deque(maxlen=datapoints)
        self.dataRPM_measured = deque(maxlen=datapoints)
        self.dataPWM = deque(maxlen=datapoints)
        self.dataDT = deque(maxlen=datapoints)
        self.dataCurrent = deque(maxlen=datapoints)

        self.curve_setpoint.setData([], [])
        self.curve_measured.setData([], [])
        self.curve_pwm.setData([], [])
        self._update_telemetry_overlays()
        self.graphWidgetRPM.setXRange(0.0, 1.0, padding=0)
        self.graphWidgetPWM.setXRange(0.0, 1.0, padding=0)
        self.graphWidgetRPM.setYRange(0.0, 120.0)
        self.graphWidgetPWM.setYRange(0.0, 255.0)

    def _reference_controls_enabled_for_mode(self):
        return True

    def _update_reference_controls_visibility(self):
        show_source = self._reference_controls_enabled_for_mode()

        for widget in (self.manualinput, self.automaticinput):
            widget.setVisible(show_source)

        for widget in (self.reference, self.reference_label, self.slider):
            widget.setVisible(True)


    def _on_reference_source_changed(self, checked=False):
        self._update_reference_controls_visibility()
        if not checked:
            return
        if hasattr(self, "graphWidgetRPM"):
            self.toggleupdate_parameters()

    def _numeric_text(self, widget, default=None, minimum=None, maximum=None, integer=False):
        value = self._line_edit_number(
            widget,
            default=default,
            minimum=minimum,
            maximum=maximum,
            integer=integer,
        )
        if integer:
            return str(value)
        return f"{value:.9g}"

    def _set_plot_y_range(self, plot_widget, ymin, ymax, include_zero=False):
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            return
        if include_zero:
            ymin = min(0.0, ymin)
            ymax = max(0.0, ymax)
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        span = ymax - ymin
        margin = max(1.0, abs(ymax) * 0.1, abs(ymin) * 0.1) if span < 1e-9 else span * 0.1
        plot_widget.setYRange(ymin - margin, ymax + margin)

    def Simulate(self):
        try:
            # Get the numerator and denominator coefficients from UI
            num = [
                float(self.sim_num3.text() or 0),
                float(self.sim_num2.text() or 0),
                float(self.sim_num1.text() or 0),
                float(self.sim_num0.text() or 0),
            ]
            den = [
                float(self.sim_den3.text() or 0),
                float(self.sim_den2.text() or 0),
                float(self.sim_den1.text() or 0),
                float(self.sim_den0.text() or 0),
            ]

            # Remove only leading zeros; zeros inside the polynomial are meaningful.
            num = self._strip_leading_zeros(num)
            den = self._strip_leading_zeros(den)

            if all(abs(x) < 1e-12 for x in den):
                self.textBrowser.setText("Simulation denominator is all zeros.")
                return

            # Create the transfer function
            sys_tf = ctrl.TransferFunction(num, den)

            # Get the user-defined time scale from x_scale
            try:
                t_max = self._line_edit_float(self.x_scale, default=5.0)  # Default to 5 sec
                if t_max <= 0:
                    self.textBrowser.setText("Time scale must be greater than zero.")
                    return
            except ValueError:
                self.textBrowser.setText("Invalid time scale. Enter a numeric value.")
                return

            # Generate time vector
            t = np.linspace(0, t_max, 500)

            # Get the selected response type from UI
            response_type = self.sim_tiposenal.currentText().lower()

            # Compute the response
            if response_type == "step":
                t, y = ctrl.step_response(sys_tf, T=t)
            elif response_type == "impulse":
                t, y = ctrl.impulse_response(sys_tf, T=t)
            elif response_type == "ramp":
                t, y = ctrl.forced_response(sys_tf, T=t, U=t)
            else:
                self.textBrowser.setText("Invalid response type selected.")
                return

            # Clear and plot the new response
            self.graphWidgetsim_response.clear()
            self._plot_time_pg(self.graphWidgetsim_response, t, y, pen='y', name=response_type.capitalize() + " Response")
            # Auto-scale Y-axis
            ymin, ymax = np.min(y), np.max(y)
            margin = (ymax - ymin) * 0.1 if ymax != ymin else 0.1  # Add 10% margin
            self.graphWidgetsim_response.setYRange(ymin - margin, ymax + margin)

        except Exception as e:
            self.textBrowser.setText(f"Error in transfer function simulation: {e}")

    def clear_control_console(self):
        self.command_input.clear()
        self.result_output.clear()

    def _read_tf_from_numden_widgets(self, num_widgets, den_widgets):
        num = []
        den = []

        for w in num_widgets:
            txt = self._coefficient_text(w).strip()
            num.append(float(txt.replace(",", ".")) if txt else 0.0)

        for w in den_widgets:
            txt = self._coefficient_text(w).strip()
            den.append(float(txt.replace(",", ".")) if txt else 0.0)

        num = self._strip_leading_zeros(num)
        den = self._strip_leading_zeros(den)

        if all(abs(x) < 1e-12 for x in den):
            raise ValueError("Denominator is all zeros.")

        return ctrl.TransferFunction(num, den)

    def _build_control_console_namespace(self):
        ns = {}

        # Basic libraries
        ns["np"] = np
        ns["ctrl"] = ctrl

        # Friendly aliases
        ns["tf"] = ctrl.TransferFunction
        ns["series"] = ctrl.series
        ns["parallel"] = ctrl.parallel
        ns["feedback"] = ctrl.feedback
        ns["minreal"] = ctrl.minreal
        ns["pole"] = ctrl.poles
        ns["poles"] = ctrl.poles
        ns["zero"] = ctrl.zeros
        ns["zeros"] = ctrl.zeros
        ns["dcgain"] = ctrl.dcgain
        ns["bandwidth"] = ctrl.bandwidth
        ns["step_info"] = ctrl.step_info
        ns["rlocus"] = ctrl.root_locus_map

        def c2d(sys, Ts, method="zoh"):
            num = np.asarray(sys.num[0][0], dtype=float)
            den = np.asarray(sys.den[0][0], dtype=float)
            numz, denz, _ = cont2discrete((num, den), Ts, method=method)
            numz = np.ravel(numz).astype(float)
            denz = np.ravel(denz).astype(float)
            if abs(denz[0]) > 0:
                numz /= denz[0]
                denz /= denz[0]
            return ctrl.TransferFunction(numz, denz, Ts)

        ns["c2d"] = c2d

        # Preload useful systems from the GUI when possible
        try:
            ns["Gsim"] = self._read_tf_from_numden_widgets(
                [self.sim_num3, self.sim_num2, self.sim_num1, self.sim_num0],
                [self.sim_den3, self.sim_den2, self.sim_den1, self.sim_den0]
            )
        except Exception:
            pass

        try:
            ns["Gs"] = self._read_tf_from_numden_widgets(
                [self.rlocus_num3, self.rlocus_num2, self.rlocus_num1, self.rlocus_num0],
                [self.rlocus_den3, self.rlocus_den2, self.rlocus_den1, self.rlocus_den0]
            )
        except Exception:
            pass

        try:
            ns["C"] = self._read_tf_from_numden_widgets(
                [self.cont_num3, self.cont_num2, self.cont_num1, self.cont_num0],
                [self.cont_den3, self.cont_den2, self.cont_den1, self.cont_den0]
            )
        except Exception:
            pass

        try:
            ns["P"] = self._read_tf_from_numden_widgets(
                [self.plant_num3, self.plant_num2, self.plant_num1, self.plant_num0],
                [self.plant_den3, self.plant_den2, self.plant_den1, self.plant_den0]
            )
        except Exception:
            pass

        if self.identified_sys_tf is not None:
            ns["Gid"] = self.identified_sys_tf

        return ns

    def _format_console_result(self, obj):
        if isinstance(obj, ctrl.TransferFunction):
            return self.tf_to_pretty_str(obj)

        if isinstance(obj, (int, float, complex, np.number)):
            return str(obj)

        if isinstance(obj, np.ndarray):
            return np.array2string(obj, precision=6, suppress_small=True)

        return str(obj)

    def run_control_command(self):
        code = self.command_input.toPlainText().strip()
        if not code:
            self.result_output.setText("No command entered.")
            return

        ns = self._build_control_console_namespace()

        try:
            lines = [line for line in code.splitlines() if line.strip()]

            # Single expression
            if len(lines) == 1 and "=" not in lines[0]:
                result = eval(lines[0], {"__builtins__": {}}, ns)
                self.result_output.setText(self._format_console_result(result))
                return

            # Multi-line script with optional final expression
            last_expr = None
            body_lines = lines[:]

            if lines:
                try:
                    compile(lines[-1], "<console>", "eval")
                    if "=" not in lines[-1]:
                        last_expr = lines[-1]
                        body_lines = lines[:-1]
                except SyntaxError:
                    pass

            if body_lines:
                exec("\n".join(body_lines), {"__builtins__": {}}, ns)

            if last_expr is not None:
                result = eval(last_expr, {"__builtins__": {}}, ns)
                self.result_output.setText(self._format_console_result(result))
            else:
                visible_vars = []
                for k in sorted(ns.keys()):
                    if k.startswith("_"):
                        continue
                    if k in ("np", "ctrl"):
                        continue
                    visible_vars.append(k)

                self.result_output.setText(
                    "Command executed.\n\nAvailable variables now:\n" + ", ".join(visible_vars)
                )

        except Exception as e:
            self.result_output.setText(f"Command error: {e}")

    def _read_tf_from_zrlocus_inputs(self):
        num = [
            self._read_float(self.rlocusz_num3),
            self._read_float(self.rlocusz_num2),
            self._read_float(self.rlocusz_num1),
            self._read_float(self.rlocusz_num0),
        ]
        den = [
            self._read_float(self.rlocusz_den3),
            self._read_float(self.rlocusz_den2),
            self._read_float(self.rlocusz_den1),
            self._read_float(self.rlocusz_den0),
        ]

        num = self._strip_leading_zeros(num)
        den = self._strip_leading_zeros(den)

        if all(abs(x) < 1e-12 for x in num):
            raise ValueError("Discrete RL numerator is all zeros.")
        if all(abs(x) < 1e-12 for x in den):
            raise ValueError("Discrete RL denominator is all zeros.")

        Ts = self._line_edit_float(self.lqr_sampling, default=0.02)
        if Ts <= 0:
            raise ValueError("Sampling time must be greater than zero.")

        return ctrl.TransferFunction(num, den, Ts)

    def _setup_discrete_rlocus_tab_plots(self):
        for canvas in (self.canvas_zrl, self.canvas_zstep):
            canvas.figure.set_size_inches(4, 3)
            canvas.figure.set_layout_engine("tight")
        self.ax_zrl = self.canvas_zrl.figure.add_subplot(111)
        self.ax_zstep = self.canvas_zstep.figure.add_subplot(111)

        self._analysis_z_timer = QTimer(self)
        self._analysis_z_timer.setSingleShot(True)
        self._analysis_z_timer.timeout.connect(self.update_discrete_rlocus_plots)

        def _schedule():
            self._analysis_z_timer.start(200)

        for le in (
            self.rlocusz_num3, self.rlocusz_num2, self.rlocusz_num1, self.rlocusz_num0,
            self.rlocusz_den3, self.rlocusz_den2, self.rlocusz_den1, self.rlocusz_den0,
            self.lqr_sampling,
        ):
            le.textChanged.connect(_schedule)

        self.canvas_zrl.mpl_connect("button_press_event", self._on_zrlocus_canvas_click)

        self.update_discrete_rlocus_plots()

    def update_discrete_rlocus_plots(self):
        try:
            # If user hasn't entered a valid plant yet, just show blank axes quietly
            num_test = [
                self._read_float(self.rlocusz_num3),
                self._read_float(self.rlocusz_num2),
                self._read_float(self.rlocusz_num1),
                self._read_float(self.rlocusz_num0),
            ]
            den_test = [
                self._read_float(self.rlocusz_den3),
                self._read_float(self.rlocusz_den2),
                self._read_float(self.rlocusz_den1),
                self._read_float(self.rlocusz_den0),
            ]
            num_test = self._strip_leading_zeros(num_test)
            den_test = self._strip_leading_zeros(den_test)

            if all(abs(x) < 1e-12 for x in den_test) or all(abs(x) < 1e-12 for x in num_test):
                self.ax_zrl.clear()
                self.ax_zstep.clear()

                try:
                    Ts_grid = float(self.lqr_sampling.text())
                except Exception:
                    Ts_grid = None
                self._set_zplane_limits(self.ax_zrl)
                self._overlay_zgrid(self.ax_zrl, show_labels=False, Ts=Ts_grid)

                self.ax_zstep.grid(True)

                self.canvas_zrl.draw_idle()
                self.canvas_zstep.draw_idle()
                return

            sysz = self._read_tf_from_zrlocus_inputs()

            self.ax_zrl.clear()
            self.ax_zstep.clear()

            # Root locus in z-plane
            plot_root_locus_clean(sysz, self.ax_zrl)
            self._set_zplane_limits(
                self.ax_zrl,
                [
                    ctrl.poles(sysz),
                    ctrl.zeros(sysz),
                ]
            )
            self._overlay_zgrid(self.ax_zrl, show_labels=False, Ts=float(sysz.dt))
            self._strip_titles_labels(self.ax_zrl, keep_x=False, keep_y=False)

            # Step response with refined x-axis
            Ts = float(sysz.dt)
            Tclose = ctrl.feedback(sysz, 1, sign=-1)

            N = 200
            t = np.arange(N, dtype=float) * Ts
            t, y = ctrl.step_response(Tclose, T=t)

            try:
                info = ctrl.step_info(Tclose)
                ts = float(info.get("SettlingTime", np.nan))
                if np.isfinite(ts) and ts > 0:
                    t_final = max(8 * Ts, 2.5 * ts)
                    N = max(80, int(np.ceil(t_final / Ts)))
                    t = np.arange(N, dtype=float) * Ts
                    t, y = ctrl.step_response(Tclose, T=t)
            except Exception:
                pass

            self._plot_time_mpl(self.ax_zstep, t, y)
            self.ax_zstep.axhline(1.0, linestyle="--", linewidth=1)
            self.ax_zstep.grid(True)
            self.ax_zstep.set_xlim(float(t[0]), float(t[-1]))
            self._strip_titles_labels(self.ax_zstep, keep_x=False, keep_y=False)

            self.canvas_zrl.draw_idle()
            self.canvas_zstep.draw_idle()

        except Exception:
            # Quiet on purpose; avoid spamming console while user is typing
            self.ax_zrl.clear()
            self.ax_zstep.clear()
            try:
                Ts_grid = float(self.lqr_sampling.text())
            except Exception:
                Ts_grid = None
            self._set_zplane_limits(self.ax_zrl)
            self._overlay_zgrid(self.ax_zrl, show_labels=False, Ts=Ts_grid)
            self.ax_zstep.grid(True)
            self.canvas_zrl.draw_idle()
            self.canvas_zstep.draw_idle()

    def _on_zrlocus_canvas_click(self, event):
        if event.inaxes != self.ax_zrl:
            return
        if event.button != 1:
            return
        if not getattr(event, "dblclick", False):
            return
        self.open_big_zrlocus_from_tab()

    def open_big_zrlocus_from_tab(self):
        try:
            sysz = self._read_tf_from_zrlocus_inputs()
            Ts = self._line_edit_float(self.lqr_sampling, default=0.02, minimum=1e-9)
            key = design_state_key("discrete", sysz, extra=(round(float(Ts), 12),))
            state = self._design_states.get(key)
            dlg = DiscreteControlPlotDialog(self, sysz, Ts, state=state)
            dlg.exec()
            self._design_states[key] = dlg.get_state()
        except Exception as e:
            if hasattr(self, "plot_error"):
                self.plot_error.setText(f"Discrete RL error: {e}")

    def _sync_sampling_lineedits(self):
        def copy_a_to_b():
            txt = self.sampling_time.text()
            self.lqr_sampling.blockSignals(True)
            self.lqr_sampling.setText(txt)
            self.lqr_sampling.blockSignals(False)

        def copy_b_to_a():
            txt = self.lqr_sampling.text()
            self.sampling_time.blockSignals(True)
            self.sampling_time.setText(txt)
            self.sampling_time.blockSignals(False)

        self.sampling_time.textChanged.connect(copy_a_to_b)
        self.lqr_sampling.textChanged.connect(copy_b_to_a)

    def _setup_pid_tuner(self):
        if hasattr(self, "tuner_speed_value"):
            self.tuner_speed_value.setText(str(self.tuner_speed_slider.value()))
        if hasattr(self, "tuner_damping_value"):
            self.tuner_damping_value.setText(str(self.tuner_damping_slider.value()))

        self._setup_pid_tuner_plots()

        self._pid_tuner_timer = QTimer(self)
        self._pid_tuner_timer.setSingleShot(True)
        self._pid_tuner_timer.timeout.connect(self._update_pid_tuner)

        def _schedule():
            self._pid_tuner_timer.start(150)

        self.tuner_speed_slider.valueChanged.connect(_schedule)
        self.tuner_damping_slider.valueChanged.connect(_schedule)
        self.tuner_controller_type.currentIndexChanged.connect(_schedule)
        self.time_constant.textChanged.connect(_schedule)

        self.tuner_load_identified_btn.clicked.connect(self._load_tuner_plant)

        if hasattr(self, "tuner_speed_value"):
            self.tuner_speed_slider.valueChanged.connect(
                lambda v: self.tuner_speed_value.setText(str(v))
            )

        if hasattr(self, "tuner_damping_value"):
            self.tuner_damping_slider.valueChanged.connect(
                lambda v: self.tuner_damping_value.setText(str(v))
            )

        self._clear_pid_tuner_plots()

    def _on_rlocus_canvas_click(self, event):
        if event.inaxes != self.ax_rl:
            return
        if not getattr(event, "dblclick", False):
            return

        # Left double click -> continuous RL
        if event.button == 1:
            self.open_big_rlocus()

        # Right double click -> discrete RL
        elif event.button == 3:
            self.open_big_zrlocus()

    def _setup_pid_tuner_plots(self):
        self.tuner_step_fig = self.tuner_step_canvas.figure
        self.tuner_step_fig.set_layout_engine("constrained")
        self.tuner_step_ax = self.tuner_step_fig.add_subplot(111)

        self.tuner_step_canvas.mpl_connect(
        "button_press_event",
        self._on_pid_tuner_plot_click
        )

    def _on_pid_tuner_plot_click(self, event):
        if event.inaxes != self.tuner_step_ax:
            return
        if event.button != 1:
            return
        if not getattr(event, "dblclick", False):
            return
        if self.pid_tuner_dialog_open:
            return

        self.open_big_pid_tuner_step()

    def _clear_pid_tuner_plots(self, error=None):
        self.tuner_step_ax.clear()
        self.tuner_step_ax.grid(True)
        self.tuner_step_ax.set_title("Closed-loop step response")
        if error:
            self.tuner_step_ax.text(.05, .5, str(error), transform=self.tuner_step_ax.transAxes,
                                    wrap=True, va="center")
        self.tuner_step_canvas.draw_idle()

    def _get_tuner_plant(self):
        if self.tuner_loaded_plant is None:
            raise ValueError("No plant loaded. Click 'Load plant' first.")
        return self.tuner_loaded_plant

    def _read_simulation_tf(self):
        num = [
            float(self.sim_num3.text() or 0),
            float(self.sim_num2.text() or 0),
            float(self.sim_num1.text() or 0),
            float(self.sim_num0.text() or 0),
        ]
        den = [
            float(self.sim_den3.text() or 0),
            float(self.sim_den2.text() or 0),
            float(self.sim_den1.text() or 0),
            float(self.sim_den0.text() or 0),
        ]

        num = self._strip_leading_zeros(num)
        den = self._strip_leading_zeros(den)

        if all(abs(x) < 1e-12 for x in den):
            raise ValueError("Simulation denominator is all zeros.")

        return ctrl.TransferFunction(num, den)

    def _load_tuner_plant(self):
        try:
            source = self.tuner_plant_source.currentText().strip().lower()

            if "identified" in source:
                if self.identified_sys_tf is None:
                    raise ValueError("No identified plant available. Run system identification first.")
                self.tuner_loaded_plant = self.identified_sys_tf

            elif "simulation" in source:
                self.tuner_loaded_plant = self._read_simulation_tf()

            else:
                raise ValueError("Unknown plant source selected.")

            self._update_pid_tuner()

        except Exception as e:
            self.tuner_loaded_plant = None
            self.tuner_kp.setText("")
            self.tuner_ki.setText("")
            self.tuner_kd.setText("")
            self._clear_pid_tuner_plots(f"Load plant error: {e}")
            print(f"Load plant error: {e}")

    def _pid_tuner_derivative_filter_time(self):
        return self._line_edit_float(self.time_constant, default=0.2, minimum=1e-6)

    def _update_pid_tuner(self):
        if self.tuner_loaded_plant is None:
            self.tuner_kp.setText("")
            self.tuner_ki.setText("")
            self.tuner_kd.setText("")
            self._clear_pid_tuner_plots()
            return

        try:
            G = self._get_tuner_plant()
            ctype = self.tuner_controller_type.currentText().strip().upper()

            kp, ki, kd, C, target_ts, target_os, _ = self._tune_pid_from_ui(G, ctype)

            self.tuner_kp.setText(f"{kp:.6g}")
            self.tuner_ki.setText(f"{ki:.6g}")
            self.tuner_kd.setText(f"{kd:.6g}")

            T = ctrl.feedback(C * G, 1)

            # First pass
            t_final = max(0.5, 4.0 * target_ts)
            t = np.linspace(0, t_final, 800)
            t_y, y = ctrl.step_response(T, T=t)

            # Refine time axis using actual settling time
            try:
                info = ctrl.step_info(T)
                ts_actual = float(info.get("SettlingTime", np.nan))
                if np.isfinite(ts_actual) and ts_actual > 0:
                    t_final = max(0.5, 2.5 * ts_actual)
                    t = np.linspace(0, t_final, 1000)
                    t_y, y = ctrl.step_response(T, T=t)
            except Exception:
                pass

            self.tuner_step_ax.clear()
            self._plot_time_mpl(self.tuner_step_ax, t_y, y, label="y(t)")
            self.tuner_step_ax.axhline(1.0, linestyle="--", linewidth=1)
            self.tuner_step_ax.grid(True)
            self.tuner_step_ax.set_title("Closed-loop step response")
            self.tuner_step_ax.set_xlabel("Time (s)")
            self.tuner_step_ax.set_ylabel("Amplitude")
            self.tuner_step_ax.legend(loc="best")
            self.tuner_step_canvas.draw_idle()

        except Exception as e:
            self.tuner_kp.setText("")
            self.tuner_ki.setText("")
            self.tuner_kd.setText("")
            self._clear_pid_tuner_plots(f"PID tuner error: {e}")
            if hasattr(self, "tuner_status"):
                self.tuner_status.setText(f"PID tuner error: {e}")

    def _tune_pid_from_ui(self, G, ctype, speed_value=None, damping_value=None):
        speed = self.tuner_speed_slider.value() if speed_value is None else speed_value
        damping = self.tuner_damping_slider.value() if damping_value is None else damping_value
        result = tune_pid(G, ctype, speed, damping, self._pid_tuner_derivative_filter_time())
        return (result.kp, result.ki, result.kd, result.controller,
                result.target_settling_time, result.target_overshoot, True)

    def reduceTF(self):
        def _f(le):
            """Read float from QLineEdit; empty/invalid -> 0."""
            try:
                txt = le.text().strip()
                return float(txt) if txt else 0.0
            except Exception:
                return 0.0

        def _strip_leading_zeros(coefs, eps=1e-12):
            c = list(map(float, coefs))
            while len(c) > 1 and abs(c[0]) < eps:
                c.pop(0)
            return c

        try:
            # ---- Controller C(s) ----
            c_num = [_f(self.cont_num3), _f(self.cont_num2), _f(self.cont_num1), _f(self.cont_num0)]
            c_den = [_f(self.cont_den3), _f(self.cont_den2), _f(self.cont_den1), _f(self.cont_den0)]

            # ---- Plant G(s) ----
            g_num = [_f(self.plant_num3), _f(self.plant_num2), _f(self.plant_num1), _f(self.plant_num0)]
            g_den = [_f(self.plant_den3), _f(self.plant_den2), _f(self.plant_den1), _f(self.plant_den0)]

            # Clean leading zeros
            c_num = _strip_leading_zeros(c_num)
            c_den = _strip_leading_zeros(c_den)
            g_num = _strip_leading_zeros(g_num)
            g_den = _strip_leading_zeros(g_den)

            # Basic sanity checks
            if all(abs(x) < 1e-12 for x in c_den):
                self.series_tf.clear()
                self.equivalent_tf.setText("Error: Controller denominator is all zeros.")
                return
            if all(abs(x) < 1e-12 for x in g_den):
                self.series_tf.clear()
                self.equivalent_tf.setText("Error: Plant denominator is all zeros.")
                return

            C = ctrl.TransferFunction(c_num, c_den)
            G = ctrl.TransferFunction(g_num, g_den)

            # Series + negative unity feedback
            L = ctrl.series(C, G)
            T = ctrl.feedback(L, 1, sign=-1)  # negative feedback

            self.series_tf.setPlainText(self.tf_to_pretty_str(L))
            self.equivalent_tf.setPlainText(self.tf_to_pretty_str(T))

        except Exception as e:
            self.series_tf.clear()
            self.equivalent_tf.setText(f"Error reducing block diagram: {e}")

    def tf_to_pretty_str(self, sys, var='s', digits=6, eps=1e-12, decimals=None):
        """Return only the transfer function fraction (no sys[..], no I/O labels)."""
        num = np.asarray(sys.num[0][0], dtype=float).copy()
        den = np.asarray(sys.den[0][0], dtype=float).copy()

        num[np.abs(num) < eps] = 0.0
        den[np.abs(den) < eps] = 0.0

        def poly_str(c):
            n = len(c) - 1
            terms = []
            for i, a in enumerate(c):
                p = n - i
                if abs(a) < eps:
                    continue
                a_str = f"{a:.{digits}g}" if decimals is None else f"{a:.{decimals}f}"

                if p == 0:
                    terms.append(f"{a_str}")
                elif p == 1:
                    if abs(a - 1.0) < eps:
                        terms.append(f"{var}")
                    elif abs(a + 1.0) < eps:
                        terms.append(f"-{var}")
                    else:
                        terms.append(f"{a_str} {var}")
                else:
                    if abs(a - 1.0) < eps:
                        terms.append(f"{var}^{p}")
                    elif abs(a + 1.0) < eps:
                        terms.append(f"-{var}^{p}")
                    else:
                        terms.append(f"{a_str} {var}^{p}")

            if not terms:
                return "0"
            return " + ".join(terms).replace("+ -", "- ")

        num_s = poly_str(num)
        den_s = poly_str(den)
        bar = "-" * max(len(num_s), len(den_s), 12)
        return f"{num_s}\n{bar}\n{den_s}"

    def _strip_titles_labels(self, ax, keep_x=False, keep_y=False):
        ax.set_title("")
        if not keep_x:
            ax.set_xlabel("")
        if not keep_y:
            ax.set_ylabel("")

    def _get_current_analysis_sys(self):
        # Use the same transfer function you already read for rlocus inputs
        return self._read_tf_from_rlocus_inputs()

    def open_big_rlocus(self):
        try:
            sys = self._read_tf_from_rlocus_inputs()
            key = design_state_key("continuous", sys)
            state = self._design_states.get(key)
            dlg = ControlPlotDialog(self, sys, state=state)
            dlg.exec()
            self._design_states[key] = dlg.get_state()
        except Exception as e:
            if hasattr(self, "plot_error"):
                self.plot_error.setText(f"Root locus error: {e}")

    def open_big_zrlocus(self):
        try:
            sysz = self._read_discrete_rlocus_sys()
            Ts = self._line_edit_float(self.sampling_time, default=0.02, minimum=1e-9)
            key = design_state_key("discrete", sysz, extra=(round(float(Ts), 12),))
            state = self._design_states.get(key)
            dlg = DiscreteControlPlotDialog(self, sysz, Ts, state=state)
            dlg.exec()
            self._design_states[key] = dlg.get_state()
        except Exception as e:
            if hasattr(self, "plot_error"):
                self.plot_error.setText(f"Discrete root locus error: {e}")

    def _read_discrete_rlocus_sys(self):
        Gs = self._read_tf_from_rlocus_inputs()

        Ts = self._line_edit_float(self.sampling_time, default=0.02)
        if Ts <= 0:
            raise ValueError("Sampling time Ts must be greater than zero.")

        num = np.asarray(Gs.num[0][0], dtype=float)
        den = np.asarray(Gs.den[0][0], dtype=float)

        numz, denz, _ = cont2discrete((num, den), Ts, method="zoh")
        numz = np.ravel(numz).astype(float)
        denz = np.ravel(denz).astype(float)

        if abs(denz[0]) > 0:
            numz /= denz[0]
            denz /= denz[0]

        return ctrl.TransferFunction(numz, denz, Ts)

    def _on_pid_tuner_dialog_closed(self, _result):
        self.pid_tuner_dialog_open = False

    def _overlay_sgrid(self, ax, show_labels=True):
        """
        Overlay an s-plane damping grid on top of an existing root-locus plot.
        Does NOT change aspect ratio or limits.

        Returns the artists it creates, so the caller can remove and redraw the
        grid when the user zooms or pans (the grid is computed from the current
        visible limits).
        """
        artists = []

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xmin, xmax = xlim
        ymin, ymax = ylim

        # Only meaningful for left-half plane
        if xmin >= 0:
            return artists

        # Turn off rectangular grid; keep only damping grid
        ax.grid(False)

        grid_color = '0.74'
        grid_lw = 0.8
        grid_ls = ':'

        # Axes
        artists.append(ax.axhline(0, color='0.65', lw=0.8, zorder=0))
        artists.append(ax.axvline(0, color='0.65', lw=0.8, zorder=0))

        # Damping ratio lines only
        zetas = [0.1, 0.2, 0.3, 0.4, 0.50, 0.6, 0.7, 0.8, 0.9]
        rmax = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) * 1.2

        for z in zetas:
            theta = np.arccos(z)

            for sign in (+1, -1):
                ang = np.pi - sign * theta
                x = np.array([0.0, rmax * np.cos(ang)])
                y = np.array([0.0, rmax * np.sin(ang)])
                artists.extend(
                    ax.plot(x, y, color=grid_color, lw=grid_lw, ls=grid_ls, zorder=0)
                )

            if show_labels:
                # Put labels on the damping lines at a fixed radius from the origin
                rlabel = 0.88 * max(abs(xmin), abs(ymax), abs(ymin))
                xlab = -z * rlabel
                ylab = np.sqrt(max(0.0, 1.0 - z**2)) * rlabel

                if xmin < xlab < xmax and ymin < ylab < ymax:
                    artists.append(
                        ax.text(
                            xlab, ylab, f"{z:.2f}",
                            color='0.45', fontsize=8,
                            ha='center', va='bottom',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.15)
                        )
                    )

                if xmin < xlab < xmax and ymin < -ylab < ymax:
                    artists.append(
                        ax.text(
                            xlab, -ylab, f"{z:.2f}",
                            color='0.45', fontsize=8,
                            ha='center', va='top',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.15)
                        )
                    )

        # ---- Natural frequency circles (left-half-plane semicircles) ----
        # Choose wn values from current visible limits
        max_r = max(abs(xmin), abs(ymin), abs(ymax))
        if max_r <= 0:
            max_r = 1.0

        # Build a few candidate radii across the visible range
        raw_wns = np.linspace(max_r / 6, max_r, 6)

        def _nice_wn(v):
            if v <= 0:
                return 0.0
            p = 10 ** np.floor(np.log10(v))
            m = v / p
            if m < 1.5:
                m = 1
            elif m < 3.5:
                m = 2
            elif m < 7.5:
                m = 5
            else:
                m = 10
            return m * p

        wns = []
        for v in raw_wns:
            nv = _nice_wn(v)
            if nv > 0 and nv not in wns:
                wns.append(nv)

        th = np.linspace(np.pi/2, 3*np.pi/2, 400)
        for wn in wns:
            x = wn * np.cos(th)
            y = wn * np.sin(th)
            artists.extend(
                ax.plot(x, y, color=grid_color, lw=grid_lw, ls=grid_ls, zorder=0)
            )

            if show_labels and xmin < -wn < xmax:
                artists.append(
                    ax.text(
                        -wn, 0.03 * (ymax - ymin), f"{wn:g}",
                        color='0.45', fontsize=8,
                        ha='center', va='bottom',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.15)
                    )
                )

        return artists

    def _finite_complex_points(self, *values):
        points = []
        for value in values:
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=complex).ravel()
            except Exception:
                continue
            mask = np.isfinite(arr.real) & np.isfinite(arr.imag)
            if np.any(mask):
                points.append(arr[mask])
        if not points:
            return np.array([], dtype=complex)
        return np.concatenate(points)

    def _axis_complex_points(self, ax):
        points = []

        for line in ax.lines:
            try:
                x = np.asarray(line.get_xdata(), dtype=float).ravel()
                y = np.asarray(line.get_ydata(), dtype=float).ravel()
            except Exception:
                continue
            n = min(len(x), len(y))
            if n:
                points.append(x[:n] + 1j * y[:n])

        for collection in ax.collections:
            try:
                offsets = np.asarray(collection.get_offsets(), dtype=float)
            except Exception:
                continue
            if offsets.ndim == 2 and offsets.shape[1] >= 2:
                points.append(offsets[:, 0] + 1j * offsets[:, 1])

        return self._finite_complex_points(*points)

    def _set_zplane_limits(self, ax, points=None, aspect="equal"):
        unit = np.exp(1j * np.linspace(0, 2 * np.pi, 360))

        values = [unit]
        if points is not None:
            if isinstance(points, (list, tuple)):
                values.extend(points)
            else:
                values.append(points)

        pts = self._finite_complex_points(*values)
        if len(pts) == 0:
            pts = unit

        xmin = float(np.min(pts.real))
        xmax = float(np.max(pts.real))
        ymin = float(np.min(pts.imag))
        ymax = float(np.max(pts.imag))

        span = max(xmax - xmin, ymax - ymin, 2.0)
        span *= 1.12

        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)

        ax.set_xlim(xmid - 0.5 * span, xmid + 0.5 * span)
        ax.set_ylim(ymid - 0.5 * span, ymid + 0.5 * span)
        if aspect == "equal":
            ax.set_aspect('equal', adjustable='box')
        else:
            # Fill the available rectangle: the unit circle becomes an ellipse
            # and the z-grid is drawn stretched, like the classic textbook plot.
            ax.set_aspect('auto')

    def _overlay_zgrid(self, ax, show_labels=True, Ts=None):
        """
        Overlay a z-plane grid on top of a discrete root-locus plot:

          - unit circle,
          - constant damping ratio curves (spirals z = exp((-zeta + j*sqrt(1-zeta^2))*w*T)),
          - constant normalized frequency curves (w*T = fraction of pi).

        Returns the artists it creates, so the caller can refresh the grid when
        the user zooms (the labels are clipped to the visible limits).
        """
        artists = []
        ax.grid(False)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        circle_color = '0.35'
        freq_color = '0.40'
        damping_color = '0.58'
        grid_lw = 0.8
        grid_ls = ':'

        def _visible(z, margin=0.0):
            return (
                (xmin - margin) <= z.real <= (xmax + margin)
                and (ymin - margin) <= z.imag <= (ymax + margin)
            )

        label_bbox = dict(facecolor='white', edgecolor='none', alpha=0.80, pad=0.6)

        # ---- Unit circle and axes ----
        theta = np.linspace(0.0, 2.0 * np.pi, 720)
        artists.append(
            ax.plot(
                np.cos(theta), np.sin(theta),
                color=circle_color, lw=1.0, zorder=1
            )[0]
        )
        artists.append(ax.axhline(0.0, color='0.72', lw=0.8, zorder=1))
        artists.append(ax.axvline(0.0, color='0.72', lw=0.8, zorder=1))

        # ---- Constant damping ratio (zeta) curves ----
        zetas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        theta_d = np.linspace(0.0, np.pi, 600)

        # Labels are placed where the zeta curves cross the w*T = 0.35*pi curve,
        # which reproduces the tidy near-vertical column of the classic z-grid.
        rho_label = 0.35 * np.pi

        for zeta in zetas:
            beta = zeta / np.sqrt(max(1e-12, 1.0 - zeta ** 2))
            radius = np.exp(-beta * theta_d)
            upper = radius * np.exp(1j * theta_d)

            artists.extend(
                ax.plot(
                    upper.real, upper.imag,
                    color=damping_color, lw=grid_lw, ls=grid_ls, zorder=1
                )
            )
            artists.extend(
                ax.plot(
                    upper.real, -upper.imag,
                    color=damping_color, lw=grid_lw, ls=grid_ls, zorder=1
                )
            )

            if show_labels:
                theta_lbl = np.sqrt(max(0.0, 1.0 - zeta ** 2)) * rho_label
                r_lbl = np.exp(-zeta * rho_label)
                z_label = r_lbl * np.exp(1j * theta_lbl)
                if _visible(z_label):
                    artists.append(
                        ax.text(
                            z_label.real, z_label.imag, f"{zeta:.1f}",
                            color='0.15', fontsize=8,
                            ha='center', va='bottom',
                            bbox=label_bbox, zorder=4
                        )
                    )

        # ---- Constant normalized frequency curves (w*T between 0 and pi) ----
        fractions = np.arange(1, 10) / 10.0

        for frac in fractions:
            rho = frac * np.pi
            theta_n = np.linspace(-rho, rho, 600)
            log_radius = -np.sqrt(np.maximum(rho ** 2 - theta_n ** 2, 0.0))
            radius = np.exp(log_radius)
            z_curve = radius * np.exp(1j * theta_n)

            artists.extend(
                ax.plot(
                    z_curve.real, z_curve.imag,
                    color=freq_color, lw=grid_lw, ls=grid_ls, zorder=1
                )
            )

            if show_labels:
                label_r = 0.90
                for sign, va in ((+1, 'top'), (-1, 'bottom')):
                    z_lab = label_r * np.exp(1j * sign * rho)
                    if _visible(z_lab):
                        artists.append(
                            ax.text(
                                z_lab.real, z_lab.imag, f"{frac:.1f}π/T",
                                color='0.15', fontsize=7.5,
                                ha='center', va=va,
                                bbox=label_bbox, zorder=4
                            )
                        )

        # Nyquist limit (w*T = pi) sits on the negative real axis.
        if show_labels:
            for sign, va in ((+1, 'bottom'), (-1, 'top')):
                z_lab = complex(-0.97, sign * 0.045)
                if _visible(z_lab):
                    artists.append(
                        ax.text(
                            z_lab.real, z_lab.imag, "1π/T",
                            color='0.15', fontsize=7.5,
                            ha='left', va=va,
                            bbox=label_bbox, zorder=4
                        )
                    )

        return artists

    def open_big_pid_tuner_step(self):
        if self.pid_tuner_dialog_open:
            return

        try:
            G = self._get_tuner_plant()
            ctype = self.tuner_controller_type.currentText().strip().upper()
            speed_value = self.tuner_speed_slider.value()
            damping_value = self.tuner_damping_slider.value()

            self.pid_tuner_dialog_open = True

            dlg = PIDTunerPlotDialog(
                self,
                G=G,
                ctype=ctype,
                speed_value=speed_value,
                damping_value=damping_value,
            )

            dlg.finished.connect(self._on_pid_tuner_dialog_closed)
            dlg.exec()

        except Exception as e:
            self.pid_tuner_dialog_open = False
            if hasattr(self, "tuner_status"):
                self.tuner_status.setText(f"PID tuner error: {e}")

    def _update_prewarp_controls(self, *_):
        is_bilinear = self.method.currentText() == "bilinear"
        self.prewarp_tustin.setEnabled(is_bilinear)
        self.prewarp_frequency.setEnabled(is_bilinear and self.prewarp_tustin.isChecked())

    def discretize_function(self):

        def _strip_leading_zeros(coefs, eps=1e-12):
            # input is [s^n, s^(n-1), ..., s^0]; remove zeros from the front
            c = list(map(float, coefs))
            while c and abs(c[0]) < eps:
                c.pop(0)
            return c if c else [0.0]

        def _poly_to_zinv_str(c, fmt='{:.5g}', eps=1e-9):
            # c = [c0, c1, c2, ...] -> c0 + c1 z^-1 + c2 z^-2 + ...
            parts = []
            for k, ck in enumerate(np.asarray(c, float)):
                if abs(ck) < eps:
                    continue
                if k == 0:
                    parts.append(fmt.format(ck))
                else:
                    parts.append(f"{fmt.format(ck)} z^-{k}")
            s = " + ".join(parts) if parts else "0"
            return s.replace("+ -", "- ")

        try:
            # Read S-domain polynomials (entered as s^3,s^2,s^1,s^0)
            num = [
                float(self.snum3.text() or 0),
                float(self.snum2.text() or 0),
                float(self.snum1.text() or 0),
                float(self.snum0.text() or 0),
            ]
            den = [
                float(self.sden3.text() or 0),
                float(self.sden2.text() or 0),
                float(self.sden1.text() or 0),
                float(self.sden0.text() or 0),
            ]
            T_s = self._line_edit_float(self.sampling_time, default=0.02)
            if not np.isfinite(T_s) or T_s <= 0:
                raise ValueError("Sampling time must be finite and greater than zero.")
            method = self.method.currentText()  # e.g., 'zoh', 'foh', 'bilinear' (tustin)
            prewarp_frequency = None
            discretization_dt = T_s
            if method == "bilinear" and self.prewarp_tustin.isChecked():
                try:
                    prewarp_frequency = float(self.prewarp_frequency.text().strip().replace(",", "."))
                except ValueError:
                    raise ValueError("Enter a numeric prewarp frequency in rad/s.") from None
                if not np.isfinite(prewarp_frequency) or not 0 < prewarp_frequency < np.pi / T_s:
                    raise ValueError(
                        f"Prewarp frequency must be finite and satisfy 0 < frequency < {np.pi / T_s:g} rad/s (Nyquist)."
                    )
                # Effective Tustin step: 2/dt = wp/tan(wp*Ts/2).
                # The actual sampling period remains T_s.
                half_angle = prewarp_frequency * T_s / 2
                discretization_dt = T_s * (np.tan(half_angle) / half_angle if half_angle else 1.0)

            # Clean leading zeros (highest powers)
            num = _strip_leading_zeros(num)
            den = _strip_leading_zeros(den)

            # Discretize (SciPy returns TF in z^-1)
            num_z, den_z, _ = cont2discrete((num, den), discretization_dt, method)
            num_z = np.ravel(num_z).astype(float)
            den_z = np.ravel(den_z).astype(float)

            # Normalize so a0 == 1
            if abs(den_z[0]) > 0:
                num_z /= den_z[0]
                den_z /= den_z[0]

            # Pretty string in z^-1 form
            num_str = _poly_to_zinv_str(num_z)
            den_str = _poly_to_zinv_str(den_z)

            # --- pretty fraction formatting (same style as tf_to_pretty_str) ---
            num_s = f"({num_str})"
            den_s = f"({den_str})"
            bar = "-" * max(len(num_s), len(den_s), 12)

            result = (
                "H(z) =\n"
                f"{num_s}\n"
                f"{bar}\n"
                f"{den_s}\n"
                f"Ts = {T_s:g} s"
            )
            if prewarp_frequency is not None:
                result += f"\nPrewarp = {prewarp_frequency:g} rad/s"

            # Use plain text if available (keeps line breaks reliably)
            if hasattr(self.discretizationresult, "setPlainText"):
                self.discretizationresult.setPlainText(result)
            else:
                self.discretizationresult.setText(result)


        except Exception as e:
            self.discretizationresult.setText(f"Error: {e}")

    def resize_deque(self):
        try:
            datapoints = self._line_edit_int(self.datapoints, default=200, minimum=1)
        except ValueError:
            return
        datapoints = max(1, datapoints)
        self.dataRPM_setpoint = deque(list(self.dataRPM_setpoint)[-datapoints:], maxlen=datapoints)
        self.dataRPM_measured = deque(list(self.dataRPM_measured)[-datapoints:], maxlen=datapoints)
        self.dataPWM = deque(list(self.dataPWM)[-datapoints:], maxlen=datapoints)
        self.dataDT = deque(list(self.dataDT)[-datapoints:], maxlen=datapoints)
        self.dataCurrent = deque(list(self.dataCurrent)[-datapoints:], maxlen=datapoints)
        self._refresh_runtime_plots()

    def _read_float(self, le):
        """Read float from QLineEdit. Empty/invalid -> 0."""
        try:
            txt = self._coefficient_text(le).strip()
            return float(txt.replace(",", ".")) if txt else 0.0
        except Exception:
            return 0.0

    def _strip_leading_zeros(self, coefs, eps=1e-12):
        c = [float(x) for x in coefs]
        while len(c) > 1 and abs(c[0]) < eps:
            c.pop(0)
        return c

    def _read_tf_from_rlocus_inputs(self):
        # NOTE: adjust if you only have num2..num0 (remove the *_3 entries)
        num = [
            self._read_float(self.rlocus_num3),
            self._read_float(self.rlocus_num2),
            self._read_float(self.rlocus_num1),
            self._read_float(self.rlocus_num0),
        ]
        den = [
            self._read_float(self.rlocus_den3),
            self._read_float(self.rlocus_den2),
            self._read_float(self.rlocus_den1),
            self._read_float(self.rlocus_den0),
        ]

        num = self._strip_leading_zeros(num)
        den = self._strip_leading_zeros(den)

        if all(abs(x) < 1e-12 for x in num):
            raise ValueError("Numerator is all zeros.")
        if all(abs(x) < 1e-12 for x in den):
            raise ValueError("Denominator is all zeros.")

        return ctrl.TransferFunction(num, den)

    def _setup_analysis_plots(self):
        """Configure the Matplotlib canvases declared in the Designer form."""
        for canvas in (self.canvas_rl, self.canvas_step):
            canvas.figure.set_size_inches(4, 3)
            canvas.figure.set_layout_engine("tight")
        self.ax_rl = self.canvas_rl.figure.add_subplot(111)
        self.ax_step = self.canvas_step.figure.add_subplot(111)

        # Debounce plot updates (so it doesn't redraw on every keystroke instantly)
        self._analysis_timer = QTimer(self)
        self._analysis_timer.setSingleShot(True)
        self._analysis_timer.timeout.connect(self.update_analysis_plots)

        def _schedule():
            self._analysis_timer.start(200)

        # Update plots when any coefficient changes
        for le in (
            self.rlocus_num3, self.rlocus_num2, self.rlocus_num1, self.rlocus_num0,
            self.rlocus_den3, self.rlocus_den2, self.rlocus_den1, self.rlocus_den0,
        ):
            le.textChanged.connect(_schedule)

        # Optional: draw once at startup (will likely show blank/default until numbers are entered)
        self.update_analysis_plots()

    def update_analysis_plots(self):
        try:
            num_test = [
                self._read_float(self.rlocus_num3),
                self._read_float(self.rlocus_num2),
                self._read_float(self.rlocus_num1),
                self._read_float(self.rlocus_num0),
            ]
            den_test = [
                self._read_float(self.rlocus_den3),
                self._read_float(self.rlocus_den2),
                self._read_float(self.rlocus_den1),
                self._read_float(self.rlocus_den0),
            ]
            num_test = self._strip_leading_zeros(num_test)
            den_test = self._strip_leading_zeros(den_test)

            if all(abs(x) < 1e-12 for x in num_test) or all(abs(x) < 1e-12 for x in den_test):
                self.ax_rl.clear()
                self.ax_step.clear()

                for ax in (self.ax_rl, self.ax_step):
                    ax.grid(True)
                    self._strip_titles_labels(ax, keep_x=False, keep_y=False)

                self.canvas_rl.draw_idle()
                self.canvas_step.draw_idle()

                if hasattr(self, "plot_error"):
                    self.plot_error.setText("")
                return

            sys = self._read_tf_from_rlocus_inputs()

            # Clear all axes once
            self.ax_rl.clear()
            self.ax_step.clear()

            # Root locus
            self.ax_rl.clear()
            plot_root_locus_clean(sys, self.ax_rl)

            xleft, xright = self.ax_rl.get_xlim()
            self.ax_rl.set_xlim(xleft, 0)

            ylow, yhigh = self.ax_rl.get_ylim()
            ymax = max(abs(ylow), abs(yhigh), 1.0)
            self.ax_rl.set_ylim(-ymax, ymax)

            self._overlay_sgrid(self.ax_rl, show_labels=False)

            # Step response
            t = np.linspace(0, 5, 800)
            t, y = ctrl.step_response(sys, T=t)
            self._plot_time_mpl(self.ax_step, t, y)
            self.ax_step.grid(True)

            # Remove titles/labels
            self._strip_titles_labels(self.ax_rl, keep_x=False, keep_y=False)
            self._strip_titles_labels(self.ax_step, keep_x=False, keep_y=False)

            # Draw once
            self.canvas_rl.draw_idle()
            self.canvas_step.draw_idle()

            if hasattr(self, "plot_error"):
                self.plot_error.setText("")

        except Exception as e:
            if hasattr(self, "plot_error"):
                self.plot_error.setText(f"Plot error: {e}")
            else:
                print("Plot error:", e)

    def sendModeOperation(self):
        self._update_reference_controls_visibility()
        if hasattr(self, "graphWidgetRPM"):
            self.toggleupdate_parameters()

    def tabChanged(self):
        self.toggleupdate_parameters()

    def _firmware_tab_id(self):
        # Protocol IDs predate Designer page reordering; keep them stable.
        page = self.tabWidget.currentWidget()
        return {
            "tab": 0, "tab_4": 1, "tab_5": 2, "tab_6": 3,
            "tab_7": 4, "tab_3": 5, "tab_2": 6, "tab_8": 7,
            "Digital": 8, "tab_9": 9, "tab_10": 10, "tab_11": 11,
            "Continuous": 8,
        }.get(page.objectName() if page is not None else "", 0)

    def toggleupdate_parameters(self, update_runtime_config=False):
        if update_runtime_config:
            self._runtime_config_values = self._read_runtime_config_values_from_ui()

        # Collect the current values from UI elements or class variables
        StartStop = '1' if self.isRunning else '0'
        coefficient_error = None
        continuous_active = self.tabWidget.currentWidget() is self.Continuous
        try:
            if continuous_active:
                numerator, feedback = self._prepare_continuous_controller(self._read_continuous_controller())
                coefficients = [*numerator[:4], *feedback, numerator[4]]
            else:
                coefficients = [float((getattr(self, name).text() or "0").replace(",", ".")) for name in "ABCDEFGHI"]
            with np.errstate(over="ignore", under="ignore"):
                deployed = np.asarray(coefficients, dtype=np.float32)
            if not np.all(np.isfinite(deployed)) or np.any((np.asarray(coefficients) != 0) & (deployed == 0)):
                raise ValueError("Coefficients must be finite and within firmware float32 range.")
        except (ValueError, TypeError, OverflowError) as exc:
            if StartStop != '0':
                self.textBrowser.setText(f"Controller was not sent: {exc}")
                return False
            # A Stop command must remain usable while invalid fields are being edited.
            deployed = np.zeros(9)
            coefficient_error = str(exc)
        A, B, C, D, E, F, G, H, I = (f"{value:.9g}" for value in deployed)
        delay = self._numeric_text(self.delay, minimum=1, integer=True)
        # Send the data
        tiporef = int(self.automaticinput.isChecked())
        tiposenal = self.tiposenal.currentIndex()
        selected_text = self.modooperacion.currentText()
        selected_mode = self.mode_map.get(selected_text, "0")
        tiemporeferencia = self._numeric_text(self.tiemporeferencia, minimum=1, integer=True)
        amplitud = self._numeric_text(self.amplitude, integer=True)
        referenciaManual = self._numeric_text(self.reference, integer=True)
        offset = self._numeric_text(self.offset, integer=True)
        activetab = self._firmware_tab_id()
        Kp = self._numeric_text(self.Kp)
        Ki = self._numeric_text(self.Ki)
        Kd = self._numeric_text(self.Kd)
        deadzone = self._numeric_text(self.deadzone, minimum=0, maximum=255, integer=True)
        time_constant = self._numeric_text(self.time_constant, minimum=1e-6)
        PIDtype = self.PIDtype.currentIndex()
        reset_time = self._numeric_text(self.reset_time, minimum=1e-6)
        runtime_config = self._runtime_config_values
        PPR = runtime_config["PPR"]
        PWM_min = runtime_config["PWM_min"]
        PWM_max = runtime_config["PWM_max"]
        RPM_filter = runtime_config["RPM_filter"]
        if not self.SendData(StartStop, selected_mode, A, B, C, D, E, F, G, H, delay, tiemporeferencia, amplitud, referenciaManual,offset,tiposenal,activetab,Kp,Ki,Kd,deadzone,time_constant,PIDtype,tiporef,reset_time,PPR,PWM_min,PWM_max,RPM_filter,I):
            return False
        if self.gui_only:
            self.textBrowser.setText("GUI only — no hardware connected, commands are not sent")
        elif not self.controller_connected:
            self.textBrowser.setText("Controlador desconectado, esperando reconexión")
        elif selected_text == "Disabled":
            self.textBrowser.setText("System disabled, select an operation mode before starting")
        elif selected_text != "Disabled":
            self.textBrowser.setText("")
        if selected_mode == "0" or selected_mode == "1":
            pi = self.graphWidgetRPM.getPlotItem()
            pi.setLabel('left', 'Motor Input / Output', units='PWM / RPM')
        if selected_mode == "2":
            pi = self.graphWidgetRPM.getPlotItem()
            pi.setLabel('left', 'Motor Speed', units='RPM')
        if selected_mode == "3":
            pi = self.graphWidgetRPM.getPlotItem()
            pi.setLabel('left', 'Motor Position', units='degrees')
        if coefficient_error:
            self.textBrowser.append(f"Invalid coefficients; correct before Start: {coefficient_error}")
        return True

    def plot_simulation(self, t, y):
        self._plot_time_pg(self.graphWidgetRPM, t, y, pen='y', name="Simulated (Threaded)")

    def toggleStartStop(self):
        if self.isRunning:
            self._test_duration_timer.stop()
            self.isRunning = False
            self.StartStop.setText("START")
            self.StopAction()
        else:
            duration_ms = None
            if hasattr(self, "time_delimited") and self.time_delimited.isChecked():
                duration_ms = self._test_duration_ms()
                if duration_ms is None:
                    return
            self.isRunning = True
            self.StartStop.setText("STOP")
            if not self.StartAction():
                self.isRunning = False
                self.StartStop.setText("START")
                self._set_runtime_plot_interaction(enabled=True)
                return
            if duration_ms is not None:
                self._test_duration_timer.start(duration_ms)

    def _estimate_static_gain(self, u, y):
        u = np.asarray(u, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if len(u) == 0 or len(y) == 0:
            return 1.0

        mask = np.abs(u) > max(1e-9, 0.05 * float(np.max(np.abs(u))))
        if np.any(mask):
            uu = u[mask]
            yy = y[mask]
        else:
            uu = u
            yy = y

        denom = float(np.dot(uu, uu))
        if denom <= 1e-12:
            return 1.0

        gain = float(np.dot(uu, yy) / denom)
        if not np.isfinite(gain) or abs(gain) < 1e-9:
            return 1.0
        return gain

    def _identification_initial_guesses(self, numorder, denorder, time, u, y):
        gain = self._estimate_static_gain(u, y)
        duration = max(float(time[-1] - time[0]), 1e-3)
        Ts = float(np.median(np.diff(time))) if len(time) > 2 else duration
        Ts = max(Ts, 1e-4)

        tau_candidates = [
            2 * Ts, 4 * Ts, 8 * Ts, 12 * Ts,
            duration / 80.0, duration / 40.0, duration / 20.0,
            duration / 10.0, duration / 5.0,
        ]
        tau_candidates = sorted({
            float(np.clip(tau, 2 * Ts, duration))
            for tau in tau_candidates
            if np.isfinite(tau) and tau > 0
        })

        guesses = []

        def add_guess(den_tail):
            den = np.r_[1.0, np.asarray(den_tail, dtype=float)]
            if len(den) != denorder + 1:
                return
            if not np.all(np.isfinite(den)):
                return

            num = np.zeros(numorder + 1, dtype=float)
            dc_den = den[-1] if denorder > 0 else 1.0
            num[-1] = gain * dc_den
            guess = np.r_[num, den[1:]]
            if np.all(np.isfinite(guess)):
                guesses.append(guess)

        if denorder == 0:
            add_guess([])
        elif denorder == 1:
            for tau in tau_candidates:
                add_guess([1.0 / tau])
        elif denorder == 2:
            for tau in tau_candidates:
                wn = 1.0 / tau
                for zeta in (0.45, 0.7, 1.0, 1.4, 2.0):
                    add_guess([2.0 * zeta * wn, wn * wn])
        else:
            for tau in tau_candidates:
                poles = np.array([-1.0 / (tau * (1.0 + 0.35 * i)) for i in range(denorder)])
                den = np.poly(poles).real
                add_guess(den[1:])

        legacy = np.ones(numorder + 1 + denorder, dtype=float)
        if denorder > 0:
            legacy[numorder + 1:] = np.maximum(legacy[numorder + 1:], 1e-6)
        guesses.append(legacy)

        unique = []
        seen = set()
        for guess in guesses:
            key = tuple(np.round(guess, 9))
            if key not in seen:
                seen.add(key)
                unique.append(guess)
        return unique

    def _simulate_identified_model(self, theta, numorder, denorder, time, u, measured=None):
        theta = np.asarray(theta, dtype=float).ravel()
        num = theta[:numorder + 1]
        den = np.r_[1.0, theta[numorder + 1:]]
        sys_tf = ctrl.TransferFunction(num, den)
        yout = simulate_zoh_transfer(num, den, time, u, measured=measured)
        return sys_tf, np.asarray(yout, dtype=float).ravel()

    def _fit_transfer_function(self, numorder, denorder, time, u, y):
        if numorder > denorder:
            raise ValueError("Numerator order must be less than or equal to denominator order.")

        time, u, y = (np.asarray(values, dtype=float).ravel() for values in (time, u, y))
        if len(time) != len(u) or len(time) != len(y):
            raise ValueError("Identification time, input and output must have matching lengths.")
        if not all(np.all(np.isfinite(values)) for values in (time, u, y)):
            raise ValueError("Identification time, input and output must be finite.")
        if len(time) <= numorder + 1 + 2 * denorder:
            raise ValueError("Insufficient samples for the requested model order and initial conditions.")
        if len(time) < 2 or np.any(np.diff(time) <= 0):
            raise ValueError("Identification requires increasing sample times.")
        Ts = time[1] - time[0]
        if not np.allclose(np.diff(time), Ts, rtol=1e-7, atol=Ts * 1e-9):
            raise ValueError("Resample identification data to uniform sample times before fitting.")

        y_scale = max(float(np.std(y)), 1.0)
        stability_margin = 1e-6
        n_params = numorder + 1 + denorder

        lower = np.full(n_params, -np.inf, dtype=float)
        upper = np.full(n_params, np.inf, dtype=float)
        if denorder > 0:
            lower[numorder + 1:] = 1e-9

        def residual(theta):
            try:
                sys_tf, yout = self._simulate_identified_model(theta, numorder, denorder, time, u, measured=y)
                if len(yout) != len(y) or not np.all(np.isfinite(yout)):
                    return np.full(len(y) + denorder, 1e6, dtype=float)

                err = (yout - y) / y_scale
                poles = np.asarray(ctrl.poles(sys_tf), dtype=complex).ravel()
                unstable = np.maximum(poles.real + stability_margin, 0.0)
                penalty = 100.0 * unstable
                return np.r_[err, penalty]
            except Exception:
                return np.full(len(y) + denorder, 1e6, dtype=float)

        best = None
        for guess in self._identification_initial_guesses(numorder, denorder, time, u, y):
            x0 = np.asarray(guess, dtype=float)
            if len(x0) != n_params:
                continue
            x0 = np.minimum(np.maximum(x0, lower + 1e-9), upper)

            try:
                result = least_squares(
                    residual,
                    x0,
                    bounds=(lower, upper),
                    loss="linear",
                    x_scale="jac",
                    max_nfev=3000,
                )
                sys_tf, yfit = self._simulate_identified_model(result.x, numorder, denorder, time, u, measured=y)
                if not np.all(np.isfinite(yfit)):
                    continue

                poles = np.asarray(ctrl.poles(sys_tf), dtype=complex).ravel()
                stable = bool(np.all(poles.real < -stability_margin)) if denorder > 0 else True
                rmse = float(np.sqrt(np.mean((y - yfit) ** 2)))
                score = rmse if stable else rmse + 1e6

                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "rmse": rmse,
                        "stable": stable,
                        "system": sys_tf,
                        "yfit": yfit,
                    }
            except Exception:
                continue

        if best is None:
            raise RuntimeError("System identification optimizer did not converge.")

        return best["system"], best["yfit"], best["stable"]

    def identify_system(self):
        try:
            # Clear previous results/plot
            self.graphWidgetforced_response.clear()

            # ---- 1) Grab data from deques ----
            u  = np.array(self.dataPWM, dtype=float)             # input (PWM)
            y  = np.array(self.dataRPM_measured, dtype=float)    # output (RPM)
            dt = np.array(self.dataDT, dtype=float) / 1000.0     # ms -> s

            numorder = self._line_edit_int(self.numorder, default=0, minimum=0)
            denorder = self._line_edit_int(self.denorder, default=1, minimum=0)

            # Basic sanity checks
            if min(len(u), len(y), len(dt)) < 10:
                self.identificationresult.setText("Insufficient data for system identification.")
                return

            # Use the most recent overlapping window
            n = min(len(u), len(y), len(dt))
            u = u[-n:]; y = y[-n:]; dt = dt[-n:]

            # ---- 2) Build cumulative time from dt ----
            time = np.empty(n, dtype=float)
            time[0] = 0.0
            if n > 1:
                time[1:] = np.cumsum(dt[:-1])

            # ---- 3) Make time uniform for ZOH simulation ----
            # Hold PWM between samples; only the measured output is interpolated.
            if n > 2:
                Ts = float(np.median(np.diff(time)))
            else:
                Ts = float(dt.mean()) if n > 1 else 1e-3

            # If jitter is small, snap; otherwise interpolate to a uniform grid
            diffs = np.diff(time) if n > 1 else np.array([Ts])
            if np.max(np.abs(diffs - Ts)) <= 0.05 * Ts:
                # Snap to perfect grid
                N = n
                time = np.arange(N, dtype=float) * Ts
            else:
                # Interpolate to uniform grid covering the same duration
                N = int(np.round(time[-1] / Ts)) + 1
                time_uniform = np.arange(N, dtype=float) * Ts
                u = resample_held_input(time, u, time_uniform)
                y = np.interp(time_uniform, time, y)
                time = time_uniform

            # ---- 4) Stable multistart fit ----
            identified_system, y_identified, stable_fit = self._fit_transfer_function(
                numorder, denorder, time, u, y
            )
            self.identified_sys_tf = identified_system
            if not stable_fit:
                self.textBrowser.setText("Warning: best fit is not strictly stable.")

            # ---- 5) Plot response vs measured ----

            y_min = float(min(np.min(y_identified), np.min(y)))
            y_max = float(max(np.max(y_identified), np.max(y)))
            if y_max - y_min < 1e-6:
                y_min, y_max = y_min - 1.0, y_max + 1.0

            self.graphWidgetforced_response.setYRange(y_min * 0.95, y_max * 1.05)
            self._plot_time_pg(self.graphWidgetforced_response, time, y_identified, pen='y', name="Identified Response")
            self._plot_time_pg(self.graphWidgetforced_response, time, y, pen='r', name="Measured Response")

            # --- Fit metrics ---
            residuals = y - y_identified
            N = len(y)
            rmse = float(np.sqrt(np.mean(residuals**2)))
            mae  = float(np.mean(np.abs(residuals)))
            yvar = float(np.var(y))
            r2   = float(1.0 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)) if N > 1 else float('nan')
            nrmse = float(rmse / (np.sqrt(yvar) if yvar > 0 else 1.0))

            self.rmse.setText(f"{rmse:.2f}")
            self.mae.setText(f"{mae:.2f}")
            self.r2.setText(f"{r2:.2f}")
            self.nrmse.setText(f"{nrmse:.2f}")
            fit_percent = 100.0 * (1.0 - nrmse) if yvar > 0 else float('nan')
            self.identificationresult.setText(
                f"{self.tf_to_pretty_str(identified_system)}\n\n"
                f"Fit to estimation data: {fit_percent:.4f}%"
            )

            # Append to the GUI text
            #txt = self.identificationresult.toPlainText()
            #txt += f"\nFit metrics:\n  RMSE = {rmse:.3g}\n  MAE = {mae:.3g}\n  R² = {r2:.3f}\n  NRMSE = {nrmse:.3f}"
            #self.identificationresult.setText(txt)

            # (Optional) residuals plot overlay
            # self.graphWidgetforced_response.plot(time, residuals, pen='w', name='Residuals')

        except Exception as e:
            self.textBrowser.setText(f"Error during system identification: {e}")

    def update_graph(self):
        try:
            if self.serial_port is None or (hasattr(self.serial_port, "is_open") and not self.serial_port.is_open):
                self._try_reconnect_serial()
                return

            # Reset file header state when saving is disabled.
            if not self.saveValuesCheckBox.isChecked():
                self._reset_recording_state()

            # Read & process all available lines
            try:
                pending = self.serial_port.in_waiting
            except (serial.SerialException, OSError) as e:
                self._show_controller_disconnected(e)
                return

            while pending:
                try:
                    line = self.serial_port.readline(128).decode('utf-8', errors='ignore').strip()
                    pending = self.serial_port.in_waiting
                except (serial.SerialException, OSError) as e:
                    self._show_controller_disconnected(e)
                    return
                self.serial_in.setText(line)
                if line == "#CAPS DC_MOTOR B4":
                    self._supports_b4 = True
                    if self.firmware_pwm_bits is None:
                        self.firmware_pwm_bits = 8  # legacy firmware has no resolution reply
                    continue
                if line.startswith("#PWM_BITS "):
                    try:
                        bits = int(line.split()[1])
                        if 1 <= bits <= 16:
                            self.firmware_pwm_bits = bits
                    except (ValueError, IndexError):
                        pass
                    continue
                if not self.isRunning:
                    continue

                try:
                    vals = [float(x) for x in line.split()]
                except ValueError:
                    continue
                if len(vals) < 5:
                    continue

                has_current_avg = len(vals) >= 6
                sp, meas, dt_ms = vals[0], vals[1], vals[2]
                raw_current = vals[3]
                current_avg = vals[4] if has_current_avg else None
                pwm = vals[-1]

                # Append to buffers (all with the same maxlen)
                self.dataRPM_setpoint.append(sp)
                self.dataRPM_measured.append(meas)
                self.dataPWM.append(pwm)
                self.dataDT.append(dt_ms)
                # Six-field Teensy frames already contain calibrated, filtered
                # current. The five-field legacy frame contains unscaled ADC.
                self.dataCurrent.append(current_avg if has_current_avg else np.nan)

                # Save this sample if requested
                if self.saveValuesCheckBox.isChecked():
                    filepath = os.path.join("data", "data.txt")
                    os.makedirs("data", exist_ok=True)
                    if not getattr(self, "header_written", False):
                        mode_text = self.modooperacion.currentText()
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self._recording_include_dmm = self._dmm_requested()
                        self._recording_start_monotonic = monotonic()
                        dmm_columns = self._dmm_columns_header() if self._recording_include_dmm else ""
                        if has_current_avg:
                            columns = f"REF,MEAS,DT_ms,CURRENT_RAW,CURRENT_AVG,PWM{dmm_columns}"
                        else:
                            columns = f"REF,MEAS,DT_ms,CURR,PWM{dmm_columns}"
                        header = f"Mode: {mode_text}\nDate: {now}\n{columns}\n"
                        with open(filepath, 'w', newline='') as f:
                            f.write(header)
                        self.header_written = True

                    with open(filepath, 'a', newline='') as f:
                        dmm_suffix = self._dmm_row_suffix()
                        if has_current_avg:
                            f.write(
                                f"{int(sp)},{int(meas)},{dt_ms:.3f},"
                                f"{raw_current:.2f},{current_avg:.2f},{pwm:.9g}{dmm_suffix}\n"
                            )
                        else:
                            f.write(
                                f"{int(sp)},{int(meas)},{dt_ms:.3f},"
                                f"{raw_current:.2f},{pwm:.9g}{dmm_suffix}\n"
                            )

            if self.isRunning:
                self._refresh_runtime_plots()

        except Exception as e:
            print("update_graph error:", e)

    def ok_button_clicked(self):
        self.identificationresult.setText("Ok button pressed")

    def cancel_button_clicked(self):
        self.identificationresult.setText("Cancel button pressed")

    def updateDateTime(self):
        now = datetime.now()
        dateTimeString = now.strftime("%H:%M:%S\n%d-%m-%Y")
        self.dateTimeLabel.setText(dateTimeString)

    def StartAction(self):
        print("Inicio control motor")
        self._reset_recording_state()
        self._reset_runtime_plots()
        self._set_runtime_plot_interaction(enabled=False)
        return self.toggleupdate_parameters()

    def StopAction(self):
        print("Paro de control de motor")
        self.toggleupdate_parameters()
        self._set_runtime_plot_interaction(enabled=True)

    def closeEvent(self, event):
        try:
            self._test_duration_timer.stop()
            self._stop_dmm_worker(wait=True)
            self.StartStop.setText("START")
            self.isRunning = False
            if self.serial_port is not None and (not hasattr(self.serial_port, "is_open") or self.serial_port.is_open):
                self.toggleupdate_parameters()
                self.serial_port.close()
        except Exception as e:
            print(f"closeEvent serial cleanup error: {e}")
        super().closeEvent(event)

    def SendData(self, StartStop, selected_mode, A, B, C, D, E, F, G, H,delay,tiemporeferencia,amplitud,referenciaManual,offset,tiposenal,activetab,Kp,Ki,Kd,deadzone,time_constant,PIDtype,tiporef,reset_time,PPR,PWM_min,PWM_max,RPM_filter,I="0"):
        data_string=f"{StartStop},{selected_mode},{A},{B},{C},{D},{E},{F},{G},{H},{delay},{tiemporeferencia},{amplitud},{referenciaManual},{offset},{tiposenal},{activetab},{Kp},{Ki},{Kd},{deadzone},{time_constant},{PIDtype},{tiporef},{reset_time},{PPR},{PWM_min},{PWM_max},{RPM_filter}"
        try:
            b4 = float(str(I).replace(",", "."))
            values = np.array([A, B, C, D, E, F, G, H, b4], dtype=float)
            with np.errstate(over="ignore", under="ignore"):
                firmware_values = values.astype(np.float32)
            if not np.all(np.isfinite(firmware_values)) or np.any((values != 0) & (firmware_values == 0)):
                raise ValueError("Coefficients exceed firmware float32 range.")
            if b4 != 0:
                if not self.gui_only and not self._supports_b4 and str(StartStop) != "0":
                    raise ValueError("Firmware has not confirmed B4 support. Update/connect the firmware before sending I != 0.")
                if self.gui_only or self._supports_b4:
                    data_string += f",{b4:.9g}"
        except (ValueError, TypeError) as exc:
            self.textBrowser.setText(f"Controller was not sent: {exc}")
            return False
        data_bytes = (data_string + '\n').encode('utf-8')
        self.serial_out.setText(data_string)
        if self.serial_port is None or (hasattr(self.serial_port, "is_open") and not self.serial_port.is_open):
            self._try_reconnect_serial()
            return True

        try:
            self.serial_port.write(data_bytes)
            return True
        except (serial.SerialTimeoutException, serial.SerialException, OSError) as e:
            self._show_controller_disconnected(e)
            return False

    def update_slider_from_line_edit(self):
        try:
            value = int(self.reference.text())
        except ValueError:
            value = 0
        self.slider.setValue(value)
        if self.manualinput.isChecked():
            self.toggleupdate_parameters()

    def update_line_edit_from_slider(self):
        value = self.slider.value()
        self.reference.setText(str(value))
        if self.manualinput.isChecked():
            self.toggleupdate_parameters()


def main():
    # "python GUI.py --gui-only" opens the interface without looking for hardware.
    gui_only_flag = any(
        str(arg).strip().lower() in ("--gui-only", "-g", "--gui_only")
        for arg in sys.argv[1:]
    )
    app = QtWidgets.QApplication([
        arg for arg in sys.argv
        if str(arg).strip().lower() not in ("--gui-only", "-g", "--gui_only")
    ])

    port = None if gui_only_flag else auto_find_port()
    gui_only = gui_only_flag
    while True:
        if not port and not gui_only:
            sel = PortSelectDialog()
            if sel.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                sys.exit(0)
            gui_only = bool(sel.gui_only)
            port = None if gui_only else sel.selected_port()
            if not gui_only and not port:
                continue

        dlg = MyDialog(port=port, gui_only=gui_only)
        if gui_only or dlg.serial_port is not None:   # opened OK in the constructor
            dlg.show()
            sys.exit(app.exec())

        QtWidgets.QMessageBox.warning(None, "Serial",
                                      f"Couldn't open {port}. Close other apps or pick another.")
        port = None  # force dialog on next loop
        gui_only = False

if __name__ == "__main__":
    main()
