import sys
import os
import pyqtgraph as pg
import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
import control as ctrl
from datetime import datetime
from collections import deque
from PyQt6.QtCore import Qt
from PyQt6 import QtCore
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QDialogButtonBox, QButtonGroup, QMessageBox, QDialog, QLabel, QComboBox, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt6.QtCore import QTimer
from scipy.optimize import curve_fit, minimize
from scipy.signal import cont2discrete
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QTextEdit, QWidget, QFormLayout, QGroupBox, QSizePolicy, QFrame
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

KNOWN_VIDPID = {
    (0x16C0, 0x0483),  # Teensy 4.x USB-Serial (PJRC)
    (0x2341, 0x006D),  # Arduino UNO R4 WiFi
    (0x2341, 0x006C),  # Arduino UNO R4 Minima
}
PREF_DESCR = ("Teensy", "UNO R4")  # optional: bias by description text

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

class ControlPlotDialog(QDialog):
    """
    Root locus big window:
      - Left: Root locus + Step response (stacked)
      - Right: Text output + compensator builder
      - Click on locus => updates BOTH info + step plot
    """

    def __init__(self, parent, sys):
        super().__init__(parent)
        self.setWindowTitle("Root locus (large)")
        self.resize(1400, 850)

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

        # ---- Figure: 2 rows (RL on top, Step on bottom) ----
        self.fig = Figure(constrained_layout=True)
        self.ax_rl = self.fig.add_subplot(211)
        self.ax_step = self.fig.add_subplot(212)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # ---- Right side info ----
        self.info = QTextEdit()
        self.info.setReadOnly(True)

        # ---- Compensator builder widgets ----
        self.comp_title = QLabel("Compensator C(s)")
        self.comp_type = QComboBox()
        self.comp_type.addItems([
            "Real zero",
            "Real pole",
            "Complex zero pair",
            "Complex pole pair",
            "Integrator (1/s)",
            "Zero at origin (s)"
        ])

        self.comp_value = QtWidgets.QLineEdit()
        self.comp_value.setPlaceholderText("e.g. 2  ->  s = -2")
        self.comp_imag = QtWidgets.QLineEdit()
        self.comp_imag.setPlaceholderText("e.g. 3  ->  ±j3")

        self.comp_add_btn = QPushButton("Add")
        self.comp_undo_btn = QPushButton("Undo")
        self.comp_clear_btn = QPushButton("Clear")

        self.comp_display = QTextEdit()
        self.comp_display.setReadOnly(True)
        self.comp_display.setMaximumHeight(140)

        # ---- Layout ----
        main = QHBoxLayout(self)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas)
        main.addWidget(left, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        info_group = QGroupBox("Design info")
        info_group_layout = QVBoxLayout(info_group)
        info_group_layout.setContentsMargins(8, 8, 8, 8)
        info_group_layout.addWidget(self.info)

        comp_group = QGroupBox("Compensator C(s)")
        comp_group_layout = QVBoxLayout(comp_group)
        comp_group_layout.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        form.addRow("Element", self.comp_type)
        form.addRow("Location", self.comp_value)
        form.addRow("Imag", self.comp_imag)
        comp_group_layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.comp_add_btn)
        btn_row.addWidget(self.comp_undo_btn)
        btn_row.addWidget(self.comp_clear_btn)
        comp_group_layout.addLayout(btn_row)

        comp_group_layout.addWidget(self.comp_display)
        self.info.setMinimumHeight(260)
        self.comp_display.setMinimumHeight(180)

        right_layout.addWidget(info_group, 3)
        right_layout.addWidget(comp_group, 2)
        main.addWidget(right, 2)

        # Root locus picking state
        self._rlist = None
        self._klist = None
        self._hl_artist = None

        # Connect buttons
        self.comp_add_btn.clicked.connect(self._add_compensator_element)
        self.comp_undo_btn.clicked.connect(self._undo_compensator_element)
        self.comp_clear_btn.clicked.connect(self._clear_compensator)
        self.comp_type.currentIndexChanged.connect(self._update_comp_input_visibility)

        # Initial plots
        self._rebuild_compensated_system()
        self._plot_root_locus_initial()
        self._plot_step_for_K(K=0.0)
        self._set_info_idle()
        self._update_comp_display()
        self._update_comp_input_visibility()

        # Connect click picking
        self.canvas.mpl_connect("button_press_event", self._on_click_rlocus)

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

        form = self.comp_value.parentWidget().layout()
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
            self._plot_step_for_K(0.0)
            self._set_info_idle()
            self._update_comp_display()

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
        self._plot_step_for_K(0.0)
        self._set_info_idle()
        self._update_comp_display()

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
        self._plot_step_for_K(0.0)
        self._set_info_idle()
        self._update_comp_display()

    # ---------- Root locus ----------
    def _plot_root_locus_initial(self):
        self.ax_rl.clear()

        rldata = ctrl.root_locus_map(self.sys)
        self._rlist = np.asarray(rldata.loci)
        self._klist = np.asarray(rdata.gains) if False else np.asarray(rldata.gains)

        ctrl.root_locus_plot(self.sys, ax=self.ax_rl, grid=False)

        xleft, _ = self.ax_rl.get_xlim()
        self.ax_rl.set_xlim(xleft, 0.0)

        ylow, yhigh = self.ax_rl.get_ylim()
        ymax = max(abs(ylow), abs(yhigh), 1.0)
        self.ax_rl.set_ylim(-ymax, ymax)

        if hasattr(self.parent(), "_overlay_sgrid"):
            self.parent()._overlay_sgrid(self.ax_rl, show_labels=True)

        self.ax_rl.set_title("")
        self.ax_rl.set_xlabel("")
        self.ax_rl.set_ylabel("")

        self.canvas.draw_idle()

    def _on_click_rlocus(self, event):
        if event.inaxes != self.ax_rl:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._rlist is None or self._klist is None:
            return

        s_click = event.xdata + 1j * event.ydata
        diffs = np.abs(self._rlist - s_click)
        idx_flat = np.argmin(diffs)
        ik, ip = np.unravel_index(idx_flat, diffs.shape)

        K = float(self._klist[ik])
        pole = self._rlist[ik, ip]

        if self._hl_artist is not None:
            try:
                self._hl_artist.remove()
            except Exception:
                pass

        self._hl_artist = self.ax_rl.plot(
            [pole.real], [pole.imag],
            marker="o",
            markersize=8,
            markerfacecolor="red",
            markeredgecolor="red",
            linestyle="None",
            zorder=5
        )[0]

        self.canvas.draw_idle()
        self._plot_step_for_K(K)
        self._update_info_for_K(K, pole)

    # ---------- Step response ----------
    def _plot_step_for_K(self, K: float):
        self.ax_step.clear()

        L = K * self.sys
        T = ctrl.feedback(L, 1, sign=-1)

        try:
            t = np.linspace(0, 0.5, 800)
            t, y = ctrl.step_response(T, T=t)

            try:
                info = ctrl.step_info(T)
                ts = float(info.get("SettlingTime", np.nan))
                if np.isfinite(ts) and ts > 0:
                    t_final = max(0.12, 2.5 * ts)
                    t = np.linspace(0, t_final, 1000)
                    t, y = ctrl.step_response(T, T=t)
            except Exception:
                pass

            self.ax_step.plot(t, y)
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
            "Click on the root locus to select a gain K.\n"
            "The step response and parameters update instantly."
        )

    def _update_info_for_K(self, K: float, pole):
        L = K * self.sys
        T = ctrl.feedback(L, 1, sign=-1)

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
            f"Open-loop margins (L = K·C·G):\n{margin_txt}"
        )

class DiscreteControlPlotDialog(QDialog):
    """
    Discrete root locus big window:
      - Left: Root locus in z-plane + step response
      - Right: text info + gain slider + compensator builder
      - Click on locus => updates info + step plot + slider
      - Export K*C(z) to digital controller tab (A..H)
    """

    def __init__(self, parent, sysz, Ts):
        super().__init__(parent)
        self.setWindowTitle("Discrete root locus (large)")
        self.resize(1450, 900)

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

        # ---- selected locus point ----
        self.selected_k = 0.0
        self.selected_pole = None
        self.selected_index = 0

        # ---- figure ----
        self.fig = Figure(constrained_layout=True)
        self.ax_rl = self.fig.add_subplot(211)
        self.ax_step = self.fig.add_subplot(212)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # ---- right-side widgets ----
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMinimumHeight(220)
        self.info.setMaximumHeight(380)

        self.k_slider_label = QLabel("Gain K")
        self.k_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.k_slider.setMinimum(0)
        self.k_slider.setMaximum(0)

        self.k_value = QtWidgets.QLineEdit()
        self.k_value.setReadOnly(False)
        self.k_value.setPlaceholderText("Enter K and press Enter")

        self.export_btn = QPushButton("Close and export to digital controller tab")

        self.comp_type = QComboBox()
        self.comp_type.addItems([
            "Real zero",
            "Real pole",
            "Complex zero pair",
            "Complex pole pair",
            "Zero at origin",
            "Sample delay (z^-1)",
            "Integrator"
        ])

        self.comp_value = QtWidgets.QLineEdit()
        self.comp_value.setPlaceholderText("e.g. 0.6")

        self.comp_imag = QtWidgets.QLineEdit()
        self.comp_imag.setPlaceholderText("e.g. 0.2")

        self.comp_add_btn = QPushButton("Add")
        self.comp_undo_btn = QPushButton("Undo")
        self.comp_clear_btn = QPushButton("Clear")

        self.comp_display = QTextEdit()
        self.comp_display.setReadOnly(True)
        self.comp_display.setMinimumHeight(180)

        # ---- layout ----
        main = QHBoxLayout(self)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas)
        main.addWidget(left, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        info_group = QGroupBox("Design info")
        info_group_layout = QVBoxLayout(info_group)
        info_group_layout.setContentsMargins(8, 8, 8, 8)
        info_group_layout.setSpacing(8)

        # Make the text box expand, but not invade the controls below
        self.info.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.info.setMinimumHeight(220)
        self.info.setMaximumHeight(380)

        info_group_layout.addWidget(self.info, 1)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        info_group_layout.addWidget(separator)

        k_row = QHBoxLayout()
        k_row.setContentsMargins(0, 0, 0, 0)
        k_row.setSpacing(8)
        k_row.addWidget(self.k_slider_label, 0)
        k_row.addWidget(self.k_slider, 1)
        k_row.addWidget(self.k_value, 0)

        info_group_layout.addLayout(k_row, 0)

        btn_row_export = QHBoxLayout()
        btn_row_export.setContentsMargins(0, 0, 0, 0)
        btn_row_export.addWidget(self.export_btn)

        info_group_layout.addLayout(btn_row_export, 0)

        comp_group = QGroupBox("Compensator C(z)")
        comp_group_layout = QVBoxLayout(comp_group)
        comp_group_layout.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        form.addRow("Element", self.comp_type)
        form.addRow("Location", self.comp_value)
        form.addRow("Imag", self.comp_imag)
        comp_group_layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.comp_add_btn)
        btn_row.addWidget(self.comp_undo_btn)
        btn_row.addWidget(self.comp_clear_btn)
        comp_group_layout.addLayout(btn_row)

        comp_group_layout.addWidget(self.comp_display)

        right_layout.addWidget(info_group, 3)
        right_layout.addWidget(comp_group, 2)
        main.addWidget(right, 2)

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

        self.canvas.mpl_connect("button_press_event", self._on_click_rlocus)

        # ---- init ----
        self._rebuild_compensated_system()
        self._compute_root_locus_data()
        self._update_comp_input_visibility()
        self._update_comp_display()

        if self._klist is not None and len(self._klist) > 0:
            self.k_slider.blockSignals(True)
            self.k_slider.setMinimum(0)
            self.k_slider.setMaximum(len(self._klist) - 1)
            self.k_slider.setValue(0)
            self.k_slider.blockSignals(False)
            self._select_point_from_k_index(0)
        else:
            self._draw_root_locus(selected_poles=None)
            self._plot_step_for_K(0.0)
            self._set_info_idle()

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

        return ctrl.minreal(C, verbose=False)

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

        form = self.comp_value.parentWidget().layout()
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

            if self._klist is not None and len(self._klist) > 0:
                idx = min(self.selected_index, len(self._klist) - 1)
                self.k_slider.blockSignals(True)
                self.k_slider.setMinimum(0)
                self.k_slider.setMaximum(len(self._klist) - 1)
                self.k_slider.setValue(idx)
                self.k_slider.blockSignals(False)
                self._select_point_from_k_index(idx)
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

        if self._klist is not None and len(self._klist) > 0:
            idx = min(self.selected_index, len(self._klist) - 1)
            self.k_slider.blockSignals(True)
            self.k_slider.setMinimum(0)
            self.k_slider.setMaximum(len(self._klist) - 1)
            self.k_slider.setValue(idx)
            self.k_slider.blockSignals(False)
            self._select_point_from_k_index(idx)
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

        self._rebuild_compensated_system()
        self._compute_root_locus_data()
        self._update_comp_display()

        if self._klist is not None and len(self._klist) > 0:
            self.k_slider.blockSignals(True)
            self.k_slider.setMinimum(0)
            self.k_slider.setMaximum(len(self._klist) - 1)
            self.k_slider.setValue(0)
            self.k_slider.blockSignals(False)
            self._select_point_from_k_index(0)
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

    def _draw_root_locus(self, selected_poles=None):
        self.ax_rl.clear()

        ctrl.root_locus_plot(self.sys, ax=self.ax_rl, grid=False)

        self.ax_rl.set_xlim(-1.1, 1.1)
        self.ax_rl.set_ylim(-1.1, 1.1)
        self.ax_rl.set_aspect('equal', adjustable='box')

        if hasattr(self.parent(), "_overlay_zgrid"):
            self.parent()._overlay_zgrid(self.ax_rl, show_labels=True)

        self.ax_rl.set_title("")
        self.ax_rl.set_xlabel("")
        self.ax_rl.set_ylabel("")

        if selected_poles is not None:
            selected_poles = np.atleast_1d(selected_poles)
            for p in selected_poles:
                if np.isfinite(p.real) and np.isfinite(p.imag):
                    self.ax_rl.plot(
                        [p.real], [p.imag],
                        marker="s",
                        markersize=7,
                        markerfacecolor="black",
                        markeredgecolor="black",
                        linestyle="None",
                        zorder=6
                    )

        self.canvas.draw_idle()

    def _select_point_from_k_index(self, ik):
        if self._rlist is None or self._klist is None:
            return
        if ik < 0 or ik >= len(self._klist):
            return

        K = float(self._klist[ik])
        poles = np.asarray(self._rlist[ik, :]).ravel()

        mags = np.abs(poles)
        inside = np.where(mags < 1.0)[0]

        if len(inside) > 0:
            idx_main = inside[np.argmax(mags[inside])]
        else:
            idx_main = np.argmax(mags)

        pole_main = poles[idx_main]

        self.selected_index = int(ik)
        self.selected_k = K
        self.selected_pole = pole_main

        self.k_slider.blockSignals(True)
        self.k_slider.setValue(int(ik))
        self.k_slider.blockSignals(False)

        self.k_value.blockSignals(True)
        self.k_value.setText(f"{K:.6g}")
        self.k_value.blockSignals(False)

        self._draw_root_locus(selected_poles=poles)
        self._plot_step_for_K(K)
        self._update_info_for_K(K, pole_main)

    def _select_exact_k(self, K):
        K = float(K)

        L = ctrl.minreal(K * self.sys, verbose=False)
        T = ctrl.feedback(L, 1, sign=-1)
        poles = np.asarray(ctrl.poles(T)).ravel()

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

        # keep slider approximately synced to nearest sampled k
        if self._klist is not None and len(self._klist) > 0:
            karr = np.asarray(self._klist, dtype=float).ravel()
            idx_near = int(np.argmin(np.abs(karr - K)))
            self.selected_index = idx_near

            self.k_slider.blockSignals(True)
            self.k_slider.setValue(idx_near)
            self.k_slider.blockSignals(False)

        self.k_value.blockSignals(True)
        self.k_value.setText(f"{K:.6g}")
        self.k_value.blockSignals(False)

        self._draw_root_locus(selected_poles=poles)
        self._plot_step_for_K(K)
        self._update_info_for_K(K, pole_main)

    def _on_k_slider_changed(self, value):
        if self._klist is None or len(self._klist) == 0:
            return
        K = float(self._klist[int(value)])
        self._select_exact_k(K)

    def _set_k_from_value(self, K_user):
        if self._klist is None or len(self._klist) == 0:
            return

        karr = np.asarray(self._klist, dtype=float).ravel()
        idx = int(np.argmin(np.abs(karr - K_user)))
        self._select_point_from_k_index(idx)

    def _on_k_edit_finished(self):
        txt = self.k_value.text().strip()
        if not txt:
            self.k_value.blockSignals(True)
            self.k_value.setText(f"{self.selected_k:.6g}")
            self.k_value.blockSignals(False)
            return

        try:
            K_user = float(txt)
        except ValueError:
            self.k_value.blockSignals(True)
            self.k_value.setText(f"{self.selected_k:.6g}")
            self.k_value.blockSignals(False)
            return

        if not np.isfinite(K_user):
            self.k_value.blockSignals(True)
            self.k_value.setText(f"{self.selected_k:.6g}")
            self.k_value.blockSignals(False)
            return

        self._select_exact_k(K_user)

    def _on_click_rlocus(self, event):
        if event.inaxes != self.ax_rl:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._rlist is None or self._klist is None:
            return

        z_click = event.xdata + 1j * event.ydata

        diffs = np.abs(self._rlist - z_click)
        idx_flat = np.argmin(diffs)
        ik, _ = np.unravel_index(idx_flat, diffs.shape)

        K = float(self._klist[ik])
        self._select_exact_k(K)

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
                info = ctrl.step_info(T)
                ts = float(info.get("SettlingTime", np.nan))
                if np.isfinite(ts) and ts > 0:
                    t_final = max(8 * self.Ts, 2.5 * ts)
                    N = max(80, int(np.ceil(t_final / self.Ts)))
                    t = np.arange(N, dtype=float) * self.Ts
                    t, y = ctrl.step_response(T, T=t)
            except Exception:
                pass

            self.ax_step.step(t, y, where="post", label="y[k]")
            self.ax_step.axhline(1.0, linestyle="--", linewidth=1)
            self.ax_step.grid(True)
            self.ax_step.set_xlim(float(t[0]), float(t[-1]))
            self.ax_step.legend(loc="best")

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

        self.ax_step.set_title("")
        self.ax_step.set_xlabel("")
        self.ax_step.set_ylabel("")
        self.canvas.draw_idle()

    def _set_info_idle(self):
        self.info.setText(
            "Discrete root locus + step response\n\n"
            "Use the K slider or click on the locus to select a gain."
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
        Ck = ctrl.minreal(float(K) * self.comp_sys, verbose=False)
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
            f"Controller to implement K·C(z) in z^-1 form:\n"
            f"{implemented_txt}\n\n"
            f"Closed-loop step_info(T):\n{step_txt}"
        )

    # =========================================================
    # Export
    # =========================================================
    def _controller_to_zinv_coeffs(self, sys):
        """
        Convert a discrete TF written in descending powers of z into coefficients
        written in descending powers of z^-1, using the denominator order as reference.

        Example:
            0.13111 / (z - 1)
        becomes
            0.13111 z^-1 / (1 - z^-1)

        so numerator coeffs become [0, 0.13111]
        and denominator coeffs become [1, -1].
        """
        num = np.asarray(sys.num[0][0], dtype=float).ravel()
        den = np.asarray(sys.den[0][0], dtype=float).ravel()

        # Normalize so den[0] = 1 in the polynomial-in-z representation
        if abs(den[0]) > 1e-12:
            num = num / den[0]
            den = den / den[0]

        # Let denominator order define the z^-1 expansion order
        n_den = len(den) - 1
        n_num = len(num) - 1

        if n_num > n_den:
            raise ValueError("Controller is improper: numerator order exceeds denominator order.")

        # Convert to z^-1 form by dividing by z^n_den.
        # This means LEFT-padding numerator so it aligns with powers:
        # [z^m ... z^0] -> [z^0, z^-1, ..., z^-n_den]
        pad_left = n_den - n_num
        num_zinv = np.pad(num, (pad_left, 0), mode='constant')
        den_zinv = den.copy()

        # Normalize again just in case
        if abs(den_zinv[0]) > 1e-12:
            num_zinv = num_zinv / den_zinv[0]
            den_zinv = den_zinv / den_zinv[0]

        return num_zinv, den_zinv

    def _close_and_export_to_digital_tab(self):
        try:
            K = float(self.selected_k) if self.selected_k is not None else 0.0
            Ck = ctrl.minreal(K * self.comp_sys, verbose=False)

            num, den = self._controller_to_zinv_coeffs(Ck)

            # Target form:
            # H(z) = (A + B z^-1 + C z^-2 + D z^-3) /
            #        (1 - E z^-1 - F z^-2 - G z^-3 - H z^-4)

            # Pad/truncate to GUI size
            num = np.pad(num, (0, max(0, 4 - len(num))), mode='constant')[:4]
            den = np.pad(den, (0, max(0, 5 - len(den))), mode='constant')[:5]

            # Ensure den[0] = 1
            if abs(den[0]) > 1e-12:
                num = num / den[0]
                den = den / den[0]

            p = self.parent()

            # Numerator in z^-1 form
            p.A.setText(f"{num[0]:.6g}")
            p.B.setText(f"{num[1]:.6g}")
            p.C.setText(f"{num[2]:.6g}")
            p.D.setText(f"{num[3]:.6g}")

            # Denominator expected as 1 - E z^-1 - F z^-2 - ...
            p.E.setText(f"{(-den[1]):.6g}")
            p.F.setText(f"{(-den[2]):.6g}")
            p.G.setText(f"{(-den[3]):.6g}")
            p.H.setText(f"{(-den[4]):.6g}")

            # jump to "Discrete Contr"
            p.tabWidget.setCurrentIndex(8)

            p.toggleupdate_parameters()
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

        self.setWindowTitle("PID Tuner (large)")
        self.resize(1300, 850)

        flags = self.windowFlags()
        self.setWindowFlags(
            flags
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        # ---------- Figure ----------
        self.fig = Figure(constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # ---------- Right-side widgets ----------
        self.controller_type_label = QLabel("Controller type:")
        self.controller_type_combo = QComboBox()
        self.controller_type_combo.addItems(["P", "PI", "PD", "PID"])
        idx = self.controller_type_combo.findText(ctype.upper())
        if idx >= 0:
            self.controller_type_combo.setCurrentIndex(idx)

        self.speed_label = QLabel("Response speed")
        self.speed_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(0, 100)
        self.speed_slider.setValue(speed_value)
        self.speed_value_label = QLabel(str(speed_value))

        self.damping_label = QLabel("Transient behavior")
        self.damping_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.damping_slider.setRange(0, 100)
        self.damping_slider.setValue(damping_value)
        self.damping_value_label = QLabel(str(damping_value))

        self.kp_label = QLabel("Kp")
        self.kp_edit = QtWidgets.QLineEdit()
        self.kp_edit.setReadOnly(True)

        self.ki_label = QLabel("Ki")
        self.ki_edit = QtWidgets.QLineEdit()
        self.ki_edit.setReadOnly(True)

        self.kd_label = QLabel("Kd")
        self.kd_edit = QtWidgets.QLineEdit()
        self.kd_edit.setReadOnly(True)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.export_btn = QPushButton("Close and export to PID deployment tab")

        # ---------- Layout ----------
        main = QHBoxLayout(self)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        row_type = QHBoxLayout()
        row_type.addWidget(self.controller_type_label)
        row_type.addWidget(self.controller_type_combo)

        row_speed = QHBoxLayout()
        row_speed.addWidget(self.speed_label)
        row_speed.addWidget(self.speed_slider)
        row_speed.addWidget(self.speed_value_label)

        row_damping = QHBoxLayout()
        row_damping.addWidget(self.damping_label)
        row_damping.addWidget(self.damping_slider)
        row_damping.addWidget(self.damping_value_label)

        row_kp = QHBoxLayout()
        row_kp.addWidget(self.kp_label)
        row_kp.addWidget(self.kp_edit)

        row_ki = QHBoxLayout()
        row_ki.addWidget(self.ki_label)
        row_ki.addWidget(self.ki_edit)

        row_kd = QHBoxLayout()
        row_kd.addWidget(self.kd_label)
        row_kd.addWidget(self.kd_edit)

        right_layout.addLayout(row_type)
        right_layout.addLayout(row_speed)
        right_layout.addLayout(row_damping)
        right_layout.addSpacing(10)
        right_layout.addLayout(row_kp)
        right_layout.addLayout(row_ki)
        right_layout.addLayout(row_kd)
        right_layout.addSpacing(10)
        right_layout.addWidget(self.info)

        export_row = QHBoxLayout()
        export_row.addStretch()
        export_row.addWidget(self.export_btn)

        right_layout.addStretch()
        right_layout.addLayout(export_row)

        main.addWidget(left, 3)
        main.addWidget(right, 2)

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
        try:
            p = self.parent_dialog

            # Export continuous gains to the deployment/PID tab
            p.Kp.setText(f"{self.current_kp:.6g}")
            p.Ki.setText(f"{self.current_ki:.6g}")
            p.Kd.setText(f"{self.current_kd:.6g}")

            # Optional: move to the PID deployment tab if you know its index/name
            # Example:
            p.tabWidget.setCurrentIndex(5)

            # Push values through your existing flow
            p.toggleupdate_parameters()

            self.accept()

        except Exception as e:
            self.info.append(f"\nExport error: {e}")

    def _update_view(self):
        ctype = self.controller_type_combo.currentText().strip().upper()

        # Temporarily reuse parent's tuning logic
        old_type = self.parent_dialog.tuner_controller_type.currentText()
        old_speed = self.parent_dialog.tuner_speed_slider.value()
        old_damping = self.parent_dialog.tuner_damping_slider.value()

        try:
            # Mirror current dialog settings into parent logic
            idx = self.parent_dialog.tuner_controller_type.findText(ctype)
            if idx >= 0:
                self.parent_dialog.tuner_controller_type.setCurrentIndex(idx)

            self.parent_dialog.tuner_speed_slider.setValue(self.speed_slider.value())
            self.parent_dialog.tuner_damping_slider.setValue(self.damping_slider.value())

            kp, ki, kd, C, target_ts, target_os, ok = self.parent_dialog._tune_pid_from_ui(self.G, ctype)

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
            self.ax.plot(t_y, y, label="y(t)")
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

                f"Allowed overshoot ≤ {target_os:.4g} %\n\n"
                f"Closed-loop step_info(T):\n{step_txt}\n\n"
                f"Closed-loop bandwidth(T) (-3 dB): {bw_txt}"
            )

            if not ok:
                txt += "\n\nOptimizer fallback used."

            self.info.setText(txt)

        except Exception as e:
            self.ax.clear()
            self.ax.text(
                0.05, 0.5,
                f"PID tuner error: {e}",
                transform=self.ax.transAxes
            )
            self.ax.grid(True)
            self.canvas.draw_idle()
            self.info.setText(f"PID tuner error: {e}")

        finally:
            # restore parent UI state
            idx_old = self.parent_dialog.tuner_controller_type.findText(old_type)
            if idx_old >= 0:
                self.parent_dialog.tuner_controller_type.setCurrentIndex(idx_old)
            self.parent_dialog.tuner_speed_slider.setValue(old_speed)
            self.parent_dialog.tuner_damping_slider.setValue(old_damping)

class PortSelectDialog(QDialog):
    def __init__(self, initial_port=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select COM Port")
        self.initial_port = initial_port
        self._ports = []
        # Widgets
        self.label = QLabel("Choose the serial (COM) port to use:")
        self.combo = QComboBox()
        self.btn_refresh = QPushButton("Refresh")
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)

        # Layout
        top = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(self.combo)
        row.addWidget(self.btn_refresh)
        top.addWidget(self.label)
        top.addLayout(row)
        top.addWidget(self.buttons)
        self.setLayout(top)

        # Signals
        self.btn_refresh.clicked.connect(self.populate)
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

    def selected_port(self):
        idx = self.combo.currentIndex()
        if idx < 0:
            return None
        return self.combo.itemData(idx)

    def validate_and_accept(self):
        port = self.selected_port()
        if not port:
            QMessageBox.warning(self, "No port selected", "Please select a COM port or click Cancel to exit.")
            return
        self.accept()

class MyDialog(QtWidgets.QDialog):
    def __init__(self, port=None, parent=None):
        super(MyDialog, self).__init__(parent)
        uic.loadUi('QtDesignerGUI.ui', self)  # load the UI first
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
        self.setSizeGripEnabled(True)      # corner size grip on some platforms
        self.setMinimumSize(900, 600)      # pick a sane minimum so content isn’t cramped

        base_title = "Motor DC Control - By A Von Chong"
        if port:
            self.setWindowTitle(f"{base_title} — {port}")
        else:
            self.setWindowTitle(f"{base_title} — no port")
        self.serial_port = None
        if port:
            try:
                self.serial_port = serial.Serial(port, 115200, timeout=1)
            except Exception as e:
                QMessageBox.critical(self, "Serial error",
                                     f"Could not open port {port}:\n{e}")
                self.serial_port = None

        print("=== tabWidget order ===")
        for i in range(self.tabWidget.count()):
            print(i, self.tabWidget.tabText(i))

        # Initialize data lists
        self.header_written = False
        self.reference.setVisible(False)
        self.reference_label.setVisible(False)
        self.slider.setVisible(False)
        self.slider_label.setVisible(False)
        self.manualinput.setVisible(False)
        self.automaticinput.setVisible(False)

        datapoints = 1000
        self.dataRPM_setpoint = deque(maxlen=datapoints)
        self.dataRPM_measured = deque(maxlen=datapoints)
        self.dataPWM = deque(maxlen=datapoints)
        self.dataDT = deque(maxlen=datapoints)   # ms per sample
        self.tabWidget.currentChanged.connect(self.tabChanged)

        self.reference.textChanged.connect(self.update_slider_from_line_edit)
        self.slider.valueChanged.connect(self.update_line_edit_from_slider)

        # Define mode_map at the class level
        self.mode_map = {
            "Disabled": "0",
            "System Identification": "1",
            "Speed Control": "2",
            "Position Control": "3",
            "Speed - manual": "4",
            "Position - manual": "5",
        }

        # Set placeholder texts (optional, but user-friendly)
        self.snum3.setPlaceholderText("s^3")
        self.snum2.setPlaceholderText("s^2")
        self.snum1.setPlaceholderText("s^1")
        self.snum0.setPlaceholderText("s^0")
        self.sden3.setPlaceholderText("s^3")
        self.sden2.setPlaceholderText("s^2")
        self.sden1.setPlaceholderText("s^1")
        self.sden0.setPlaceholderText("s^0")
        self.sampling_time.setPlaceholderText("Ts")
        self.lqr_sampling.setPlaceholderText("Ts")
        self._sync_sampling_lineedits()

        # valor inicial compartido
        if self.sampling_time.text().strip() and not self.lqr_sampling.text().strip():
            self.lqr_sampling.setText(self.sampling_time.text())
        elif self.lqr_sampling.text().strip() and not self.sampling_time.text().strip():
            self.sampling_time.setText(self.lqr_sampling.text())

        self.sim_num3.setPlaceholderText("s^3")
        self.sim_num2.setPlaceholderText("s^2")
        self.sim_num1.setPlaceholderText("s^1")
        self.sim_num0.setPlaceholderText("s^0")
        self.sim_den3.setPlaceholderText("s^3")
        self.sim_den2.setPlaceholderText("s^2")
        self.sim_den1.setPlaceholderText("s^1")
        self.sim_den0.setPlaceholderText("s^0")

        self.cont_num3.setPlaceholderText("s^3")
        self.cont_num2.setPlaceholderText("s^2")
        self.cont_num1.setPlaceholderText("s^1")
        self.cont_num0.setPlaceholderText("s^0")
        self.plant_num3.setPlaceholderText("s^3")
        self.plant_num2.setPlaceholderText("s^2")
        self.plant_num1.setPlaceholderText("s^1")
        self.plant_num0.setPlaceholderText("s^0")

        self.cont_den3.setPlaceholderText("s^3")
        self.cont_den2.setPlaceholderText("s^2")
        self.cont_den1.setPlaceholderText("s^1")
        self.cont_den0.setPlaceholderText("s^0")
        self.plant_den3.setPlaceholderText("s^3")
        self.plant_den2.setPlaceholderText("s^2")
        self.plant_den1.setPlaceholderText("s^1")
        self.plant_den0.setPlaceholderText("s^0")
        
        self.rlocus_num3.setPlaceholderText("s^3")
        self.rlocus_num2.setPlaceholderText("s^2")
        self.rlocus_num1.setPlaceholderText("s^1")
        self.rlocus_num0.setPlaceholderText("s^0")
        self.rlocus_den3.setPlaceholderText("s^3")
        self.rlocus_den2.setPlaceholderText("s^2")
        self.rlocus_den1.setPlaceholderText("s^1")
        self.rlocus_den0.setPlaceholderText("s^0")

        self.rlocusz_num3.setPlaceholderText("z^3")
        self.rlocusz_num2.setPlaceholderText("z^2")
        self.rlocusz_num1.setPlaceholderText("z^1")
        self.rlocusz_num0.setPlaceholderText("z^0")
        self.rlocusz_den3.setPlaceholderText("z^3")
        self.rlocusz_den2.setPlaceholderText("z^2")
        self.rlocusz_den1.setPlaceholderText("z^1")
        self.rlocusz_den0.setPlaceholderText("z^0")

        self.datapoints.textChanged.connect(self.resize_deque)
        self.modooperacion.currentIndexChanged.connect(self.sendModeOperation)
        self.grupo_checkboxes = QButtonGroup(self)
        self.grupo_checkboxes.addButton(self.manualinput, 1)
        self.grupo_checkboxes.addButton(self.automaticinput, 0)
        self.grupo_checkboxes.setExclusive(True)
        self.automaticinput.setChecked(True)

        self.grupo_identificationdata = QButtonGroup(self)
        self.grupo_identificationdata.addButton(self.identdatagraph, 1)
        self.grupo_identificationdata.addButton(self.identdatafile, 0)
        self.grupo_identificationdata.setExclusive(True)
        self.identdatagraph.setChecked(True)
        self.identdatagraph.setVisible(False)
        self.identdatafile.setVisible(False)
        self.identdata.setVisible(False)
        self.tuner_ts.setVisible(False)
        self.tuner_ts_label.setVisible(False)
        self.tuner_tf.setVisible(False)
        self.tuner_tf_label.setVisible(False)
        self.tuner_loaded_plant_label.setVisible(False)
        self.tuner_status.setVisible(False)
        self.tuner_control_plot.setVisible(False)
        self.tuner_output_max.setVisible(False)
        self.tuner_output_min.setVisible(False)
        self.tuner_output_max_label.setVisible(False)
        self.tuner_output_min_label.setVisible(False)
        self.tuner_use_aw.setVisible(False)
        self.tuner_aw_time.setVisible(False)
        self.tuner_aw_label.setVisible(False)
        self._last_tuner_solution = {}

        # Define el botón de start/stop
        self.StartStop.clicked.connect(self.toggleStartStop)
        self.isRunning = False

        # Realiza la discretización
        self.discretize.clicked.connect(self.discretize_function)

        # Define botón de update parameters
        self.update_parameters.clicked.connect(self.toggleupdate_parameters)
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

        # Ensure the placeholder widget is inside a layout
        # El widget para la gráfica había que meterlo dentro de un recipiente (layout)
        # El widget se llama RPM y aquí se asegura que tiene uno asociado
        placeholderLayoutRPM = self.RPM.parentWidget().layout()
        placeholderLayoutPWM = self.PWM.parentWidget().layout()
        placeholderLayoutforced_response = self.forced_response.parentWidget().layout()
        placeholderLayoutsim_response = self.sim_response.parentWidget().layout()

        self.graphWidgetRPM = pg.PlotWidget()

        self.legendRPM  = self.graphWidgetRPM.addLegend()
        pi = self.graphWidgetRPM.getPlotItem()
        self.legendRPM = pi.addLegend()
        self.legendRPM.setParentItem(pi.vb)
        self.legendRPM.anchor(itemPos=(1,0), parentPos=(1,0), offset=(10, -10))

        pi = self.graphWidgetRPM.getPlotItem()
        pi.setLabel('left',   'Motor Input / Output', units='PWM / RPM')
        self.graphWidgetRPM.setYRange(0, 120)
        placeholderLayoutRPM.replaceWidget(self.RPM, self.graphWidgetRPM)
        self.RPM.deleteLater()

        self.graphWidgetPWM = pg.PlotWidget()
        self.legendPWM  = self.graphWidgetPWM.addLegend()
        piSETPOINT = self.graphWidgetPWM.getPlotItem()
        self.legendPWM = piSETPOINT.addLegend()
        self.legendPWM.setParentItem(piSETPOINT.vb)
        self.legendPWM.anchor(itemPos=(1,0), parentPos=(1,0), offset=(10, -10))
        pi = self.graphWidgetPWM.getPlotItem()
        pi.setLabel('left',   'uController output', units='PWM')
        pi.setLabel('bottom', 'Time',  units='s') 
        self.graphWidgetPWM.setYRange(0, 255)
        placeholderLayoutPWM.replaceWidget(self.PWM, self.graphWidgetPWM)
        self.PWM.deleteLater()

        self.graphWidgetforced_response = pg.PlotWidget()
        self.graphWidgetforced_response.setYRange(0, 1.2)
        placeholderLayoutforced_response.replaceWidget(self.forced_response, self.graphWidgetforced_response)
        self.forced_response.deleteLater()

        self.graphWidgetsim_response = pg.PlotWidget()
        self.graphWidgetsim_response.setYRange(0, 1.2)
        placeholderLayoutsim_response.replaceWidget(self.sim_response, self.graphWidgetsim_response)
        self.sim_response.deleteLater()

        # Initialize curves for fast updates
        self.graphWidgetPWM.setXLink(self.graphWidgetRPM)
        self.curve_setpoint = self.graphWidgetRPM.plot(name="Input", pen=pg.mkPen(color='b', width=3, style=QtCore.Qt.PenStyle.SolidLine))  # Blue line for setpoint
        self.curve_measured = self.graphWidgetRPM.plot(name="Output", pen=pg.mkPen(color='r', width=3, style=QtCore.Qt.PenStyle.SolidLine))  # Red line for measured
        self.curve_pwm = self.graphWidgetPWM.plot(name="PWM", pen=pg.mkPen(color='g', width=3, style=QtCore.Qt.PenStyle.SolidLine))       # Green line for PWM

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

        # Se inicializan los valores
        self.A.setText("0")
        self.B.setText("0")
        self.C.setText("0")
        self.D.setText("0")
        self.E.setText("0")
        self.F.setText("0")
        self.G.setText("0")
        self.H.setText("0")
        self.deadzone.setText("0")
        self.offset.setText("0")
        self.serial_in.setText(" ")
        self.serial_out.setText(" ")
        self.tiemporeferencia.setText("2000")
        self.amplitude.setText("150")
        self.reference.setText("150")
        self.delay.setText("20")
        self.modooperacion.setCurrentIndex(0)
        self.Kp.setText("0")
        self.Ki.setText("0")
        self.Kd.setText("0")
        self.denorder.setText("1")
        self.numorder.setText("0")
        self.datapoints.setText("200")
        self.x_scale.setText("5")
        self.time_constant.setText("0.2")
        self.reset_time.setText("0.5")
        self.tuner_output_min.setText("0")
        self.tuner_output_max.setText("255")
        self.sampling_time.setText("0.02")

        # Comienza todos los timers
        self.timer.start()
        self.timerHora.start()

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

            # Remove leading zeros to prevent errors
            num = [coef for coef in num if coef != 0] or [0]
            den = [coef for coef in den if coef != 0] or [1]

            # Create the transfer function
            sys_tf = ctrl.TransferFunction(num, den)

            # Get the user-defined time scale from x_scale
            try:
                t_max = float(self.x_scale.text()) if self.x_scale.text() else 5.0  # Default to 5 sec
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
            self.graphWidgetsim_response.plot(t, y, pen='y', name=response_type.capitalize() + " Response")
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
            txt = w.text().strip()
            num.append(float(txt) if txt else 0.0)

        for w in den_widgets:
            txt = w.text().strip()
            den.append(float(txt) if txt else 0.0)

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

        if all(abs(x) < 1e-12 for x in den):
            raise ValueError("Discrete RL denominator is all zeros.")

        Ts_txt = self.lqr_sampling.text().strip()
        if not Ts_txt:
            raise ValueError("Enter sampling time first.")

        Ts = float(Ts_txt)
        if Ts <= 0:
            raise ValueError("Sampling time must be greater than zero.")

        return ctrl.TransferFunction(num, den, Ts)

    def _setup_discrete_rlocus_tab_plots(self):
        fig_zrl = Figure(figsize=(4, 3), tight_layout=True)
        self.canvas_zrl = FigureCanvas(fig_zrl)
        self.ax_zrl = fig_zrl.add_subplot(111)

        fig_zstep = Figure(figsize=(4, 3), tight_layout=True)
        self.canvas_zstep = FigureCanvas(fig_zstep)
        self.ax_zstep = fig_zstep.add_subplot(111)

        def _replace_in_parent_layout(placeholder_widget, new_widget):
            parent = placeholder_widget.parentWidget()
            if parent is None or parent.layout() is None:
                raise RuntimeError(
                    f"Placeholder '{placeholder_widget.objectName()}' has no parent layout."
                )
            parent.layout().replaceWidget(placeholder_widget, new_widget)
            placeholder_widget.setParent(None)
            placeholder_widget.deleteLater()

        _replace_in_parent_layout(self.unitcircle_rlocus, self.canvas_zrl)
        _replace_in_parent_layout(self.response_rlocus, self.canvas_zstep)

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
            # If user hasn't entered a denominator yet, just show blank axes quietly
            den_test = [
                self._read_float(self.rlocusz_den3),
                self._read_float(self.rlocusz_den2),
                self._read_float(self.rlocusz_den1),
                self._read_float(self.rlocusz_den0),
            ]
            den_test = self._strip_leading_zeros(den_test)

            if all(abs(x) < 1e-12 for x in den_test):
                self.ax_zrl.clear()
                self.ax_zstep.clear()

                self.ax_zrl.set_xlim(-1.1, 1.1)
                self.ax_zrl.set_ylim(-1.1, 1.1)
                self.ax_zrl.set_aspect('equal', adjustable='box')
                self._overlay_zgrid(self.ax_zrl, show_labels=False)

                self.ax_zstep.grid(True)

                self.canvas_zrl.draw_idle()
                self.canvas_zstep.draw_idle()
                return

            sysz = self._read_tf_from_zrlocus_inputs()

            self.ax_zrl.clear()
            self.ax_zstep.clear()

            # Root locus in z-plane
            ctrl.root_locus_plot(sysz, ax=self.ax_zrl, grid=False)
            self.ax_zrl.set_xlim(-1.1, 1.1)
            self.ax_zrl.set_ylim(-1.1, 1.1)
            self.ax_zrl.set_aspect('equal', adjustable='box')
            self._overlay_zgrid(self.ax_zrl, show_labels=False)
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

            self.ax_zstep.plot(t, y)
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
            self.ax_zrl.set_xlim(-1.1, 1.1)
            self.ax_zrl.set_ylim(-1.1, 1.1)
            self.ax_zrl.set_aspect('equal', adjustable='box')
            self._overlay_zgrid(self.ax_zrl, show_labels=False)
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
        sysz = self._read_tf_from_zrlocus_inputs()
        Ts = float(self.lqr_sampling.text())
        dlg = DiscreteControlPlotDialog(self, sysz, Ts)
        dlg.exec()

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
        self.tuner_speed_slider.setRange(0, 100)
        self.tuner_speed_slider.setValue(50)

        self.tuner_damping_slider.setRange(0, 100)
        self.tuner_damping_slider.setValue(60)

        if hasattr(self, "tuner_speed_value"):
            self.tuner_speed_value.setText(str(self.tuner_speed_slider.value()))
        if hasattr(self, "tuner_damping_value"):
            self.tuner_damping_value.setText(str(self.tuner_damping_slider.value()))

        self.tuner_kp.setReadOnly(True)
        self.tuner_ki.setReadOnly(True)
        self.tuner_kd.setReadOnly(True)

        self._setup_pid_tuner_plots()

        self._pid_tuner_timer = QTimer(self)
        self._pid_tuner_timer.setSingleShot(True)
        self._pid_tuner_timer.timeout.connect(self._update_pid_tuner)

        def _schedule():
            self._pid_tuner_timer.start(150)

        self.tuner_speed_slider.valueChanged.connect(_schedule)
        self.tuner_damping_slider.valueChanged.connect(_schedule)
        self.tuner_controller_type.currentIndexChanged.connect(_schedule)

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
        layout_step = self.tuner_step_plot.parentWidget().layout()
        self.tuner_step_fig = Figure(constrained_layout=True)
        self.tuner_step_ax = self.tuner_step_fig.add_subplot(111)
        self.tuner_step_canvas = FigureCanvas(self.tuner_step_fig)
        layout_step.replaceWidget(self.tuner_step_plot, self.tuner_step_canvas)
        self.tuner_step_plot.deleteLater()

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

    def _clear_pid_tuner_plots(self):
        self.tuner_step_ax.clear()
        self.tuner_step_ax.grid(True)
        self.tuner_step_ax.set_title("Closed-loop step response")
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
            self._clear_pid_tuner_plots()
            print(f"Load plant error: {e}")

    def _estimate_time_scale(self, G):
        poles = ctrl.poles(G)
        stable_real_parts = [p.real for p in poles if p.real < -1e-9]

        if len(stable_real_parts) == 0:
            return 0.2

        slowest = min(abs(r) for r in stable_real_parts)
        tau = 1.0 / slowest
        return max(1e-3, tau)

    def _make_parallel_pid(self, ctype, kp, ki, kd):
        s = ctrl.TransferFunction.s
        ctype = ctype.upper()

        # Internal derivative filter for tuner simulation only
        Tf = 0.008

        if ctype == "P":
            C = ctrl.TransferFunction([kp], [1])

        elif ctype == "PI":
            C = kp + ki / s

        elif ctype == "PD":
            # Simulate as PDF internally
            Df = (kd * s) / (1 + Tf * s)
            C = kp + Df

        elif ctype == "PID":
            # Simulate as PIDF internally
            Df = (kd * s) / (1 + Tf * s)
            C = kp + ki / s + Df

        else:
            raise ValueError(f"Unsupported controller type: {ctype}")

        return ctrl.minreal(C, verbose=False)
    
    def _update_pid_tuner(self):
        try:
            G = self._get_tuner_plant()
            ctype = self.tuner_controller_type.currentText().strip().upper()

            kp, ki, kd, C, target_ts, target_os, ok = self._tune_pid_from_ui(G, ctype)

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
            self.tuner_step_ax.plot(t_y, y, label="y(t)")
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
            self._clear_pid_tuner_plots()
            print(f"PID tuner error: {e}")

    def _initial_guess_from_plant(self, G, ctype):
        try:
            kdc = float(np.real(ctrl.dcgain(G)))
        except Exception:
            kdc = 1.0

        if not np.isfinite(kdc) or abs(kdc) < 1e-9:
            kdc = 1.0

        tau = self._estimate_time_scale(G)

        kp0 = max(1e-3, 1.0 / abs(kdc))
        ki0 = max(1e-3, kp0 / max(tau, 1e-3))
        kd0 = max(1e-5, 0.1 * kp0 * tau)

        ctype = ctype.upper()
        if ctype == "P":
            return np.array([np.log10(kp0)])
        elif ctype == "PI":
            return np.array([np.log10(kp0), np.log10(ki0)])
        elif ctype == "PD":
            return np.array([np.log10(kp0), np.log10(kd0)])
        elif ctype == "PID":
            return np.array([np.log10(kp0), np.log10(ki0), np.log10(kd0)])
        else:
            raise ValueError("Invalid controller type")

    def _target_specs_from_sliders(self, G):
        tau = self._estimate_time_scale(G)

        speed = self.tuner_speed_slider.value() / 100.0
        damping = self.tuner_damping_slider.value() / 100.0

        target_ts = tau * (15.0 - 14.5 * speed)
        target_ts = max(0.03, target_ts)

        target_os = 35.0 * (1.0 - damping)
        target_os = max(0.0, min(35.0, target_os))

        return target_ts, target_os

    def _controller_params_from_x(self, x, ctype):
        vals = 10 ** np.array(x)
        ctype = ctype.upper()

        kp = ki = kd = 0.0

        if ctype == "P":
            kp = vals[0]
        elif ctype == "PI":
            kp, ki = vals
        elif ctype == "PD":
            kp, kd = vals
        elif ctype == "PID":
            kp, ki, kd = vals

        return float(kp), float(ki), float(kd)

    def _cost_pid_tuner(self, x, G, ctype, target_ts, target_os):
        try:
            kp, ki, kd = self._controller_params_from_x(x, ctype)
            C = self._make_parallel_pid(ctype, kp, ki, kd)

            T = ctrl.feedback(C * G, 1)

            t_final = max(2.0, 8.0 * target_ts)
            t = np.linspace(0, t_final, 800)

            t_y, y = ctrl.step_response(T, T=t)

            if not np.all(np.isfinite(y)):
                return 1e12

            info = ctrl.step_info(T)

            ts = float(info.get("SettlingTime", np.inf))
            os = float(info.get("Overshoot", 1e6))
            yfin = float(y[-1]) if len(y) else np.nan

            if not np.isfinite(ts):
                ts = 10 * t_final
            if not np.isfinite(os):
                os = 1e6
            if not np.isfinite(yfin):
                return 1e12

            ess = abs(1.0 - yfin)

            J = 0.0
            J += 4.0 * ((ts - target_ts) / max(target_ts, 1e-3)) ** 2
            J += 1.5 * ((max(0.0, os - target_os)) / max(5.0, target_os + 1.0)) ** 2
            J += 8.0 * ess ** 2

            # ----- normalized gain penalties -----
            x_ref = self._initial_guess_from_plant(G, ctype)
            kp_ref, ki_ref, kd_ref = self._controller_params_from_x(x_ref, ctype)

            kp_ref = max(kp_ref, 1e-6)
            ki_ref = max(ki_ref, 1e-6)
            kd_ref = max(kd_ref, 1e-6)

            ctype_u = ctype.upper()

            if ctype_u == "P":
                J += 1.365 * (kp / kp_ref) ** 2

            elif ctype_u == "PI":
                J += 0.03 * (kp / kp_ref) ** 2
                J += 0.003 * (ki / ki_ref) ** 2

                # Soft penalty against almost-zero proportional action
                kp_min_pref = 0.35 * kp_ref
                if kp < kp_min_pref:
                    J += 10.0 * ((kp_min_pref - kp) / kp_ref) ** 2

            elif ctype_u == "PD":
                J += 0.04 * (kp / kp_ref) ** 2
                J += 0.01 * (kd / kd_ref) ** 2

            elif ctype_u == "PID":
                J += 0.03 * (kp / kp_ref) ** 2
                J += 0.003 * (ki / ki_ref) ** 2
                J += 0.01 * (kd / kd_ref) ** 2

                kp_min_pref = 0.25 * kp_ref
                if kp < kp_min_pref:
                    J += 8.0 * ((kp_min_pref - kp) / kp_ref) ** 2

            if np.max(np.abs(y)) > 50:
                J += 1e6

            return float(J)

        except Exception:
            return 1e12

    def _tune_pid_from_ui(self, G, ctype):
        target_ts, target_os = self._target_specs_from_sliders(G)

        # Default initial guess from plant
        x0 = self._initial_guess_from_plant(G, ctype)

        # Warm start from previous solution for the same controller type
        if ctype in self._last_tuner_solution:
            try:
                x_prev = self._last_tuner_solution[ctype]
                if len(x_prev) == len(x0):
                    x0 = np.array(x_prev, dtype=float)
            except Exception:
                pass

        res = minimize(
            self._cost_pid_tuner,
            x0,
            args=(G, ctype, target_ts, target_os),
            method="Nelder-Mead",
            options={"maxiter": 250, "xatol": 1e-2, "fatol": 1e-3},
        )

        x_best = res.x if res.success else x0

        # Save for continuity in nearby slider positions
        self._last_tuner_solution[ctype] = np.array(x_best, dtype=float)

        kp, ki, kd = self._controller_params_from_x(x_best, ctype)
        C = self._make_parallel_pid(ctype, kp, ki, kd)

        return kp, ki, kd, C, target_ts, target_os, res.success

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
                self.equivalent_tf.setText("Error: Controller denominator is all zeros.")
                return
            if all(abs(x) < 1e-12 for x in g_den):
                self.equivalent_tf.setText("Error: Plant denominator is all zeros.")
                return

            C = ctrl.TransferFunction(c_num, c_den)
            G = ctrl.TransferFunction(g_num, g_den)

            # Series + negative unity feedback
            L = ctrl.series(C, G)
            T = ctrl.feedback(L, 1, sign=-1)  # negative feedback

            # Pretty output (and also raw coefficient lists)
            num = np.asarray(T.num[0][0], dtype=float)
            den = np.asarray(T.den[0][0], dtype=float)

            out = []
            out.append("Series reduction:")
            out.append(self.tf_to_pretty_str(L))
            out.append("\nEquivalent system:")
            out.append(self.tf_to_pretty_str(T))
            self.equivalent_tf.setText("\n".join(out))


        except Exception as e:
            self.equivalent_tf.setText(f"Error reducing block diagram: {e}")

    def tf_to_pretty_str(self, sys, var='s', digits=6, eps=1e-12):
                """Return only the transfer function fraction (no sys[..], no I/O labels)."""
                # SISO assumed
                num = np.asarray(sys.num[0][0], dtype=float)
                den = np.asarray(sys.den[0][0], dtype=float)

                # trim tiny coeffs
                num[np.abs(num) < eps] = 0.0
                den[np.abs(den) < eps] = 0.0

                def poly_str(c):
                    # c is descending powers
                    n = len(c) - 1
                    terms = []
                    for i, a in enumerate(c):
                        p = n - i
                        if abs(a) < eps:
                            continue
                        a_str = f"{a:.{digits}g}"

                        if p == 0:
                            terms.append(f"{a_str}")
                        elif p == 1:
                            if abs(a - 1.0) < eps:   terms.append(f"{var}")
                            elif abs(a + 1.0) < eps: terms.append(f"-{var}")
                            else:                    terms.append(f"{a_str} {var}")
                        else:
                            if abs(a - 1.0) < eps:   terms.append(f"{var}^{p}")
                            elif abs(a + 1.0) < eps: terms.append(f"-{var}^{p}")
                            else:                    terms.append(f"{a_str} {var}^{p}")

                    if not terms:
                        return "0"
                    s = " + ".join(terms)
                    return s.replace("+ -", "- ")

                num_s = poly_str(num)
                den_s = poly_str(den)

                # simple fraction formatting
                bar = "-" * max(len(num_s), len(den_s), 12)
                return f"{num_s}\n{bar}\n{den_s}"

    def _read_tf_from_rlocus_inputs(self):
        def _f(le):
            try:
                t = le.text().strip()
                return float(t) if t else 0.0
            except Exception:
                return 0.0

        def _strip_leading_zeros(coefs, eps=1e-12):
            c = list(map(float, coefs))
            while len(c) > 1 and abs(c[0]) < eps:
                c.pop(0)
            return c

        num = [_f(self.rlocus_num3), _f(self.rlocus_num2), _f(self.rlocus_num1), _f(self.rlocus_num0)]
        den = [_f(self.rlocus_den3), _f(self.rlocus_den2), _f(self.rlocus_den1), _f(self.rlocus_den0)]

        num = _strip_leading_zeros(num)
        den = _strip_leading_zeros(den)

        if all(abs(x) < 1e-12 for x in den):
            raise ValueError("Denominator is all zeros.")

        return ctrl.TransferFunction(num, den)
    
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
        sys = self._read_tf_from_rlocus_inputs()
        dlg = ControlPlotDialog(self, sys)
        dlg.exec()

    def open_big_zrlocus(self):
        sysz = self._read_discrete_rlocus_sys()
        Ts = float(self.sampling_time.text())
        dlg = DiscreteControlPlotDialog(self, sysz, Ts)
        dlg.exec()

    def _read_discrete_rlocus_sys(self):
        Gs = self._read_tf_from_rlocus_inputs()

        Ts_txt = self.sampling_time.text().strip()
        if not Ts_txt:
            raise ValueError("Enter sampling time Ts first.")

        Ts = float(Ts_txt)
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
        """
        import numpy as np

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xmin, xmax = xlim
        ymin, ymax = ylim

        # Only meaningful for left-half plane
        if xmin >= 0:
            return

        # Turn off rectangular grid; keep only damping grid
        ax.grid(False)

        grid_color = '0.75'
        grid_lw = 0.8
        grid_ls = ':'

        # Axes
        ax.axhline(0, color='0.65', lw=0.8, zorder=0)
        ax.axvline(0, color='0.65', lw=0.8, zorder=0)

        # Damping ratio lines only
        zetas = [0.1, 0.2, 0.3, 0.4, 0.50, 0.6, 0.7, 0.8, 0.9]
        rmax = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) * 1.2

        for z in zetas:
            theta = np.arccos(z)

            for sign in (+1, -1):
                ang = np.pi - sign * theta
                x = np.array([0.0, rmax * np.cos(ang)])
                y = np.array([0.0, rmax * np.sin(ang)])
                ax.plot(x, y, color=grid_color, lw=grid_lw, ls=grid_ls, zorder=0)

            if show_labels:
                # Put labels on the damping lines at a fixed radius from the origin
                rlabel = 0.88 * max(abs(xmin), abs(ymax), abs(ymin))
                xlab = -z * rlabel
                ylab = np.sqrt(max(0.0, 1.0 - z**2)) * rlabel

                if xmin < xlab < xmax and ymin < ylab < ymax:
                    ax.text(
                        xlab, ylab, f"{z:.2f}",
                        color='0.45', fontsize=8,
                        ha='center', va='bottom',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.15)
                    )

                if xmin < xlab < xmax and ymin < -ylab < ymax:
                    ax.text(
                        xlab, -ylab, f"{z:.2f}",
                        color='0.45', fontsize=8,
                        ha='center', va='top',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.15)
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
            ax.plot(x, y, color=grid_color, lw=grid_lw, ls=grid_ls, zorder=0)

            if show_labels and xmin < -wn < xmax:
                ax.text(
                    -wn, 0.03 * (ymax - ymin), f"{wn:g}",
                    color='0.45', fontsize=8,
                    ha='center', va='bottom',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.15)
                )

    def _overlay_zgrid(self, ax, show_labels=True):
        theta = np.linspace(0, 2 * np.pi, 500)

        # turn off rectangular grid
        ax.grid(False)

        # unit circle
        ax.plot(
            np.cos(theta), np.sin(theta),
            color='0.65', lw=1.0, ls=':'
        )

        # axes
        ax.axhline(0, color='0.75', lw=0.8)
        ax.axvline(0, color='0.75', lw=0.8)

        # optional helper radial lines
        for ang_deg in [30, 45, 60, 120, 135, 150, -30, -45, -60, -120, -135, -150]:
            ang = np.deg2rad(ang_deg)
            r = 1.1
            ax.plot(
                [0, r * np.cos(ang)],
                [0, r * np.sin(ang)],
                color='0.88',
                lw=0.6,
                ls=':'
            )

        if show_labels:
            ax.text(1.02, 0.03, "1", color='0.45', fontsize=8)
            ax.text(-1.08, 0.03, "-1", color='0.45', fontsize=8)
            ax.text(0.03, 1.02, "j", color='0.45', fontsize=8)
            ax.text(0.03, -1.08, "-j", color='0.45', fontsize=8)

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
            print(f"open_big_pid_tuner_step error: {e}")

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
            T_s   = float(self.sampling_time.text())
            method = self.method.currentText()  # e.g., 'zoh', 'foh', 'bilinear' (tustin)

            # Clean leading zeros (highest powers)
            num = _strip_leading_zeros(num)
            den = _strip_leading_zeros(den)

            # Discretize (SciPy returns TF in z^-1)
            num_z, den_z, _ = cont2discrete((num, den), T_s, method)
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

            # Use plain text if available (keeps line breaks reliably)
            if hasattr(self.discretizationresult, "setPlainText"):
                self.discretizationresult.setPlainText(result)
            else:
                self.discretizationresult.setText(result)


        except Exception as e:
            self.discretizationresult.setText(f"Error: {e}")
    
    def resize_deque(self):
        datapoints = int(self.datapoints.text())
        self.dataRPM_setpoint = deque(list(self.dataRPM_setpoint)[-datapoints:], maxlen=datapoints)
        self.dataRPM_measured = deque(list(self.dataRPM_measured)[-datapoints:], maxlen=datapoints)
        self.dataPWM = deque(list(self.dataPWM)[-datapoints:], maxlen=datapoints)
        self.dataDT = deque(list(self.dataDT)[-datapoints:], maxlen=datapoints)

    def _read_float(self, le):
        """Read float from QLineEdit. Empty/invalid -> 0."""
        try:
            txt = le.text().strip()
            return float(txt) if txt else 0.0
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

        if all(abs(x) < 1e-12 for x in den):
            raise ValueError("Denominator is all zeros.")

        return ctrl.TransferFunction(num, den)

    def _setup_analysis_plots(self):
        """
        Replace your placeholder QWidgets:
        - self.rlocus
        - self.bode_plot
        - self.time_response
        with Matplotlib canvases.
        """
        # ---------- Root locus canvas ----------
        fig_rl = Figure(figsize=(4, 3), tight_layout=True)
        self.canvas_rl = FigureCanvas(fig_rl)
        self.ax_rl = fig_rl.add_subplot(111)

        # ---------- Bode canvas (2 axes: mag + phase) ----------
        fig_bode = Figure(figsize=(4, 3), tight_layout=True)
        self.canvas_bode = FigureCanvas(fig_bode)
        self.ax_mag = fig_bode.add_subplot(211)
        self.ax_phase = fig_bode.add_subplot(212)

        # ---------- Time response canvas ----------
        fig_step = Figure(figsize=(4, 3), tight_layout=True)
        self.canvas_step = FigureCanvas(fig_step)
        self.ax_step = fig_step.add_subplot(111)

        # Helper to replace a placeholder widget that sits inside a layout
        def _replace_in_parent_layout(placeholder_widget, new_widget):
            parent = placeholder_widget.parentWidget()
            if parent is None or parent.layout() is None:
                raise RuntimeError(
                    f"Placeholder '{placeholder_widget.objectName()}' has no parent layout. "
                    "In Designer, put it inside a layout (e.g., QVBoxLayout)."
                )
            parent.layout().replaceWidget(placeholder_widget, new_widget)
            placeholder_widget.setParent(None)
            placeholder_widget.deleteLater()

        _replace_in_parent_layout(self.rlocus, self.canvas_rl)
        _replace_in_parent_layout(self.bode_plot, self.canvas_bode)
        _replace_in_parent_layout(self.time_response, self.canvas_step)

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
            sys = self._read_tf_from_rlocus_inputs()

            # Clear all axes once
            self.ax_rl.clear()
            self.ax_mag.clear()
            self.ax_phase.clear()
            self.ax_step.clear()

            # Root locus
            self.ax_rl.clear()
            ctrl.root_locus_plot(sys, ax=self.ax_rl, grid=False)

            xleft, xright = self.ax_rl.get_xlim()
            self.ax_rl.set_xlim(xleft, 0)

            ylow, yhigh = self.ax_rl.get_ylim()
            ymax = max(abs(ylow), abs(yhigh), 1.0)
            self.ax_rl.set_ylim(-ymax, ymax)

            self._overlay_sgrid(self.ax_rl, show_labels=False)

            # Bode
            omega = np.logspace(-2, 3, 600)
            ctrl.bode_plot(
                sys,
                omega=omega,
                ax=[self.ax_mag, self.ax_phase],
                dB=True,
                deg=True,
                Hz=False,
                grid=True,
            )
            self.ax_mag.tick_params(axis="x", which="both", labelbottom=False)

            # Step response
            t = np.linspace(0, 5, 800)
            t, y = ctrl.step_response(sys, T=t)
            self.ax_step.plot(t, y)
            self.ax_step.grid(True)

            # Remove titles/labels
            self._strip_titles_labels(self.ax_rl, keep_x=False, keep_y=False)
            self._strip_titles_labels(self.ax_mag, keep_x=False, keep_y=False)
            self._strip_titles_labels(self.ax_phase, keep_x=False, keep_y=False)
            self._strip_titles_labels(self.ax_step, keep_x=False, keep_y=False)

            # Draw once
            self.canvas_rl.draw_idle()
            self.canvas_bode.draw_idle()
            self.canvas_step.draw_idle()

            if hasattr(self, "plot_error"):
                self.plot_error.setText("")

        except Exception as e:
            if hasattr(self, "plot_error"):
                self.plot_error.setText(f"Plot error: {e}")
            else:
                print("Plot error:", e)

    def sendModeOperation(self):
        selected_text = self.modooperacion.currentText()
        _ = self.mode_map.get(selected_text, "0")
        # value is sent on toggleupdate_parameters

    def tabChanged(self):
        self.toggleupdate_parameters()

    def toggleupdate_parameters(self):
        # Collect the current values from UI elements or class variables
        StartStop = '0' if self.StartStop.text()=="Start" else '1'
        A = self.A.text()
        B = self.B.text()
        C = self.C.text()
        D = self.D.text()
        E = self.E.text()
        F = self.F.text()
        G = self.G.text()
        H = self.H.text()
        serial_txt = self.serial_in.text()
        reference = self.reference.text()
        delay =  self.delay.text()
        # Send the data
        tiporef = int(self.automaticinput.isChecked())
        tiposenal = self.tiposenal.currentIndex()
        selected_text = self.modooperacion.currentText()
        selected_mode = self.mode_map.get(selected_text, "0")
        tiemporeferencia = self.tiemporeferencia.text()
        amplitud = self.amplitude.text()
        referenciaManual = self.reference.text()
        offset = self.offset.text()
        activetab = self.tabWidget.currentIndex()
        Kp = self.Kp.text()
        Ki = self.Ki.text()
        Kd = self.Kd.text()
        deadzone = self.deadzone.text()
        time_constant = self.time_constant.text()
        PIDtype = self.PIDtype.currentIndex()
        reset_time = self.reset_time.text()
        self.SendData(StartStop, selected_mode, A, B, C, D, E, F, G, H, delay, tiemporeferencia, amplitud, referenciaManual,offset,tiposenal,activetab,Kp,Ki,Kd,deadzone,time_constant,PIDtype,tiporef,reset_time)
        if selected_text == "Disabled":
            self.textBrowser.setText("System disabled, select an operation mode before starting")
        if selected_text != "Disabled":
            self.textBrowser.setText("")
        if selected_mode == "0" or selected_mode == "1" :
            pi = self.graphWidgetRPM.getPlotItem()
            pi.setLabel('left', 'Motor Input / Output', units='PWM / RPM')
        if selected_mode == "2":
            pi = self.graphWidgetRPM.getPlotItem()
            pi.setLabel('left', 'Motor Speed', units='RPM')
        if selected_mode == "3":
            pi = self.graphWidgetRPM.getPlotItem()
            pi.setLabel('left', 'Motor Position', units='degrees')

    def plot_simulation(self, t, y):
        self.graphWidgetRPM.plot(t, y, pen='y', name="Simulated (Threaded)")

    def toggleStartStop(self):
        if self.isRunning:
            self.StartStop.setText("Start")
            self.StopAction()
        else:
            self.StartStop.setText("Stop")
            self.StartAction()
        self.isRunning = not self.isRunning

    def identify_system(self):
        try:
            # Clear previous results/plot
            self.graphWidgetforced_response.clear()

            # ---- 1) Grab data from deques ----
            u  = np.array(self.dataPWM, dtype=float)             # input (PWM)
            y  = np.array(self.dataRPM_measured, dtype=float)    # output (RPM)
            dt = np.array(self.dataDT, dtype=float) / 1000.0     # ms -> s

            numorder = int(self.numorder.text())
            denorder = int(self.denorder.text())

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

            # ---- 3) Make time uniform (lsim/forced_response requirement) ----
            # Use median Ts and interpolate u,y to that grid
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
                u = np.interp(time_uniform, time, u)
                y = np.interp(time_uniform, time, y)
                time = time_uniform

            # ---- 4) Model used by curve_fit ----
            def transfer_function_fit(t, *coefficients):
                # coefficients = [b0, b1, ..., a1, a2, ...]  (denominator leading 1)
                num_coeffs = coefficients[:numorder + 1]
                den_tail   = coefficients[numorder + 1:]
                den_coeffs = np.insert(den_tail, 0, 1.0)

                sys_tf = ctrl.TransferFunction(num_coeffs, den_coeffs)
                # Simulate with same input/ time vectors used in curve_fit
                _, yout = ctrl.forced_response(sys_tf, T=t, U=u)
                return yout

            # ---- 5) Initial guess and fit ----
            initial_guess = [1.0] * (numorder + 1 + denorder)
            popt, _ = curve_fit(transfer_function_fit, time, y, p0=initial_guess, maxfev=5000)

            # ---- 6) Build identified TF and show it ----
            num_fitted = popt[:numorder + 1]
            den_fitted = np.insert(popt[numorder + 1:], 0, 1.0)
            identified_system = ctrl.TransferFunction(num_fitted, den_fitted)
            self.identified_sys_tf = identified_system
            self.identificationresult.setText(f"Identified Transfer Function:\n{identified_system}")

            # ---- 7) Plot response vs measured ----
            _, y_identified = ctrl.forced_response(identified_system, T=time, U=u)

            y_min = float(min(np.min(y_identified), np.min(y)))
            y_max = float(max(np.max(y_identified), np.max(y)))
            if y_max - y_min < 1e-6:
                y_min, y_max = y_min - 1.0, y_max + 1.0

            self.graphWidgetforced_response.setYRange(y_min * 0.95, y_max * 1.05)
            self.graphWidgetforced_response.plot(time, y_identified, pen='y', name="Identified Response")
            self.graphWidgetforced_response.plot(time, y,           pen='r', name="Measured Response")

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
                return

            # Create file header once when saving is enabled
            if self.saveValuesCheckBox.isChecked() and not getattr(self, "header_written", False):
                from datetime import datetime
                mode_text = self.modooperacion.currentText()
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header = f"Mode: {mode_text}\nDate: {now}\nREF,MEAS,DT_ms,CURR,PWM\n"
                os.makedirs("data", exist_ok=True)              # <-- crea la carpeta si no existe
                filepath = os.path.join("data", "data.txt")     # <-- ruta dentro de /data
                with open(filepath, 'w', newline='') as f:
                    f.write(header)
                self.header_written = True

            # Read & process all available lines
            while self.serial_port.in_waiting:
                line = self.serial_port.readline(128).decode('utf-8', errors='ignore').strip()
                self.serial_in.setText(line)

                try:
                    vals = [float(x) for x in line.split()]
                except ValueError:
                    continue
                if len(vals) < 5:
                    continue

                sp, meas, dt_ms, curr, pwm = vals[0], vals[1], vals[2], vals[3], vals[-1]

                # Append to buffers (all with the same maxlen)
                self.dataRPM_setpoint.append(sp)
                self.dataRPM_measured.append(meas)
                self.dataPWM.append(pwm)
                self.dataDT.append(dt_ms)

                # Save this sample if requested
                if self.saveValuesCheckBox.isChecked():
                    filepath = os.path.join("data", "data.txt")
                    with open(filepath, 'a', newline='') as f:
                        f.write(f"{int(sp)},{int(meas)},{int(dt_ms)},{curr:.2f},{int(pwm)}\n")

            # ----- Plot latest window (equal-length slices) -----
            N = min(len(self.dataDT), len(self.dataRPM_setpoint), len(self.dataRPM_measured), len(self.dataPWM))
            if N == 0:
                return

            dt_s = np.asarray(self.dataDT, dtype=float)[-N:] * 1e-3
            t = np.cumsum(dt_s); t -= t[0]

            if len(t) > 0:
                xmin = float(t[0])
                xmax = float(t[-1])
                self.graphWidgetRPM.setXRange(xmin, xmax, padding=0)
                self.graphWidgetPWM.setXRange(xmin, xmax, padding=0)

            y_sp  = np.asarray(self.dataRPM_setpoint, dtype=float)[-N:]
            y_mea = np.asarray(self.dataRPM_measured, dtype=float)[-N:]
            y_pwm = np.asarray(self.dataPWM, dtype=float)[-N:]

            self.curve_setpoint.setData(t, y_sp)
            self.curve_measured.setData(t, y_mea)
            self.curve_pwm.setData(t, y_pwm)

            if "Position" in self.modooperacion.currentText():
                ymin = float(min(y_sp.min(initial=0), y_mea.min(initial=0)))
                ymax = float(max(y_sp.max(initial=0), y_mea.max(initial=0)))
                self.graphWidgetRPM.setYRange(ymin*1.1, ymax*1.1)
            else:
                self.graphWidgetRPM.setYRange(0, max(1.0, float(max(y_sp.max(initial=0), y_mea.max(initial=0)))) * 1.1)
            self.graphWidgetPWM.setYRange(0, max(1.0, float(y_pwm.max(initial=0))) * 1.1)

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
        StartStop=b'1'
        self.toggleupdate_parameters()

    def StopAction(self):
        print("Paro de control de motor")
        StartStop=b'0'
        self.toggleupdate_parameters()

    def SendData(self, StartStop, selected_mode, A, B, C, D, E, F, G, H,delay,tiemporeferencia,amplitud,referenciaManual,offset,tiposenal,activetab,Kp,Ki,Kd,deadzone,time_constant,PIDtype,tiporef,reset_time):
        data_string=f"{StartStop},{selected_mode},{A},{B},{C},{D},{E},{F},{G},{H},{delay},{tiemporeferencia},{amplitud},{referenciaManual},{offset},{tiposenal},{activetab},{Kp},{Ki},{Kd},{deadzone},{time_constant},{PIDtype},{tiporef},{reset_time}"
        data_bytes = data_string.encode('utf-8')
        self.serial_out.setText(str(data_bytes))
        data_bytes = (data_string + '\n').encode('utf-8')
        if self.serial_port is not None and (not hasattr(self.serial_port, "is_open") or self.serial_port.is_open):
            try:
                self.serial_port.write(data_bytes)
            except Exception as e:
                # If writing fails, show once
                print(f"Serial write error: {e}")

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
    app = QtWidgets.QApplication(sys.argv)

    port = auto_find_port()
    while True:
        if not port:
            sel = PortSelectDialog()
            if sel.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                sys.exit(0)
            port = sel.selected_port()

        dlg = MyDialog(port=port)
        if dlg.serial_port is not None:   # opened OK in your constructor
            dlg.setWindowTitle(f"Motor DC Control — {port} - By A Von Chong")
            dlg.show()
            sys.exit(app.exec())

        QtWidgets.QMessageBox.warning(None, "Serial",
                                      f"Couldn't open {port}. Close other apps or pick another.")
        port = None  # force dialog on next loop

if __name__ == "__main__":
    main()
