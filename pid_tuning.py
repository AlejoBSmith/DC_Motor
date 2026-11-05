"""Deterministic continuous PID tuning against a reference response.

The sliders select a second-order reference model. With the filtered parallel
PID bases B=[1, 1/s, s/(1+Tf*s)], its model-matching residual is linear in gains:

    (1 - Tref) * G * B * gains - Tref

Nonnegative, dimensionless gains and a positive quadratic regularizer give a
strictly convex problem with a unique solution, independent of previous calls.
P/PD also penalize their actual steady-state error; this term is convex too.
Matching a reference is an approximation, not a guaranteed overshoot bound.
"""
from dataclasses import dataclass

import control as ct
import numpy as np
from scipy.optimize import brentq, lsq_linear


@dataclass(frozen=True)
class PIDTuningResult:
    kp: float
    ki: float
    kd: float
    controller: object
    target_settling_time: float
    target_overshoot: float


def _plant_properties(plant):
    if plant.ninputs != 1 or plant.noutputs != 1:
        raise ValueError("PID tuning requires a SISO plant.")
    if ct.isdtime(plant, strict=True):
        raise ValueError("PID tuning requires a continuous-time plant.")
    plant = ct.tf(plant)
    num = np.asarray(plant.num[0][0], dtype=float)
    den = np.asarray(plant.den[0][0], dtype=float)
    if not np.all(np.isfinite(num)) or not np.all(np.isfinite(den)):
        raise ValueError("Plant coefficients must be finite.")
    if not np.any(num) or len(num) > len(den):
        raise ValueError("PID tuning requires a nonzero, proper plant.")
    poles = ct.poles(plant)
    if np.any(poles.real > 1e-12 * max(1., np.max(np.abs(poles), initial=0.))):
        raise ValueError("Automatic PID tuning requires a stable plant or plant integrators.")
    stable_rates = -poles.real[poles.real < -1e-12]
    tau = 1. / float(np.min(stable_rates)) if stable_rates.size else .2
    dc = float(np.real(ct.dcgain(plant)))
    finite_dc = np.isfinite(dc)
    if finite_dc:
        if dc == 0:
            raise ValueError("The plant has zero DC gain; use root-locus design for this plant.")
        gain_scale = dc
    else:
        # A position plant can have poles at zero. Normalize its magnitude at
        # 1/tau, retaining the low-frequency sign of the nonzero coefficients.
        direction = np.sign(num[np.nonzero(num)[0][-1]] / den[np.nonzero(den)[0][-1]])
        gain_scale = direction * abs(ct.evalfr(plant, 1j / tau))
    if not np.isfinite(gain_scale) or gain_scale == 0 or not np.isfinite(tau):
        raise ValueError("Could not determine a finite plant scale.")
    return plant, num, den, tau, gain_scale, finite_dc


def _p_pd_solution(matrix, target, count):
    """Solve the convex quadratic plus 8/(1+xP)^2 without a local search.

    xP=Kp*G(0), so the extra term is the exact squared DC tracking error
    for a P or filtered PD controller. Eliminate D at its conditional quadratic
    optimum, then find the zero of the strictly increasing P derivative.
    """
    hessian = matrix.T @ matrix / count
    gradient = matrix.T @ target / count

    def derivative_gain(p):
        if len(gradient) == 1:
            return 0.
        return max(0., float((gradient[1] - hessian[1, 0] * p) / hessian[1, 1]))

    def slope(p):
        cross = hessian[0, 1] * derivative_gain(p) if len(gradient) > 1 else 0.
        return 2 * (hessian[0, 0] * p + cross - gradient[0]) - 16 / (1 + p) ** 3

    if slope(0.) >= 0:
        proportional = 0.
    else:
        upper = 1.
        for _ in range(80):
            if slope(upper) >= 0:
                break
            upper *= 2
        else:
            raise ValueError("Could not bracket the PID gain solution.")
        proportional = brentq(slope, 0., upper, xtol=1e-12, rtol=1e-12)
    return np.array([proportional] if len(gradient) == 1 else
                    [proportional, derivative_gain(proportional)])


def tune_pid(plant, controller_type, speed, transient, derivative_filter_time):
    """Tune a filtered parallel P/PI/PD/PID; never read GUI or tuning history.

    Speed 0..100 selects a reference settling time of 15*tau..0.5*tau.
    Transient 0..100 selects reference overshoot 35%..0% (critical damping).
    All calculations are continuous-time, including the configured D filter.
    The selected structure may not reproduce the reference exactly. Unstable
    results are rejected rather than exported or replaced with previous gains.
    """
    ids = {"P": [0], "PI": [0, 1], "PD": [0, 2], "PID": [0, 1, 2]}.get(controller_type.upper())
    if ids is None:
        raise ValueError("Unsupported PID controller type.")
    if not np.all(np.isfinite([speed, transient, derivative_filter_time])):
        raise ValueError("Tuner settings must be finite.")
    if not (0 <= speed <= 100 and 0 <= transient <= 100) or derivative_filter_time <= 0:
        raise ValueError("Sliders must be in 0..100 and the derivative filter must be positive.")
    plant, num, den, tau, kdc, finite_dc = _plant_properties(plant)
    target_ts = tau * (15. - 14.5 * float(speed) / 100.)
    target_os = 35. * (1. - float(transient) / 100.)
    log_os = np.log(target_os / 100.) if target_os else 0.
    zeta = -log_os / np.sqrt(np.pi ** 2 + log_os ** 2) if target_os else 1.
    omega_n = 4. / (zeta * target_ts)
    q = np.logspace(-1, 1, 100)
    s = 1j * q * omega_n
    gp = np.polyval(num, s) / np.polyval(den, s)
    reference = omega_n ** 2 / (s ** 2 + 2 * zeta * omega_n * s + omega_n ** 2)
    bases = np.column_stack([np.ones_like(s), 1 / s, s / (1 + derivative_filter_time * s)])[:, ids]
    scales = np.array([1 / kdc, 1 / (kdc * tau), tau / kdc])[ids]
    model = (1 - reference)[:, None] * gp[:, None] * bases * scales
    weights = 1 / (1 + (q / 4) ** 2)
    matrix = np.vstack([model.real * weights[:, None], model.imag * weights[:, None],
                        np.sqrt(.005) * np.eye(len(ids))])
    target = np.r_[reference.real * weights, reference.imag * weights, np.zeros(len(ids))]
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError("The plant frequency response could not be evaluated.")

    if controller_type.upper() in ("P", "PD") and finite_dc:
        normalized = _p_pd_solution(matrix, target, len(q))
    else:
        result = lsq_linear(matrix, target, bounds=(0., np.inf), method="bvls",
                            tol=1e-12, max_iter=100)
        if not result.success:
            raise ValueError("PID tuning did not converge; no controller was exported.")
        normalized = result.x.copy()
    # Numerical lower-bound remnants must not create almost-cancelled poles at
    # zero, which can give step_info an effectively infinite simulation horizon.
    normalized[normalized < 1e-10] = 0.
    gains = np.zeros(3)
    gains[ids] = normalized * scales
    if not np.all(np.isfinite(gains)) or not np.any(gains):
        raise ValueError("No finite, nonzero PID matches the selected response target.")
    kp, ki, kd = (float(value) for value in gains)
    controller = ct.tf([kp], [1])
    if ki:
        controller += ct.tf([ki], [1, 0])
    if kd:
        controller += ct.tf([kd, 0], [derivative_filter_time, 1])
    closed = ct.feedback(controller * plant, 1)
    if np.any(ct.poles(closed).real >= 0):
        raise ValueError("The selected response target gives an unstable PID. Reduce response speed or change controller type.")
    return PIDTuningResult(kp, ki, kd, controller, float(target_ts), float(target_os))
