"""ZOH simulation used while estimating continuous transfer functions from PWM data."""
import numpy as np
from scipy.signal import cont2discrete, lfilter


def resample_held_input(time, values, sample_time):
    """Sample a piecewise-constant input without turning PWM steps into ramps."""
    indices = np.searchsorted(time, sample_time, side="right") - 1
    return np.asarray(values)[np.clip(indices, 0, len(values) - 1)]


def simulate_zoh_transfer(num, den, time, u, measured=None):
    """Simulate a continuous TF with held input and optional estimated initial state.

    Initial conditions are fitted by linear least squares at each candidate TF.
    Their free response spans the homogeneous denominator recurrence; this is
    equivalent to fitting the initial states in an observable filter realization.
    They belong to this data record, not to the exported transfer function.
    """
    time = np.asarray(time, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    num = np.asarray(num, dtype=float).ravel()
    den = np.asarray(den, dtype=float).ravel()
    if len(time) < 2 or len(u) != len(time):
        raise ValueError("Identification requires matching time/input arrays with at least two samples.")
    if (not np.all(np.isfinite(time)) or not np.all(np.isfinite(u))
            or not np.all(np.isfinite(num)) or not np.all(np.isfinite(den))):
        raise ValueError("Identification samples and coefficients must be finite.")
    if not len(num) or not len(den) or den[0] == 0:
        raise ValueError("The transfer function must have a nonzero leading denominator coefficient.")
    Ts = time[1] - time[0]
    if Ts <= 0 or not np.allclose(np.diff(time), Ts, rtol=1e-7, atol=Ts * 1e-9):
        raise ValueError("Identification simulation requires uniformly spaced, increasing sample times.")
    if measured is not None:
        measured = np.asarray(measured, dtype=float).ravel()
        if len(measured) != len(time) or not np.all(np.isfinite(measured)):
            raise ValueError("Measured output must be finite and match the input length.")
        if len(time) < len(den):
            raise ValueError("Insufficient samples to estimate the model's initial conditions.")

    # Remove exact leading numerator zeros before SciPy normalizes the TF.
    num = np.trim_zeros(num, trim="f")
    if len(num) > len(den):
        raise ValueError("Numerator order must be less than or equal to denominator order.")
    if len(den) == 1:
        return (num[0] / den[0] if len(num) else 0.0) * u
    # A zero numerator still has a well-defined homogeneous denominator response.
    numerator = num if len(num) else np.array([1.0])
    bz, az, _ = cont2discrete((numerator, den), Ts, method="zoh")
    yout = lfilter(bz.ravel(), az, u) if len(num) else np.zeros_like(u)
    if measured is not None:
        # For 1/A(z^-1), a unit filter state produces its impulse response;
        # the other initial filter states produce successively delayed copies.
        impulse = np.zeros(len(time))
        impulse[0] = 1.0
        free = lfilter([1.0], az, impulse)
        basis = np.column_stack([
            np.r_[np.zeros(i), free[:len(time) - i]] for i in range(len(az) - 1)
        ])
        if not np.all(np.isfinite(basis)):
            raise ValueError("The candidate model has a nonfinite initial-state response.")
        initial_state = np.linalg.lstsq(basis, measured - yout, rcond=None)[0]
        yout = yout + basis @ initial_state
    if not np.all(np.isfinite(yout)):
        raise ValueError("The candidate model has a nonfinite response.")
    return yout
