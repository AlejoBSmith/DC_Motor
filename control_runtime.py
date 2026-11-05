"""Deployment conventions and sampled PWM simulation shared by GUI and tests."""
import numpy as np
import control as ct
from scipy.signal import cont2discrete


def continuous_controller_coefficients(system, max_order=4):
    """Validate a realizable continuous compensator and normalize its denominator."""
    if system.dt not in (None, 0) or system.ninputs != 1 or system.noutputs != 1:
        raise ValueError("Select a continuous SISO controller.")
    b = np.asarray(system.num[0][0], dtype=float).ravel()
    a = np.asarray(system.den[0][0], dtype=float).ravel()
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)) or a[0] == 0:
        raise ValueError("Controller coefficients must be finite and the denominator must be nonzero.")
    if len(b) > len(a):
        raise ValueError("Controller is improper: add a pole/filter before exporting.")
    if len(a) > max_order + 1:
        raise ValueError(f"Firmware supports controller order <= {max_order}; no coefficients were exported.")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_b, normalized_a = b / a[0], a / a[0]
    if (not np.all(np.isfinite(normalized_b)) or not np.all(np.isfinite(normalized_a))
            or np.any((b != 0) & (normalized_b == 0)) or np.any((a != 0) & (normalized_a == 0))):
        raise ValueError("Normalized controller coefficients are not finite.")
    return normalized_b, normalized_a


def discretize_continuous_controller(system, period):
    """Standard Tustin conversion for the arbitrary continuous-controller tab."""
    deployment_period_ms(period)
    b, a = continuous_controller_coefficients(system)
    if not np.any(b) or len(a) == 1:
        discrete = ct.tf([b[0]], [1], period)
    else:
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                bz, az, _ = cont2discrete((b, a), period, method="bilinear")
        except (ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"Controller cannot be discretized at this sampling time: {exc}") from exc
        discrete = ct.tf(bz.ravel(), az, period)
    deployment_coefficients(discrete)  # check the actual firmware float32 format
    return discrete


def controller_to_zinv(system, max_order=None):
    """Convert descending powers of z to increasing delays; preserve leading delay."""
    if not ct.isdtime(system, strict=True):
        raise ValueError("Select a discrete controller before exporting.")
    num = np.asarray(system.num[0][0], dtype=float).ravel()
    den = np.asarray(system.den[0][0], dtype=float).ravel()
    if not np.all(np.isfinite(num)) or not np.all(np.isfinite(den)) or den[0] == 0:
        raise ValueError("Controller coefficients must be finite and a0 must be nonzero.")
    if len(num) > len(den):
        raise ValueError("Controller is noncausal: numerator order exceeds denominator order.")
    if max_order is not None and len(den) > max_order + 1:
        raise ValueError(f"Firmware supports controller order <= {max_order}; no coefficients were exported.")
    b = np.pad(num / den[0], (len(den) - len(num), 0))
    a = den / den[0]
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)):
        raise ValueError("Normalized controller coefficients are not finite.")
    return b, a


def deployment_coefficients(system):
    b, a = controller_to_zinv(system, max_order=4)
    b = np.pad(b, (0, 5 - len(b)))
    a = np.pad(a, (0, 5 - len(a)))
    values = np.r_[b, a[1:]]
    with np.errstate(over="ignore", under="ignore"):
        deployed = values.astype(np.float32)
    if not np.all(np.isfinite(deployed)) or np.any((values != 0) & (deployed == 0)):
        raise ValueError("Controller coefficients exceed firmware float32 range.")
    return deployed[:5], -deployed[5:]


def deployment_period_ms(period):
    milliseconds = float(period) * 1000
    if (not np.isfinite(milliseconds) or milliseconds < 1
            or milliseconds > 2147483 or abs(milliseconds - round(milliseconds)) > 1e-7):
        raise ValueError("Firmware requires a sampling period in whole milliseconds (e.g. 0.02 s = 20 ms).")
    return int(round(milliseconds))


def round_pwm(value):
    """C++ lroundf convention, including half-way values (away from zero)."""
    return np.copysign(np.floor(np.abs(value) + 0.5), value)


def quantize_pwm(value, pwm_bits=8):
    """Convert normalized 0..255 output to native counts, then back to applied units."""
    if not isinstance(pwm_bits, (int, np.integer)) or not 1 <= pwm_bits <= 16:
        raise ValueError("PWM resolution must be an integer from 1 to 16 bits.")
    full_scale = (1 << int(pwm_bits)) - 1
    normalized = np.float32(np.clip(np.float32(value), 0, 255))
    counts = int(round_pwm(np.float32(normalized * np.float32(full_scale / 255.0))))
    applied = float(np.float32(np.float32(counts) * np.float32(255.0 / full_scale)))
    return counts, applied


class DifferenceController:
    """Float32 arithmetic and saturated memory matching DifferenceController.h."""
    def __init__(self, b, feedback):
        self.b = np.asarray(b, dtype=np.float32)
        self.feedback = np.asarray(feedback, dtype=np.float32)
        self.reset()

    def reset(self):
        self.x = np.zeros(5, dtype=np.float32)
        self.y = np.zeros(4, dtype=np.float32)

    def step(self, error, mode):
        self.x[1:] = self.x[:-1].copy()
        self.x[0] = error
        value = np.float32(0)
        for coeff, state in zip(self.b, self.x):
            value = np.float32(value + np.float32(coeff * state))
        for coeff, state in zip(self.feedback, self.y):
            value = np.float32(value + np.float32(coeff * state))
        value = np.float32(np.clip(value, 0 if mode == 2 else -255, 255))
        self.y[1:] = self.y[:-1].copy()
        self.y[0] = value
        return float(value)


def simulate_pwm_loop(plant, controller, reference, *, mode=2, deadzone=0,
                      plant_input="applied", pwm_bits=8):
    """Sample output, compute error, quantize actuator, then advance the ZOH plant.

    'applied': G input is the actual PWM, as in GUI system identification.
    'effective': speed input is max(PWM - deadzone, 0), for a model explicitly
    identified in compensated input units. Position has signed PWM, no offset.
    'pwm' remains in normalized 0..255 units; 'pwm_counts' contains integer native
    counts (signed in position mode to include the direction). Eight bits remain
    the default for existing callers; the GUI supplies the configured resolution.
    Sensor quantization, RPM averaging, timing jitter and motor friction are not modeled.
    """
    if mode not in (2, 3) or plant_input not in ("applied", "effective"):
        raise ValueError("Select speed/position and applied/effective PWM input.")
    if not 0 <= deadzone <= 255 or int(deadzone) != deadzone:
        raise ValueError("Dead-zone compensation must be an integer from 0 to 255.")
    quantize_pwm(0, pwm_bits)  # validate even if the entire speed reference is zero
    if not ct.isdtime(plant, strict=True) or not np.isclose(plant.dt, controller.dt):
        raise ValueError("Plant and controller must have the same discrete sampling period.")
    deployment_period_ms(controller.dt)
    b, feedback = deployment_coefficients(controller)
    runtime = DifferenceController(b, feedback)
    ss = ct.ss(plant)
    if np.any(ss.D != 0):
        raise ValueError("PWM simulation requires a strictly proper plant (no direct feedthrough).")
    ref = np.asarray(reference, dtype=float).ravel()
    if not len(ref) or not np.all(np.isfinite(ref)):
        raise ValueError("Reference samples must be finite and nonempty.")
    state = np.zeros(ss.A.shape[0])
    output, internal, pwm, model_input = (np.zeros(len(ref)) for _ in range(4))
    pwm_counts = np.zeros(len(ref), dtype=int)
    for k, target in enumerate(ref):
        output[k] = (ss.C @ state).item()
        u = runtime.step(target - output[k], mode)
        if mode == 2:
            if target == 0:
                runtime.reset()
                u = 0.0
                applied = 0.0
                counts = 0
            else:
                # The addition is float32, just as at the firmware actuator.
                counts, applied = quantize_pwm(np.float32(u) + np.float32(deadzone), pwm_bits)
            effective = max(applied - deadzone, 0)
        else:
            counts, applied = quantize_pwm(abs(u), pwm_bits)
            if u < 0:
                counts, applied = -counts, -applied
            effective = applied
        plant_u = effective if plant_input == "effective" else applied
        internal[k], pwm[k], model_input[k] = u, applied, plant_u
        pwm_counts[k] = counts
        state = ss.A @ state + ss.B[:, 0] * plant_u
        if not np.all(np.isfinite(state)):
            raise ValueError("PWM simulation diverged; check controller gain and plant.")
    return {"output": output, "internal": internal, "pwm": pwm,
            "pwm_counts": pwm_counts, "plant_input": model_input}
