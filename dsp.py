"""Signal processing primitives.

Pure functions only. No state, no I/O, no globals mutated. Filter state for streaming is
passed in and handed back rather than hidden in a module-level variable, so the caller
owns continuity and these functions stay trivially testable on synthetic input.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

# Order-4 Butterworth. Flat passband, no ripple to distort the relative amplitude of S1
# against S2 -- ratios matter to the segmenter, so ripple would be a real cost.
FILTER_ORDER = 4

# Below 25 Hz is handling noise and DC drift from the contact mic; above 200 Hz is not
# heart sound.
HEART_BAND = (25.0, 200.0)

# Standard respiratory band is quoted as 100-1000 Hz. At fs = 2000 Hz, 1000 Hz *is*
# Nyquist and is not a realisable filter edge -- the digital design needs 0 < Wn < 1.
# 950 Hz is the usable approximation. What sits between 950 and 1000 Hz is negligible
# for lung sound and unreachable at this sample rate regardless of how the band is
# written down.
LUNG_BAND = (100.0, 950.0)

# Shannon energy smoothing. ~50 ms is long enough to merge the several oscillations
# inside one heart sound into a single envelope bump, short enough to keep S1 and S2
# separate at a plausible heart rate.
ENVELOPE_SMOOTH_MS = 50.0


def design_bandpass(
    low_hz: float, high_hz: float, sample_rate: int, order: int = FILTER_ORDER
) -> np.ndarray:
    """Second-order-sections bandpass.

    SOS rather than transfer-function (b, a) form deliberately: an order-4 bandpass is
    order 8 overall, and its polynomial coefficients span enough orders of magnitude
    that float64 root-finding loses the poles. In `ba` form this filter is numerically
    unstable and the failure is silent -- it produces plausible-looking garbage.
    """
    nyquist = sample_rate / 2.0
    if not 0.0 < low_hz < high_hz:
        raise ValueError(f"need 0 < low < high, got low={low_hz}, high={high_hz}")
    if high_hz >= nyquist:
        raise ValueError(
            f"high cutoff {high_hz} Hz is at or above Nyquist ({nyquist} Hz) for "
            f"sample_rate={sample_rate}; it is not a realisable filter edge"
        )
    return butter(order, [low_hz, high_hz], btype="band", fs=sample_rate, output="sos")


def heart_sos(sample_rate: int) -> np.ndarray:
    return design_bandpass(*HEART_BAND, sample_rate)


def lung_sos(sample_rate: int) -> np.ndarray:
    return design_bandpass(*LUNG_BAND, sample_rate)


def filter_offline(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Zero-phase filter for whole recordings.

    Runs the filter forwards then backwards, so group delay cancels exactly and an event
    stays at the sample where it happened. That matters because S1/S2 positions are the
    output; a frequency-dependent time shift would smear the systole/diastole ratio the
    segmenter reasons about.

    Not usable for streaming -- it needs the whole signal, including the future.
    """
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64)).astype(np.float32)


def make_stream_state(sos: np.ndarray, first_sample: float = 0.0) -> np.ndarray:
    """Initial delay-line state for streaming.

    Scaled to a steady-state response for a constant input of ``first_sample`` rather
    than zeroed. Zeroed state makes the filter behave as though the signal began at
    silence, producing a decaying transient at the very start of capture that the
    envelope detector reads as a large fake beat.
    """
    return sosfilt_zi(sos) * float(first_sample)


def filter_stream(
    x: np.ndarray, sos: np.ndarray, zi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Filter one frame, carrying delay-line state across the boundary.

    Returns (filtered_frame, next_zi). The caller holds the state; this stays pure.
    Causal and single-pass, so unlike filter_offline it introduces real group delay --
    that delay is constant, so relative beat timing (and therefore BPM) survives it.
    """
    y, zo = sosfilt(sos, np.asarray(x, dtype=np.float64), zi=zi)
    return y.astype(np.float32), zo


def shannon_energy(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pointwise Shannon energy, -x^2 * log(x^2).

    Chosen over plain rectification or x^2 because of how it weights amplitude: it
    emphasises the mid-range where heart sounds live while attenuating both very low
    amplitudes (noise floor) and, relatively, the very largest peaks. The practical
    effect is that a click artefact does not dominate the envelope the way it does under
    squaring.
    """
    x = np.asarray(x, dtype=np.float64)
    x2 = np.clip(x * x, eps, None)
    return -x2 * np.log(x2)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, same length out as in.

    The window is forced odd so the result is exactly centred -- an even window shifts
    the output by half a sample, and a systematic timing bias in the envelope is a
    systematic bias in every beat position derived from it.

    Edges are padded by repeating the boundary value rather than with zeros. Zero
    padding creates an artificial dip in the first and last half-window, which reads as
    low energy and can swallow a genuine beat at the very start of a recording.
    """
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return np.asarray(x, dtype=np.float64).copy()

    x = np.asarray(x, dtype=np.float64)
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def envelope(
    x: np.ndarray,
    sample_rate: int,
    smooth_ms: float = ENVELOPE_SMOOTH_MS,
    normalize: bool = True,
) -> np.ndarray:
    """Smoothed Shannon-energy envelope of an already-bandpassed signal.

    ``normalize`` standardises to zero mean and unit standard deviation, which is what
    makes a threshold in the segmenter mean the same thing across recordings made at
    different gains. Turn it off to inspect absolute energy.
    """
    window = int(round(smooth_ms / 1000.0 * sample_rate))
    env = moving_average(shannon_energy(x), window)

    if normalize:
        std = float(env.std())
        env = (env - env.mean()) / std if std > 0 else env - env.mean()

    return env.astype(np.float32)
