"""Prove the audio I/O path works before any DSP is written.

Synthesises a phonocardiogram-like signal at a known heart rate, writes it to WAV at
the project sample rate, reads it back, checks the round trip, and plots it.

Two reasons this exists at hour zero:

1. An I/O or dtype bug found at hour four is fatal. This surfaces it now.
2. Every later module needs a signal whose correct answer is already known. CinC gives
   real audio but this gives ground truth: exact S1/S2 sample positions and an exact
   heart rate to check beat detection against.

Run:  python tools/check_io.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display in CI or a container

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Project-wide constant. Everything runs at this rate; CinC 2016 ships at it, so
# there is no resampling anywhere on the training path.
SAMPLE_RATE = 2000

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def _burst(
    n_samples: int, freq_hz: float, duration_s: float, sample_rate: int, rng: np.random.Generator
) -> np.ndarray:
    """A Gaussian-windowed sinusoid — a crude but adequate stand-in for a heart sound.

    Real S1/S2 are broadband transients, not tones. A windowed sinusoid is close enough
    to exercise a bandpass and an envelope detector, and its centre frequency and
    duration are known exactly, which is the whole point.
    """
    n = int(round(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    # Window sigma set so the burst decays to near zero inside its stated duration.
    window = np.exp(-0.5 * ((t - duration_s / 2) / (duration_s / 6)) ** 2)
    phase = rng.uniform(0, 2 * np.pi)
    return (np.sin(2 * np.pi * freq_hz * t + phase) * window).astype(np.float64)


def synth_pcg(
    duration_s: float = 10.0,
    bpm: float = 72.0,
    systole_s: float = 0.32,
    sample_rate: int = SAMPLE_RATE,
    noise_rms: float = 0.01,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic PCG. Returns (signal, s1_indices, s2_indices).

    The timing asymmetry is the property under test: at 72 bpm the beat period is
    833 ms, so systole (S1->S2) is 320 ms and diastole (S2->S1) is 513 ms. Diastole is
    the longer gap, which is what lets a deterministic segmenter decide which peak is
    S1 without any labels.

    S1 is modelled lower and longer than S2 (~50 Hz / 100 ms vs ~75 Hz / 60 ms), and
    louder, which is what you hear at the apex.
    """
    rng = np.random.default_rng(seed)

    n_total = int(round(duration_s * sample_rate))
    signal = np.zeros(n_total, dtype=np.float64)

    beat_period_s = 60.0 / bpm
    if systole_s >= beat_period_s:
        raise ValueError(f"systole {systole_s}s does not fit in a {beat_period_s:.3f}s beat")

    s1_idx: list[int] = []
    s2_idx: list[int] = []

    n_beats = int(duration_s / beat_period_s)
    for beat in range(n_beats):
        t0 = beat * beat_period_s

        for offset_s, freq, dur, amp, sink in (
            (t0, 50.0, 0.10, 1.00, s1_idx),
            (t0 + systole_s, 75.0, 0.06, 0.65, s2_idx),
        ):
            start = int(round(offset_s * sample_rate))
            burst = _burst(n_total, freq, dur, sample_rate, rng) * amp
            end = min(start + burst.size, n_total)
            if start >= n_total:
                continue
            signal[start:end] += burst[: end - start]
            # Label the burst's energy centre, not its onset: that is what an envelope
            # peak detector will actually find.
            sink.append(start + burst.size // 2)

    signal += rng.normal(0.0, noise_rms, n_total)

    # Normalise into [-1, 1] with headroom. The contract says float32 in [-1, 1].
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.95 * signal / peak

    return signal.astype(np.float32), np.asarray(s1_idx), np.asarray(s2_idx)


def _check_roundtrip(signal: np.ndarray, path: Path, subtype: str, tol: float) -> np.ndarray:
    """Write, read back, and assert the signal survived. Returns what came back."""
    sf.write(path, signal, SAMPLE_RATE, subtype=subtype)
    read_back, rate = sf.read(path, dtype="float32", always_2d=False)

    assert rate == SAMPLE_RATE, f"{subtype}: sample rate changed {SAMPLE_RATE} -> {rate}"
    assert read_back.dtype == np.float32, f"{subtype}: dtype is {read_back.dtype}, want float32"
    assert read_back.ndim == 1, f"{subtype}: got {read_back.ndim} dims, want mono"
    assert read_back.shape == signal.shape, f"{subtype}: length changed"

    err = float(np.max(np.abs(read_back - signal)))
    assert err <= tol, f"{subtype}: round trip error {err:.2e} exceeds {tol:.2e}"
    print(f"  {subtype:<6} round trip ok  (max abs error {err:.2e}, tol {tol:.0e})")
    return read_back


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)

    bpm = 72.0
    duration_s = 10.0
    signal, s1_idx, s2_idx = synth_pcg(duration_s=duration_s, bpm=bpm)

    print(f"synthesised {duration_s:.0f}s at {SAMPLE_RATE} Hz, {bpm:.0f} bpm")
    print(f"  {signal.size} samples, dtype {signal.dtype}, peak {np.max(np.abs(signal)):.3f}")
    print(f"  {s1_idx.size} S1 events, {s2_idx.size} S2 events")

    assert signal.dtype == np.float32
    assert np.max(np.abs(signal)) <= 1.0, "signal escaped [-1, 1]"

    # Ground truth check: the interval between consecutive S1s must equal the beat
    # period. If this fails the synthesiser is wrong and nothing downstream can be
    # trusted to validate against it.
    expected_period = 60.0 / bpm * SAMPLE_RATE
    s1_intervals = np.diff(s1_idx)
    assert np.allclose(s1_intervals, expected_period, atol=1.0), (
        f"S1 spacing {s1_intervals[:3]} != expected {expected_period:.1f} samples"
    )
    print(f"  S1-S1 interval {s1_intervals[0]} samples == {expected_period:.1f} expected")

    # Systole must be shorter than diastole, or the deterministic S1/S2 rule is void.
    systole = s2_idx[0] - s1_idx[0]
    diastole = s1_idx[1] - s2_idx[0]
    assert systole < diastole, f"systole {systole} !< diastole {diastole}"
    print(f"  systole {systole} < diastole {diastole} samples — asymmetry present")

    print("round trip:")
    # FLOAT is lossless, so demand exactness. PCM_16 is what CinC actually ships, so
    # test it too and allow one quantisation step (2 / 2**16).
    _check_roundtrip(signal, OUT_DIR / "check_io_float.wav", "FLOAT", tol=0.0)
    wav_path = OUT_DIR / "check_io_pcm16.wav"
    read_back = _check_roundtrip(signal, wav_path, "PCM_16", tol=2.0 / 2**15)

    # Plot the first three beats — enough to see structure, not so many it is a smear.
    n_show = int(round(3 * 60.0 / bpm * SAMPLE_RATE))
    t = np.arange(n_show) / SAMPLE_RATE

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(t, read_back[:n_show], linewidth=0.8, color="#1f77b4")
    for i, idx in enumerate(s1_idx[s1_idx < n_show]):
        ax.axvline(idx / SAMPLE_RATE, color="#d62728", alpha=0.7, linewidth=1.2,
                   label="S1 (ground truth)" if i == 0 else None)
    for i, idx in enumerate(s2_idx[s2_idx < n_show]):
        ax.axvline(idx / SAMPLE_RATE, color="#2ca02c", alpha=0.7, linewidth=1.2,
                   linestyle="--", label="S2 (ground truth)" if i == 0 else None)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title(f"Synthetic PCG — {bpm:.0f} bpm @ {SAMPLE_RATE} Hz (read back from WAV)")
    ax.legend(loc="upper right", fontsize=8)
    ax.margins(x=0)
    fig.tight_layout()

    plot_path = OUT_DIR / "check_io.png"
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)

    print(f"wrote {wav_path}")
    print(f"wrote {plot_path}")
    print("\nI/O path ok.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
