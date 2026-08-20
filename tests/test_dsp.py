"""dsp.py against signals whose correct answer is known in advance.

Pure tones, chirps and impulses only. A filter bug found on real audio at hour four is
fatal, and real audio cannot tell you whether a wrong answer came from the filter or
from the recording.

Run:  python tests/test_dsp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import chirp, sosfilt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import dsp  # noqa: E402
from check_io import synth_pcg  # noqa: E402

FS = 2000
PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    PASSED += 1
    print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))


def tone(freq: float, seconds: float = 4.0, fs: int = FS) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return np.sin(2 * np.pi * freq * t)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def test_design_guards() -> None:
    print("design_bandpass guards")

    # The literal 100-1000 Hz lung band is unrealisable at fs=2000: 1000 Hz is Nyquist.
    # This must fail loudly rather than produce a filter that quietly does nothing.
    try:
        dsp.design_bandpass(100, 1000, 2000)
        raise AssertionError("expected ValueError for cutoff at Nyquist")
    except ValueError as exc:
        check("rejects cutoff at Nyquist", "Nyquist" in str(exc))

    try:
        dsp.design_bandpass(200, 25, 2000)
        raise AssertionError("expected ValueError for inverted band")
    except ValueError:
        check("rejects inverted band", True)

    sos = dsp.heart_sos(FS)
    check("heart filter is SOS", sos.ndim == 2 and sos.shape[1] == 6, f"shape {sos.shape}")
    check("order 4 bandpass -> 4 sections", sos.shape[0] == 4)


def test_passband_and_stopband() -> None:
    """Gain at known frequencies. The single most load-bearing property of the filter."""
    print("heart bandpass 25-200 Hz, gain at known tones")
    sos = dsp.heart_sos(FS)
    edge = 400  # discard filter edge transients before measuring

    for freq, lo, hi, label in [
        (100.0, 0.90, 1.10, "passband centre"),
        (50.0, 0.85, 1.10, "passband"),
        (5.0, 0.0, 0.05, "below band"),
        (500.0, 0.0, 0.05, "above band"),
        (900.0, 0.0, 0.01, "far above band"),
    ]:
        x = tone(freq)
        y = dsp.filter_offline(x, sos)
        gain = rms(y[edge:-edge]) / rms(x[edge:-edge])
        db = 20 * np.log10(max(gain, 1e-12))
        check(f"{freq:>5.0f} Hz {label}", lo <= gain <= hi, f"gain {gain:.4f} = {db:+.1f} dB")


def test_zero_phase() -> None:
    """filter_offline must not move an event in time."""
    print("filter_offline is zero-phase")
    sos = dsp.heart_sos(FS)

    # A symmetric burst centred in the buffer. Zero-phase filtering preserves symmetry;
    # any group delay would shift the energy centroid off centre.
    n = 4000
    x = np.zeros(n)
    centre = n // 2
    t = (np.arange(n) - centre) / FS
    x += np.exp(-0.5 * (t / 0.02) ** 2) * np.sin(2 * np.pi * 100 * t)

    y = dsp.filter_offline(x, sos)
    energy = np.square(y.astype(np.float64))
    centroid = float((np.arange(n) * energy).sum() / energy.sum())
    check("energy centroid unmoved", abs(centroid - centre) < 1.0,
          f"centroid {centroid:.2f} vs centre {centre}")

    # Contrast: the causal single-pass path is expected to delay. Confirm the test can
    # actually detect a shift, otherwise the assertion above proves nothing.
    zi = dsp.make_stream_state(sos, x[0])
    y_causal, _ = dsp.filter_stream(x, sos, zi)
    e2 = np.square(y_causal.astype(np.float64))
    centroid_causal = float((np.arange(n) * e2).sum() / e2.sum())
    check("causal path does shift (test is sensitive)", centroid_causal - centre > 2.0,
          f"causal centroid {centroid_causal:.2f}, delay {centroid_causal - centre:.1f} samples")


def test_streaming_continuity() -> None:
    """Frame-by-frame filtering must equal whole-signal filtering.

    This is the bug that hides: filter each frame with fresh state and every frame
    boundary gets a discontinuity, which the envelope detector reads as a beat at
    exactly 1024-sample intervals -- a rock-steady, completely fake heart rate.
    """
    print("streaming continuity")
    sos = dsp.heart_sos(FS)
    x = synth_pcg(duration_s=6.0, bpm=72.0)[0].astype(np.float64)

    zi0 = dsp.make_stream_state(sos, x[0])
    reference, _ = sosfilt(sos, x, zi=zi0)

    frame = 1024
    zi = dsp.make_stream_state(sos, x[0])
    chunks = []
    for start in range(0, len(x) - frame + 1, frame):
        y, zi = dsp.filter_stream(x[start : start + frame], sos, zi)
        chunks.append(y)
    streamed = np.concatenate(chunks)

    err = float(np.max(np.abs(streamed - reference[: streamed.size].astype(np.float32))))
    check("chunked == whole-signal", err < 1e-5, f"max abs error {err:.2e}")

    # And show the naive version genuinely fails, so the check above has teeth.
    naive = np.concatenate([
        sosfilt(sos, x[s : s + frame]) for s in range(0, len(x) - frame + 1, frame)
    ])
    naive_err = float(np.max(np.abs(naive - reference[: naive.size])))
    check("stateless framing is detectably wrong", naive_err > 1e-3,
          f"max abs error {naive_err:.2e}")


def test_stream_state_transient() -> None:
    """Non-zero initial state must suppress the startup transient on a DC-offset input."""
    print("startup transient")
    sos = dsp.heart_sos(FS)
    x = np.full(2000, 0.5)  # constant input: a bandpass should output ~nothing

    y_zero, _ = dsp.filter_stream(x, sos, np.zeros_like(dsp.sosfilt_zi(sos)))
    y_scaled, _ = dsp.filter_stream(x, sos, dsp.make_stream_state(sos, x[0]))

    peak_zero = float(np.max(np.abs(y_zero[:200])))
    peak_scaled = float(np.max(np.abs(y_scaled[:200])))
    check("scaled state beats zeroed state", peak_scaled < peak_zero / 10,
          f"transient {peak_scaled:.2e} vs {peak_zero:.2e}")


def test_chirp_response() -> None:
    """A sweep must be attenuated outside the band and pass inside it."""
    print("chirp sweep 1 -> 1000 Hz")
    sos = dsp.heart_sos(FS)
    seconds = 20.0
    t = np.arange(int(seconds * FS)) / FS
    x = chirp(t, f0=1.0, f1=1000.0, t1=seconds, method="linear")
    y = dsp.filter_offline(x, sos)

    def band_rms(f_lo: float, f_hi: float) -> float:
        # Linear sweep: instantaneous frequency maps linearly onto time.
        i0 = int(f_lo / 1000.0 * len(t))
        i1 = int(f_hi / 1000.0 * len(t))
        return rms(y[i0:i1])

    inside = band_rms(60, 160)
    below = band_rms(2, 15)
    above = band_rms(400, 900)
    check("sweep passes inside band", inside > 0.5, f"rms {inside:.3f}")
    check("sweep blocked below band", below < inside / 20, f"rms {below:.4f}")
    check("sweep blocked above band", above < inside / 20, f"rms {above:.4f}")


def test_shannon_energy() -> None:
    print("shannon energy")
    # -x^2 log(x^2) is zero at |x| = 1 and non-negative on [-1, 1].
    check("zero at |x|=1", abs(float(dsp.shannon_energy(np.array([1.0]))[0])) < 1e-12)
    x = np.linspace(-1, 1, 501)
    check("non-negative on [-1,1]", bool(np.all(dsp.shannon_energy(x) >= -1e-12)))
    check("even function", np.allclose(dsp.shannon_energy(x), dsp.shannon_energy(-x)))
    check("silence -> ~0", float(np.max(dsp.shannon_energy(np.zeros(100)))) < 1e-9)

    # Peak of -u log(u) in u = x^2 is at u = 1/e, i.e. |x| = 1/sqrt(e) ~ 0.6065.
    grid = np.linspace(0.01, 1.0, 20001)
    peak_at = float(grid[int(np.argmax(dsp.shannon_energy(grid)))])
    check("peaks at 1/sqrt(e)", abs(peak_at - float(np.exp(-0.5))) < 1e-3,
          f"peak |x| = {peak_at:.4f}, expected {float(np.exp(-0.5)):.4f}")


def test_moving_average() -> None:
    print("moving average")
    check("length preserved", dsp.moving_average(np.zeros(1000), 101).size == 1000)

    const = np.full(500, 0.7)
    out = dsp.moving_average(const, 101)
    check("constant in -> constant out incl. edges",
          float(np.max(np.abs(out - 0.7))) < 1e-12,
          "edge padding, not zero padding")

    # Symmetric input must give symmetric output, or the envelope is time-shifted.
    x = np.zeros(1001)
    x[500] = 1.0
    out = dsp.moving_average(x, 50)  # even window -> forced odd internally
    check("symmetric in -> symmetric out", np.allclose(out, out[::-1], atol=1e-12))

    check("window=1 is identity", np.allclose(dsp.moving_average(x, 1), x))


def test_envelope_on_known_pcg() -> None:
    """The end-to-end deterministic path on a signal with known S1/S2 positions."""
    print("envelope on synthetic PCG with ground truth")
    from scipy.signal import find_peaks

    signal, s1_idx, s2_idx = synth_pcg(duration_s=10.0, bpm=72.0)
    sos = dsp.heart_sos(FS)
    env = dsp.envelope(dsp.filter_offline(signal, sos), FS)

    check("envelope length preserved", env.size == signal.size)
    check("envelope is normalised", abs(float(env.mean())) < 1e-4 and abs(float(env.std()) - 1) < 1e-3,
          f"mean {env.mean():.2e}, std {env.std():.4f}")

    # 200 ms refractory period, as specified, to avoid double-counting one sound.
    peaks, _ = find_peaks(env, height=0.5, distance=int(0.2 * FS))

    truth = np.sort(np.concatenate([s1_idx, s2_idx]))
    check("finds every heart sound", peaks.size == truth.size,
          f"{peaks.size} peaks vs {truth.size} true events")

    errors = np.array([np.min(np.abs(truth - p)) for p in peaks])
    worst_ms = float(errors.max()) / FS * 1000
    check("peaks land on true events", worst_ms < 25.0, f"worst offset {worst_ms:.1f} ms")

    # The property the deterministic segmenter depends on: diastole > systole.
    s1_peaks = np.array([p for p in peaks if np.min(np.abs(s1_idx - p)) < 50])
    check("S1 events all found", s1_peaks.size == s1_idx.size)
    bpm = 60.0 * FS / float(np.mean(np.diff(s1_peaks)))
    check("recovered heart rate", abs(bpm - 72.0) < 1.0, f"{bpm:.2f} bpm, expected 72.00")


def main() -> int:
    for fn in [
        test_design_guards,
        test_passband_and_stopband,
        test_zero_phase,
        test_streaming_continuity,
        test_stream_state_transient,
        test_chirp_response,
        test_shannon_energy,
        test_moving_average,
        test_envelope_on_known_pcg,
    ]:
        fn()
    print(f"\n{PASSED} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
