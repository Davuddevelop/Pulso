"""segment.py against synthetic PCGs with known S1/S2 positions and BPM.

Run:  python tests/test_segment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import segment  # noqa: E402
from check_io import synth_pcg  # noqa: E402
from segment import HeartSegmenter, label_s1_s2, segment_offline  # noqa: E402
from sources import FRAME_SIZE, SAMPLE_RATE  # noqa: E402

PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    PASSED += 1
    print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))


def test_offline_normal_rate() -> None:
    print("offline: 72 bpm, clear asymmetry")
    signal, s1_idx, s2_idx = synth_pcg(duration_s=15.0, bpm=72.0)
    result = segment_offline(signal, SAMPLE_RATE)

    check("bpm recovered", result.bpm is not None and abs(result.bpm - 72.0) < 1.0,
          f"{result.bpm}")
    check("s1/s2 confident at normal rate", result.s1s2_confident)
    check("quality good", result.quality == "good")

    got_s1 = sorted(b.sample_idx for b in result.beats if b.kind == "S1")
    got_s2 = sorted(b.sample_idx for b in result.beats if b.kind == "S2")
    check("S1 count matches ground truth", len(got_s1) == len(s1_idx),
          f"{len(got_s1)} vs {len(s1_idx)}")
    check("S2 count matches ground truth", len(got_s2) == len(s2_idx),
          f"{len(got_s2)} vs {len(s2_idx)}")

    # Every detected S1 must be near a true S1, not a mislabelled S2.
    max_err = max(
        (min(abs(g - t) for t in s1_idx) for g in got_s1), default=0
    )
    check("S1 labels land on true S1 events", max_err < 50, f"worst offset {max_err} samples")


def test_offline_bradycardia_and_tachycardia() -> None:
    """BPM recovery across a range where the timing asymmetry genuinely holds.

    Real systole duration is close to constant (~280-320 ms) across normal heart
    rates; it is diastole that shortens as rate rises. That is also *why* the
    asymmetry-collapse failure mode is real physiology and not just a simulation
    artefact: push the rate high enough (~150+ bpm) and diastole shrinks down toward
    systole on its own. That regime is covered separately, below, by
    test_asymmetry_collapse_degrades_honestly -- this test stays in the range where
    confident S1/S2 labelling is a fair thing to ask for.
    """
    print("offline: bpm across a physiological range")
    for bpm in (45.0, 60.0, 75.0, 90.0):
        signal, s1_idx, _ = synth_pcg(duration_s=15.0, bpm=bpm, systole_s=0.30)
        result = segment_offline(signal, SAMPLE_RATE)
        check(f"{bpm:.0f} bpm recovered", result.bpm is not None and abs(result.bpm - bpm) < 2.0,
              f"got {result.bpm}")
        check(f"{bpm:.0f} bpm confident", result.s1s2_confident)


def test_asymmetry_collapse_degrades_honestly() -> None:
    """At a high enough rate systole/diastole become indistinguishable.

    This is the specific failure CLAUDE.md calls out: the segmenter must notice and
    say S1/S2 is uncertain rather than confidently mislabel every other beat.
    """
    print("offline: asymmetry collapse -> uncertain, not wrong")
    # Equal-length systole/diastole: the pathological case where the timing cue this
    # whole method relies on is simply not present in the signal.
    #
    # bpm=170 rather than a rounder number: synth_pcg's S1 and S2 bursts have
    # different durations (100 ms / 60 ms), which shifts each envelope peak slightly
    # off its nominal onset. That shift is a near-fixed number of samples, so its
    # effect on the measured interval ratio does not scale down smoothly as the
    # nominal gap shrinks -- confirmed empirically, 150 bpm still measures a
    # deceptively separable ratio (~1.22) because of it, while 170 bpm measures ~1.01.
    # The unit test below (test_label_s1_s2_unit) exercises the boundary itself on
    # exact, artifact-free intervals; this one is the integration check that a real
    # collapse, through the whole filter/envelope/peak pipeline, is caught.
    bpm = 170.0
    period = 60.0 / bpm
    signal, s1_idx, s2_idx = synth_pcg(duration_s=15.0, bpm=bpm, systole_s=period / 2)
    result = segment_offline(signal, SAMPLE_RATE)

    check("still detects beats", len(result.beats) > 0, f"{len(result.beats)} beats")
    check("declines to label S1/S2 with confidence", not result.s1s2_confident)
    check("beats carry 'uncertain' kind", all(b.kind == "uncertain" for b in result.beats))


def test_label_s1_s2_unit() -> None:
    print("label_s1_s2 unit: hand-built interval pattern")
    # Peaks at 0, 320, 833, 1153, 1666 ms (systole=320ms, diastole=513ms, two beats).
    fs = SAMPLE_RATE
    times_ms = [0, 320, 833, 1153, 1666, 1986]
    idx = np.array([int(t / 1000 * fs) for t in times_ms])
    amps = np.ones(idx.size)

    beats, confident = label_s1_s2(idx, amps)
    check("confident on clean alternating pattern", confident)
    kinds = [b.kind for b in beats]
    check("labels alternate starting S1", kinds == ["S1", "S2"] * 3, f"{kinds}")

    # Too few peaks: must not crash, must come back uncertain.
    beats2, confident2 = label_s1_s2(idx[:2], amps[:2])
    check("too few peaks -> uncertain, no crash", not confident2 and len(beats2) == 2)

    # Exact intervals, no burst-shape confound: the threshold itself, on the numbers
    # it actually compares.
    even_idx = np.arange(0, 6 * 400, 400)  # every gap identical -> no asymmetry at all
    _, confident3 = label_s1_s2(even_idx, np.ones(even_idx.size))
    check("perfectly uniform spacing -> uncertain", not confident3)

    def alternating(short: float, long: float, n_pairs: int = 3) -> np.ndarray:
        deltas = [short, long] * n_pairs
        return np.concatenate([[0], np.cumsum(deltas)]).astype(int)

    short = 400.0
    just_under = alternating(short, short * segment.MIN_ASYMMETRY_RATIO - 1)
    _, confident4 = label_s1_s2(just_under, np.ones(just_under.size))
    check("ratio just under threshold -> uncertain", not confident4)

    just_over = alternating(short, short * segment.MIN_ASYMMETRY_RATIO + 1)
    _, confident5 = label_s1_s2(just_over, np.ones(just_over.size))
    check("ratio just over threshold -> confident", confident5)


def test_noise_reports_noisy() -> None:
    print("offline: pure noise input")
    rng = np.random.default_rng(1)
    noise = (rng.normal(0, 0.02, 15 * SAMPLE_RATE)).astype(np.float32)
    result = segment_offline(noise, SAMPLE_RATE)
    check("no fabricated beats on noise", result.quality == "noisy" or result.bpm is None,
          f"quality={result.quality!r} bpm={result.bpm}")


def test_silence_reports_noisy() -> None:
    print("offline: silence")
    silence = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)
    result = segment_offline(silence, SAMPLE_RATE)
    check("silence -> noisy, no beats", result.quality == "noisy" and len(result.beats) == 0,
          f"quality={result.quality!r} beats={len(result.beats)}")


def test_streaming_matches_offline_bpm() -> None:
    """The live path (persistent zi, buffered re-analysis) must agree with offline."""
    print("streaming vs offline agreement")
    signal, _, _ = synth_pcg(duration_s=12.0, bpm=84.0)
    offline = segment_offline(signal, SAMPLE_RATE)

    seg = HeartSegmenter(sample_rate=SAMPLE_RATE, buffer_s=8.0)
    result = None
    for start in range(0, signal.size - FRAME_SIZE + 1, FRAME_SIZE):
        result = seg.push(signal[start : start + FRAME_SIZE])

    check("streaming produces a reading", result is not None and result.bpm is not None,
          f"{result}")
    check("streaming bpm close to offline bpm", abs(result.bpm - offline.bpm) < 1.5,
          f"streaming {result.bpm:.2f} vs offline {offline.bpm:.2f}")


def test_streaming_beat_indices_are_absolute() -> None:
    """A beat reported after frame N must be indexed against the whole stream, not the buffer."""
    print("streaming beat indices stay absolute across buffer trims")
    signal, s1_idx, _ = synth_pcg(duration_s=20.0, bpm=72.0)  # long enough to overflow an 8s buffer
    seg = HeartSegmenter(sample_rate=SAMPLE_RATE, buffer_s=8.0)

    result = None
    for start in range(0, signal.size - FRAME_SIZE + 1, FRAME_SIZE):
        result = seg.push(signal[start : start + FRAME_SIZE])

    check("buffer actually trimmed (stream longer than buffer)", signal.size > seg.buffer_len)
    last_beat = result.beats[-1].sample_idx
    check("last beat index is near the end of the stream, not reset to the buffer-local range",
          last_beat > seg.buffer_len, f"last beat idx {last_beat}, buffer_len {seg.buffer_len}")


def main() -> int:
    for fn in [
        test_offline_normal_rate,
        test_offline_bradycardia_and_tachycardia,
        test_asymmetry_collapse_degrades_honestly,
        test_label_s1_s2_unit,
        test_noise_reports_noisy,
        test_silence_reports_noisy,
        test_streaming_matches_offline_bpm,
        test_streaming_beat_indices_are_absolute,
    ]:
        fn()
    print(f"\n{PASSED} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
