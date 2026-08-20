"""classify.py against hand-built SegmentResults and real synthetic-audio integration.

Run:  python tests/test_classify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from check_io import synth_pcg  # noqa: E402
from classify import ClassifyResult, classify  # noqa: E402
from segment import Beat, SegmentResult, segment_offline  # noqa: E402
from sources import SAMPLE_RATE  # noqa: E402

PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    PASSED += 1
    print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))


def s1_beats(s1_s1_gaps: list[int]) -> list[Beat]:
    """Build S1 beats spaced by the given consecutive S1-S1 gaps, one S2 between each.

    classify.py's rhythm check only reads kind=="S1" timestamps, so the S2 placement
    below is cosmetic realism, not load-bearing -- what matters is that each element
    of ``s1_s1_gaps`` lands directly as one S1-to-S1 interval.
    """
    beats = []
    s1_time = 0
    for gap in s1_s1_gaps:
        beats.append(Beat(s1_time, "S1", 1.0))
        beats.append(Beat(s1_time + gap // 2, "S2", 1.0))
        s1_time += gap
    beats.append(Beat(s1_time, "S1", 1.0))
    return beats


def test_quality_gate_first() -> None:
    print("quality gate takes priority over everything else")
    noisy = SegmentResult([], None, False, "noisy", "no heart sounds detected above the noise floor")
    r = classify(noisy)
    check("noisy -> signal too noisy", r.label == "signal too noisy")
    check("reason passed through from segment.py", "noise floor" in r.reason)
    check("never fabricates confidence on noise", r.confidence == 0.0)

    # bpm is None even if quality happens to say "good" (should not happen in
    # practice, but classify must not trust bpm=None regardless of the quality field).
    weird = SegmentResult([], None, False, "good", "ok")
    check("bpm=None -> too noisy even if quality says good", classify(weird).label == "signal too noisy")


def test_normal_rate_regular_rhythm() -> None:
    print("normal: in-range rate, regular rhythm")
    beats = s1_beats([1666] * 10)  # 72 bpm at 2000 Hz, perfectly regular
    result = SegmentResult(beats, 72.0, True, "good", "ok")
    r = classify(result)
    check("in-range regular rhythm -> normal", r.label == "normal")
    check("no findings named", r.reason == "heart rate and rhythm within typical range")


def test_rate_out_of_range() -> None:
    print("review recommended: rate outside typical range")
    slow_beats = s1_beats([2400] * 8)  # 50 bpm boundary... use clearly-outside instead
    for bpm, beats in [
        (35.0, s1_beats([int(60 / 35 * SAMPLE_RATE)] * 8)),
        (150.0, s1_beats([int(60 / 150 * SAMPLE_RATE)] * 8)),
    ]:
        result = SegmentResult(beats, bpm, True, "good", "ok")
        r = classify(result)
        check(f"{bpm:.0f} bpm -> review recommended", r.label == "review recommended", r.reason)
        check(f"{bpm:.0f} bpm reason names the rate", "outside the typical resting range" in r.reason)
        check(f"{bpm:.0f} bpm reason names no disease", not any(
            w in r.reason.lower() for w in ["diagnos", "arrhythmia", "murmur", "disease"]
        ))


def test_irregular_rhythm() -> None:
    print("review recommended: irregular rhythm at a normal average rate")
    # Alternate short and long S1-S1 gaps so the mean rate looks normal (~72 bpm) but
    # beat-to-beat spacing clearly is not.
    fast, slow = 1200, 2200  # samples: averages to 1700 ~ 70.6 bpm
    intervals = [fast, slow] * 6
    beats = s1_beats(intervals)
    mean_bpm = 60.0 * SAMPLE_RATE / np.mean(intervals)
    result = SegmentResult(beats, float(mean_bpm), True, "good", "ok")

    r = classify(result)
    check("mean rate alone is in the typical range", 50.0 <= mean_bpm <= 110.0, f"{mean_bpm:.1f}")
    check("irregular spacing still flagged", r.label == "review recommended", r.reason)
    check("reason names rhythm, not rate", "irregular" in r.reason)


def test_too_few_beats_skips_rhythm_check() -> None:
    print("too few S1 beats: rhythm check silently skipped, not guessed at")
    beats = s1_beats([1666, 1666])  # only 3 S1/S2 pairs worth -> 2 S1 beats
    result = SegmentResult(beats, 72.0, True, "good", "ok")
    r = classify(result)
    check("insufficient beats for rhythm -> falls back to rate-only, normal", r.label == "normal", r.reason)


def test_urgency_defaults_to_none() -> None:
    print("urgency/action are None outside 'review recommended'")
    noisy = classify(SegmentResult([], None, False, "noisy", "no signal"))
    check("noisy -> urgency None", noisy.urgency is None)
    check("noisy -> action None", noisy.action is None)

    normal = classify(SegmentResult(s1_beats([1666] * 10), 72.0, True, "good", "ok"))
    check("normal -> urgency None", normal.urgency is None)
    check("normal -> action None", normal.action is None)


def test_urgency_bands() -> None:
    print("urgency scales with how far the reading deviates, not with a guess at cause")
    # deviation from TYPICAL_BPM_RANGE=(50,110): 35->15 (moderate/prompt boundary),
    # 45->5 (mild/routine), 120->10 (mild/routine), 130->20 (moderate/prompt),
    # 150->40 (marked/urgent). Computed and verified directly against the
    # implementation before writing these as expectations, not guessed.
    cases = [
        (35.0, "prompt"), (45.0, "routine"), (120.0, "routine"),
        (130.0, "prompt"), (150.0, "urgent"),
    ]
    for bpm, expected in cases:
        beats = s1_beats([int(60 / bpm * SAMPLE_RATE)] * 8)
        r = classify(SegmentResult(beats, bpm, True, "good", "ok"))
        check(f"{bpm:.0f} bpm -> urgency {expected!r}", r.urgency == expected, f"got {r.urgency!r}")
        check(f"{bpm:.0f} bpm -> action text present", bool(r.action))

    check("routine action mentions no urgency", "routine" in classify(
        SegmentResult(s1_beats([int(60 / 45 * SAMPLE_RATE)] * 8), 45.0, True, "good", "ok")
    ).action.lower())
    check("urgent action says as soon as possible", "as soon as possible" in classify(
        SegmentResult(s1_beats([int(60 / 150 * SAMPLE_RATE)] * 8), 150.0, True, "good", "ok")
    ).action.lower())


def test_combined_findings_escalate() -> None:
    print("two simultaneous mild findings escalate past routine")
    # 45 bpm alone is a mild rate deviation -> routine. Confirm that, then add a mild
    # rhythm irregularity at a similarly mild-deviation rate and check it moves past
    # routine even though neither finding alone would.
    solo = classify(SegmentResult(
        s1_beats([int(60 / 45 * SAMPLE_RATE)] * 8), 45.0, True, "good", "ok"
    ))
    check("rate deviation alone is routine", solo.urgency == "routine")

    # Both a mild rate deviation and a mild rhythm irregularity at once: mean rate
    # ~117 bpm (7 bpm past the 110 boundary, mild) with cv~0.141 (mild irregularity,
    # 0.12-0.20 band). Values computed and verified directly before writing this test.
    fast, slow = 880, 1170
    intervals = [fast, slow] * 6
    beats = s1_beats(intervals)
    mean_bpm = 60.0 * SAMPLE_RATE / np.mean(intervals)
    combo = classify(SegmentResult(beats, float(mean_bpm), True, "good", "ok"))
    check("both findings triggered", combo.label == "review recommended" and ";" in combo.reason,
          f"label={combo.label} reason={combo.reason}")
    check("two mild findings together -> past routine", combo.urgency != "routine",
          f"urgency={combo.urgency}, reason={combo.reason}")


def test_never_names_a_diagnosis() -> None:
    print("vocabulary check across every reachable code path")
    forbidden = ["diagnos", "disease", "arrhythmia", "murmur", "fibrillation", "treatment",
                 "prescri", "condition", "cardiolog", "pulmonolog", "specialist"]
    cases = [
        SegmentResult([], None, False, "noisy", "no heart sounds detected above the noise floor"),
        SegmentResult(s1_beats([1666] * 10), 72.0, True, "good", "ok"),
        SegmentResult(s1_beats([int(60 / 150 * SAMPLE_RATE)] * 8), 150.0, True, "good", "ok"),
        SegmentResult(s1_beats([1200, 2200] * 6), 70.6, True, "good", "ok"),
        SegmentResult(s1_beats([int(60 / 35 * SAMPLE_RATE)] * 8), 35.0, True, "good", "ok"),
    ]
    for result in cases:
        r = classify(result)
        check(f"label {r.label!r} is one of the three allowed",
              r.label in ("normal", "review recommended", "signal too noisy"))
        text = r.reason.lower() + " " + (r.action or "").lower()
        hit = [w for w in forbidden if w in text]
        check(f"no forbidden vocabulary in reason/action for {result.bpm}", not hit, f"found {hit}")
        if r.label == "review recommended":
            check("urgency is one of the three allowed tiers", r.urgency in ("routine", "prompt", "urgent"))
        else:
            check(f"urgency is None for label {r.label!r}", r.urgency is None)


def test_integration_via_real_audio() -> None:
    print("integration: real synthetic audio through segment_offline -> classify")
    normal_signal, _, _ = synth_pcg(duration_s=15.0, bpm=72.0)
    r = classify(segment_offline(normal_signal, SAMPLE_RATE))
    check("72 bpm clean signal -> normal", r.label == "normal", r.reason)

    fast_signal, _, _ = synth_pcg(duration_s=15.0, bpm=150.0, systole_s=0.2)
    r2 = classify(segment_offline(fast_signal, SAMPLE_RATE))
    check("150 bpm -> review recommended", r2.label == "review recommended", r2.reason)
    check("150 bpm -> urgent", r2.urgency == "urgent", f"got {r2.urgency}")
    check("150 bpm -> action present", bool(r2.action))

    silence = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)
    r3 = classify(segment_offline(silence, SAMPLE_RATE))
    check("silence -> signal too noisy", r3.label == "signal too noisy", r3.reason)


def main() -> int:
    for fn in [
        test_quality_gate_first,
        test_normal_rate_regular_rhythm,
        test_rate_out_of_range,
        test_irregular_rhythm,
        test_too_few_beats_skips_rhythm_check,
        test_urgency_defaults_to_none,
        test_urgency_bands,
        test_combined_findings_escalate,
        test_never_names_a_diagnosis,
        test_integration_via_real_audio,
    ]:
        fn()
    print(f"\n{PASSED} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
