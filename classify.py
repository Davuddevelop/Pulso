"""Triage classification: segmenter output in -> {label, confidence} out.

The CNN specified in CLAUDE.md (log-mel spectrograms, normal/abnormal, trained on
CinC 2016) was never attempted. It needs the dataset, and PhysioNet is unreachable
from this session's network policy -- confirmed as an org egress denial, not a
transient failure. Per CLAUDE.md's own instruction for exactly this situation ("if it
does not converge in one attempt, ship the deterministic layer alone and say so
honestly"), that is what ships: heart-sound-morphology classification (murmurs, rubs,
extra sounds) is unimplemented, not merely disabled or stubbed to a placeholder value.

What this module actually does is smaller and should not be mistaken for that: a
deterministic triage heuristic computed from segment.py's rate and rhythm output --
heart rate outside a typical resting range, or beat-to-beat spacing too irregular to
look like sinus rhythm. A community health worker already uses resting rate and pulse
regularity this way with two fingers and a watch; this is the same cue, timed more
precisely. It is not listening to the *sound* of the heartbeat at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from segment import SegmentResult

# Deliberately wide: a triage cue for "worth a listen", not a clinical cutoff, and
# applied across whoever walks up to a community health worker -- children through
# the elderly, at rest or having just arrived.
TYPICAL_BPM_RANGE = (50.0, 110.0)

# Coefficient of variation (std / mean) of consecutive S1-S1 intervals. Above this the
# spacing looks irregular enough to flag for a listen. This names nothing about *why*
# it is irregular -- that is exactly the diagnosis this project must never produce.
IRREGULARITY_CV_THRESHOLD = 0.12

# Below this many confidently-labelled S1 beats, a coefficient of variation is noise,
# not a rhythm reading.
MIN_BEATS_FOR_RHYTHM_CHECK = 5


@dataclass(frozen=True)
class ClassifyResult:
    label: str  # "normal" | "review recommended" | "signal too noisy" -- exactly these three
    confidence: float  # 0..1, a heuristic weight -- NOT a probability, NOT a measured
    # accuracy figure. No held-out evaluation has ever been run against this heuristic.
    # Nothing in app.py may present this number as "accuracy" or "% confidence" to a
    # user -- CLAUDE.md is explicit that an unmeasured accuracy claim is the one thing
    # that gets a judge to discount everything else on the slide.
    reason: str  # safe to show verbatim on a triage screen; names findings, not causes


def classify(result: SegmentResult) -> ClassifyResult:
    """The whole heuristic. Quality gates first -- a garbage input never gets to "normal"."""
    if result.quality == "noisy" or result.bpm is None:
        return ClassifyResult("signal too noisy", 0.0, result.quality_reason)

    reasons: list[str] = []

    if not (TYPICAL_BPM_RANGE[0] <= result.bpm <= TYPICAL_BPM_RANGE[1]):
        reasons.append(f"heart rate {result.bpm:.0f} bpm is outside the typical resting range")

    s1_idx = np.array(sorted(b.sample_idx for b in result.beats if b.kind == "S1"))
    if s1_idx.size >= MIN_BEATS_FOR_RHYTHM_CHECK:
        intervals = np.diff(s1_idx).astype(np.float64)
        cv = float(intervals.std() / intervals.mean()) if intervals.mean() > 0 else 0.0
        if cv > IRREGULARITY_CV_THRESHOLD:
            reasons.append("beat-to-beat rhythm is irregular")
    # Fewer confidently-labelled S1 beats than that, or S1/S2 labelling itself was not
    # confident (segment.py's asymmetry-collapse case): the rhythm check is silently
    # skipped rather than guessed at. The rate check above still applies -- bpm is
    # still available even when S1/S2 labels are not, via segment.py's raw-peak-pair
    # fallback.

    if reasons:
        return ClassifyResult("review recommended", 0.6, "; ".join(reasons))
    return ClassifyResult("normal", 0.5, "heart rate and rhythm within typical range")
