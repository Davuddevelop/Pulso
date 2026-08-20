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

On top of the label, a "review recommended" result also carries an escalation
timeframe (urgency) and a plain next step (action) -- e.g. "see a clinician within a
day or two" versus "mention it at the next routine visit". This is routing, not
diagnosis: it says how soon to escalate and to whom in the most generic terms
possible ("a clinician", "the nearest health facility"), scaled only by how far the
measured rate or rhythm deviates from typical. It never names a specialist type --
the pipeline only measures rate and rhythm broadly, with no basis to say which kind
of specialist would even be relevant -- and it never names a condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from segment import SegmentResult

# Deliberately wide: a triage cue for "worth a listen", not a clinical cutoff, and
# applied across whoever walks up to a community health worker -- children through
# the elderly, at rest or having just arrived.
TYPICAL_BPM_RANGE = (50.0, 110.0)

# How far outside TYPICAL_BPM_RANGE, in bpm, before the rate deviation is considered
# "moderate" or "marked" rather than "mild". Widening bands, not clinical cutoffs --
# the point is only that a rate 60 bpm outside typical is a different urgency than one
# 5 bpm outside it, not that either number means something specific.
RATE_DEVIATION_BANDS = (15.0, 35.0)  # mild < 15 <= moderate < 35 <= marked

# Coefficient of variation (std / mean) of consecutive S1-S1 intervals. Above this the
# spacing looks irregular enough to flag for a listen. This names nothing about *why*
# it is irregular -- that is exactly the diagnosis this project must never produce.
IRREGULARITY_CV_THRESHOLD = 0.12

# Same idea as RATE_DEVIATION_BANDS, for rhythm: how far past the threshold before
# irregularity is "moderate" or "marked" rather than "mild".
IRREGULARITY_BANDS = (0.20, 0.32)

# Below this many confidently-labelled S1 beats, a coefficient of variation is noise,
# not a rhythm reading.
MIN_BEATS_FOR_RHYTHM_CHECK = 5

# Escalation copy, ordered mild -> marked. Deliberately generic about who to see --
# "a clinician" / "the nearest health facility", never a named specialty -- and
# deliberately generic about what was found -- a timeframe, never a condition.
_ACTION_BY_URGENCY = {
    "routine": "Not urgent. Mention this reading at the next routine checkup.",
    "prompt": "See a clinician within the next day or two if possible.",
    "urgent": "Recommend seeing a clinician or the nearest health facility as soon as possible.",
}


def _band_index(value: float, bands: tuple[float, float]) -> int:
    """0 = mild, 1 = moderate, 2 = marked."""
    if value < bands[0]:
        return 0
    if value < bands[1]:
        return 1
    return 2


@dataclass(frozen=True)
class ClassifyResult:
    label: str  # "normal" | "review recommended" | "signal too noisy" -- exactly these three
    confidence: float  # 0..1, a heuristic weight -- NOT a probability, NOT a measured
    # accuracy figure. No held-out evaluation has ever been run against this heuristic.
    # Nothing in app.py may present this number as "accuracy" or "% confidence" to a
    # user -- CLAUDE.md is explicit that an unmeasured accuracy claim is the one thing
    # that gets a judge to discount everything else on the slide.
    reason: str  # safe to show verbatim on a triage screen; names findings, not causes
    urgency: str | None = None  # "routine" | "prompt" | "urgent", only when label is
    # "review recommended"; None otherwise. A timeframe derived from how far the
    # measurement deviates -- not a severity assessment of any underlying condition,
    # because no such assessment is being made.
    action: str | None = None  # plain next step matching `urgency`; None otherwise


def classify(result: SegmentResult) -> ClassifyResult:
    """The whole heuristic. Quality gates first -- a garbage input never gets to "normal"."""
    if result.quality == "noisy" or result.bpm is None:
        return ClassifyResult("signal too noisy", 0.0, result.quality_reason)

    reasons: list[str] = []
    worst_band = -1  # -1 = nothing flagged

    if result.bpm < TYPICAL_BPM_RANGE[0] or result.bpm > TYPICAL_BPM_RANGE[1]:
        deviation = max(TYPICAL_BPM_RANGE[0] - result.bpm, result.bpm - TYPICAL_BPM_RANGE[1])
        band = _band_index(deviation, RATE_DEVIATION_BANDS)
        worst_band = max(worst_band, band)
        reasons.append(f"heart rate {result.bpm:.0f} bpm is outside the typical resting range")

    s1_idx = np.array(sorted(b.sample_idx for b in result.beats if b.kind == "S1"))
    if s1_idx.size >= MIN_BEATS_FOR_RHYTHM_CHECK:
        intervals = np.diff(s1_idx).astype(np.float64)
        cv = float(intervals.std() / intervals.mean()) if intervals.mean() > 0 else 0.0
        if cv > IRREGULARITY_CV_THRESHOLD:
            band = _band_index(cv, IRREGULARITY_BANDS)
            worst_band = max(worst_band, band)
            reasons.append("beat-to-beat rhythm is irregular")
    # Fewer confidently-labelled S1 beats than that, or S1/S2 labelling itself was not
    # confident (segment.py's asymmetry-collapse case): the rhythm check is silently
    # skipped rather than guessed at. The rate check above still applies -- bpm is
    # still available even when S1/S2 labels are not, via segment.py's raw-peak-pair
    # fallback.

    if not reasons:
        return ClassifyResult("normal", 0.5, "heart rate and rhythm within typical range")

    # Two simultaneous mild findings are treated as one step more urgent than either
    # alone -- concurrent flags are a reasonable reason to escalate sooner even when
    # neither one individually looks severe. This is the only place severity and
    # breadth interact; it still never asks what the two findings might mean together.
    if len(reasons) > 1 and worst_band == 0:
        worst_band = 1

    urgency = ("routine", "prompt", "urgent")[worst_band]
    return ClassifyResult(
        "review recommended", 0.6, "; ".join(reasons), urgency, _ACTION_BY_URGENCY[urgency]
    )
