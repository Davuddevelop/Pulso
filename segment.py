"""Heart sound segmentation: beat detection, S1/S2 labelling, heart rate.

Stateful. Owns filter continuity (a persistent SOS delay line) and an envelope buffer
across calls, so the caller can feed it a growing recording or a live stream and get
back a live-updating picture.

CinC 2016 ships manually-corrected S1/S2 labels, which is normally reason to prefer a
supervised segmenter over hand-tuned peak detection. That path is blocked in this
environment -- PhysioNet is unreachable through the session's egress policy, so there
is no labelled data to train or validate against. What is here is the deterministic
fallback CLAUDE.md specifies anyway, using the cardiac timing asymmetry: systole
(S1->S2) is shorter than diastole (S2->S1) at normal heart rates, so the peak
following the longer gap is S1. At high heart rates that asymmetry collapses, and this
module is required to notice and degrade rather than report a confident wrong label.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

import dsp
from sources import SAMPLE_RATE

# After a detected peak, no new peak is accepted for this long. Prevents one heart
# sound's own filter ringing from being counted as two beats.
#
# CLAUDE.md's segmentation section names ~200 ms for this. Measured directly against
# a single isolated burst through this exact filter+envelope chain, the ringing that
# actually needs suppressing produces a second local max only ~3.5 ms after the first
# -- 200 ms is not protecting against that, it is a general minimum-spacing floor, and
# at that size it swallows genuine S1-S2 pairs whenever a heart rate pushes systole
# below it (confirmed: a real pair 190 ms apart at 100 bpm was collapsing into a single
# detection, which corrupted the interval statistics and produced a wildly wrong BPM,
# not just an imprecise one). 80 ms keeps more than 20x margin over the measured
# ringing while staying under systole up to a fast heart rate.
REFRACTORY_S = 0.08

# A peak must clear this many normalized-envelope units to count as a heart sound at
# all, distinct from the noise floor. The envelope from dsp.envelope() is already
# zero-mean, unit-std, so this is directly a z-score threshold.
PEAK_HEIGHT = 1.0

# mean(long interval) / mean(short interval) must clear this before S1/S2 labels are
# trusted. Below it, systole and diastole are indistinguishable and a confident label
# would be a coin flip dressed up as an answer.
MIN_ASYMMETRY_RATIO = 1.15

# Fewer than this many peaks in the analysis window and there simply is not enough
# history to say anything about rhythm or rate.
MIN_PEAKS_FOR_ANALYSIS = 3

# BPM outside this range is treated as a detector failure (double- or half-counting)
# rather than a real reading. Generous on both sides: a scared toddler and a resting
# athlete are both plausible community-health-worker encounters.
PLAUSIBLE_BPM = (35.0, 220.0)


@dataclass(frozen=True)
class Beat:
    sample_idx: int
    kind: str  # "S1" | "S2" | "uncertain"
    amplitude: float  # envelope value at the peak -- a z-score, not a physical unit


@dataclass(frozen=True)
class SegmentResult:
    beats: list[Beat]
    bpm: float | None  # None if not confidently estimable
    s1s2_confident: bool  # False once diastole/systole asymmetry has collapsed
    quality: str  # "good" | "noisy"
    quality_reason: str  # human-readable, safe to show on a triage screen


def _classify_intervals(peak_idx: np.ndarray) -> tuple[np.ndarray, float]:
    """Label each gap between consecutive peaks short/long, return (is_long, ratio).

    Threshold sits at the widest gap between consecutive *sorted* interval values --
    the natural boundary between the short (systole) and long (diastole) clusters.
    A plain median was tried first and rejected: with an uneven split between the two
    clusters (e.g. 18 short intervals vs 17 long, which is simply what an odd peak
    count gives you), the median can equal a member of the short cluster itself, and
    an inclusive ">=" then reclassifies exactly that one boundary interval as long --
    a real bug caught by test_offline_normal_rate, not a hypothetical one.
    """
    intervals = np.diff(peak_idx).astype(np.float64)
    if intervals.size < 2:
        return np.zeros_like(intervals, dtype=bool), 1.0

    order = np.argsort(intervals)
    sorted_vals = intervals[order]
    gaps = np.diff(sorted_vals)
    split = int(np.argmax(gaps))
    threshold = (sorted_vals[split] + sorted_vals[split + 1]) / 2.0
    is_long = intervals > threshold

    longs = intervals[is_long]
    shorts = intervals[~is_long]
    if longs.size == 0 or shorts.size == 0:
        return is_long, 1.0  # perfectly uniform spacing: no asymmetry to exploit
    ratio = float(longs.mean() / shorts.mean())
    return is_long, ratio


def label_s1_s2(peak_idx: np.ndarray, amplitudes: np.ndarray) -> tuple[list[Beat], bool]:
    """Apply the systole-shorter-than-diastole rule. Returns (beats, confident).

    The peak following a long (diastolic) gap is S1; the peak following a short
    (systolic) gap is S2. The very first peak is inferred from the gap after it, run
    in reverse.

    When intervals do not clearly separate into two populations -- the collapse case
    at high heart rate -- every beat comes back "uncertain" rather than a guess. The
    beats themselves (and therefore the raw peak count and interval-derived BPM) are
    still returned: "beats detected, S1/S2 uncertain" per CLAUDE.md, not silence.
    """
    n = peak_idx.size
    if n < MIN_PEAKS_FOR_ANALYSIS:
        return [
            Beat(int(i), "uncertain", float(a)) for i, a in zip(peak_idx, amplitudes)
        ], False

    is_long, ratio = _classify_intervals(peak_idx)
    confident = ratio >= MIN_ASYMMETRY_RATIO

    if not confident:
        return [
            Beat(int(i), "uncertain", float(a)) for i, a in zip(peak_idx, amplitudes)
        ], False

    kinds = [""] * n
    # interval[i] is the gap peak[i] -> peak[i+1]. Long -> peak[i+1] is S1, peak[i] is S2.
    for i, long in enumerate(is_long):
        kinds[i + 1] = "S1" if long else "S2"
    # First peak: opposite of what interval[0] assigned to peak[1] (S1<->S2 alternate).
    kinds[0] = "S2" if kinds[1] == "S1" else "S1"

    beats = [Beat(int(idx), kind, float(amp)) for idx, kind, amp in zip(peak_idx, kinds, amplitudes)]
    return beats, True


def _estimate_bpm(beats: list[Beat], sample_rate: int) -> float | None:
    """BPM from consecutive same-label peaks when confident, else from raw peak pairs."""
    if len(beats) < 2:
        return None

    s1 = [b.sample_idx for b in beats if b.kind == "S1"]
    if len(s1) >= 2:
        bpm = 60.0 * sample_rate / float(np.mean(np.diff(s1)))
    else:
        # Uncertain case: assume peaks alternate S1/S2 1:1, so consecutive peaks are
        # half a beat apart on average.
        idx = np.array([b.sample_idx for b in beats])
        mean_gap = float(np.mean(np.diff(idx)))
        if mean_gap <= 0:
            return None
        bpm = 60.0 * sample_rate / (2.0 * mean_gap)

    if not (PLAUSIBLE_BPM[0] <= bpm <= PLAUSIBLE_BPM[1]):
        return None
    return bpm


class HeartSegmenter:
    """Streaming segmenter: feed frames, get back the current picture.

    Recomputes peak-picking over a trailing buffer on every call rather than doing
    true incremental peak detection. This is a deliberate scope cut for the hour
    budget: true streaming peak-picking needs to hold a peak candidate until enough
    future samples confirm it is a local max and not a rising edge, which is a second
    state machine on top of the filter's. Recomputing over a few seconds of buffer at
    ~2 Hz (once per UI redraw) costs nothing measurable at fs=2000 and is what actually
    ships. The filter state (`zi`) is still genuinely persistent across frames, which
    is the part that is not optional -- see dsp.py streaming-continuity test for why.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, buffer_s: float = 8.0) -> None:
        self.sample_rate = sample_rate
        self.buffer_len = int(buffer_s * sample_rate)

        self._sos = dsp.heart_sos(sample_rate)
        self._zi: np.ndarray | None = None
        self._filtered = np.zeros(0, dtype=np.float32)
        self._samples_seen = 0

    def reset(self) -> None:
        self._zi = None
        self._filtered = np.zeros(0, dtype=np.float32)
        self._samples_seen = 0

    def push(self, frame: np.ndarray) -> SegmentResult:
        """Feed one frame (any length) of raw, unfiltered audio. Returns the current read."""
        if self._zi is None:
            self._zi = dsp.make_stream_state(self._sos, float(frame[0]) if frame.size else 0.0)

        filtered, self._zi = dsp.filter_stream(frame, self._sos, self._zi)
        self._samples_seen += frame.size

        self._filtered = np.concatenate([self._filtered, filtered])[-self.buffer_len :]

        return self._analyze(self._filtered, offset=self._samples_seen - self._filtered.size)

    def _analyze(self, filtered_buffer: np.ndarray, offset: int) -> SegmentResult:
        if filtered_buffer.size < int(0.5 * self.sample_rate):
            return SegmentResult([], None, False, "noisy", "warming up")

        env = dsp.envelope(filtered_buffer, self.sample_rate)
        distance = max(1, int(REFRACTORY_S * self.sample_rate))
        peaks, props = find_peaks(env, height=PEAK_HEIGHT, distance=distance)

        if peaks.size == 0:
            return SegmentResult([], None, False, "noisy", "no heart sounds detected above the noise floor")

        beats, confident = label_s1_s2(peaks, env[peaks])
        beats = [Beat(b.sample_idx + offset, b.kind, b.amplitude) for b in beats]

        bpm = _estimate_bpm(beats, self.sample_rate)

        if bpm is None:
            return SegmentResult(beats, None, confident, "noisy",
                                  "beats detected but rate is not physiologically plausible")

        quality = "good"
        reason = "ok"
        return SegmentResult(beats, bpm, confident, quality, reason)


def segment_offline(signal: np.ndarray, sample_rate: int) -> SegmentResult:
    """Whole-recording analysis, for validation and for the tests.

    Uses filter_offline (zero-phase) rather than the streaming path, since there is no
    real-time constraint here and zero-phase gives exact peak positions to check
    against ground truth.
    """
    sos = dsp.heart_sos(sample_rate)
    filtered = dsp.filter_offline(signal, sos)
    env = dsp.envelope(filtered, sample_rate)

    distance = max(1, int(REFRACTORY_S * sample_rate))
    peaks, _ = find_peaks(env, height=PEAK_HEIGHT, distance=distance)

    if peaks.size == 0:
        return SegmentResult([], None, False, "noisy", "no heart sounds detected above the noise floor")

    beats, confident = label_s1_s2(peaks, env[peaks])
    bpm = _estimate_bpm(beats, sample_rate)

    if bpm is None:
        return SegmentResult(beats, None, confident, "noisy",
                              "beats detected but rate is not physiologically plausible")

    return SegmentResult(beats, bpm, confident, "good", "ok")
