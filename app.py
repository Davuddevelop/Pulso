"""Live triage display. UI / orchestration only -- no DSP logic here.

Pulls frames from an AudioSource, feeds them to a HeartSegmenter, classifies the
result, and draws it: scrolling waveform + envelope + marked beats, BPM, a
signal-quality indicator, and the triage label. Press 'm' to switch between file
playback and the live microphone.

Not a medical device. Not for clinical use.

CANNOT BE VISUALLY VERIFIED IN THIS SESSION: this container has no display and no
audio input device, so the animation has never actually been watched running. What
has been checked is everything upstream of the draw call -- segment.py and classify.py
are both under test and passing -- and a headless smoke test (tools/check_app.py) that
drives the real update loop for a fixed number of frames and asserts state advances
correctly and nothing raises. That is real coverage of the orchestration logic, but it
is not the same as having watched the plot move. Run this on a machine with a screen
before it goes in front of a judge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from classify import ClassifyResult, classify
from segment import HeartSegmenter, SegmentResult
from sources import AudioSource, FileSource, FRAME_SIZE, MicSource, SAMPLE_RATE

DISCLAIMER = "Not a medical device. Not for clinical use."

# Redraw cadence. A few frames per screen update trades a little display latency for
# noticeably less matplotlib overhead per second -- worth it on a laptop CPU that also
# owns the filter and the peak search.
FRAMES_PER_UPDATE = 4

BUFFER_S = 6.0

# UI-only color coding for the triage label -- has no bearing on classify.py's logic.
LABEL_COLOR = {
    "normal": "#2ca02c",
    "review recommended": "#d62728",
    "signal too noisy": "#7f7f7f",
}


class TriageApp:
    """Owns the live figure. All DSP happens in segment.py / classify.py / dsp.py --
    this class reads their outputs and draws them."""

    def __init__(self, source: AudioSource, mic_factory=None, file_source: AudioSource | None = None) -> None:
        self.source = source
        self.file_source = file_source  # kept so 'm' can switch back to it
        self.mic_factory = mic_factory  # callable, so a mic is only opened on demand
        self.segmenter = HeartSegmenter(sample_rate=source.sample_rate, buffer_s=BUFFER_S)

        self.result: SegmentResult | None = None
        self.classification: ClassifyResult | None = None
        self.frame_count = 0

        self._build_figure()

    def _build_figure(self) -> None:
        self.fig, (self.ax_wave, self.ax_env) = plt.subplots(
            2, 1, figsize=(11, 6), sharex=True, height_ratios=[1, 1]
        )
        self.fig.suptitle("Pulso -- auscultation triage (demo)", fontsize=13, fontweight="bold")

        (self.wave_line,) = self.ax_wave.plot([], [], color="#1f77b4", linewidth=0.7)
        self.ax_wave.set_ylabel("waveform")
        self.ax_wave.set_ylim(-1.05, 1.05)

        (self.env_line,) = self.ax_env.plot([], [], color="#ff7f0e", linewidth=1.0)
        self.beat_scatter = self.ax_env.scatter([], [], marker="v", zorder=5)
        self.ax_env.set_ylabel("envelope (z-score)")
        self.ax_env.set_xlabel("time (s)")
        self.ax_env.set_ylim(-2, 6)

        self.status_text = self.fig.text(
            0.02, 0.955, "", fontsize=11, fontweight="bold", va="top"
        )
        self.detail_text = self.fig.text(0.02, 0.915, "", fontsize=9, va="top", color="#444444")
        self.source_text = self.fig.text(
            0.98, 0.955, "", fontsize=9, va="top", ha="right", color="#444444"
        )
        self.fig.text(
            0.5, 0.01, DISCLAIMER, fontsize=9, ha="center", color="#b00020", fontweight="bold"
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.tight_layout(rect=(0, 0.03, 1, 0.90))

    def _on_key(self, event) -> None:
        if event.key == "m":
            self._toggle_source()
        elif event.key == "q":
            plt.close(self.fig)

    def _toggle_source(self) -> None:
        """One-key switch between file playback and the live mic, per CLAUDE.md's
        on-stage-fallback requirement. Failures here (no PortAudio, no device) are
        shown in the status line rather than raised -- a demo should never crash on a
        keypress.
        """
        try:
            if isinstance(self.source, MicSource):
                if self.file_source is None:
                    raise RuntimeError("no file source configured")
                self.source.close()
                self.source = self.file_source
            else:
                if self.mic_factory is None:
                    raise RuntimeError("no microphone configured")
                new_mic = self.mic_factory()
                self.source = new_mic
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: never crash on 'm'
            self._source_switch_error = str(exc)
            return
        self._source_switch_error = None
        self.segmenter.reset()

    def _source_label(self) -> str:
        if isinstance(self.source, MicSource):
            return "source: microphone  ['m' = file, 'q' = quit]"
        name = getattr(self.source, "path", "file")
        return f"source: {Path(name).name if name != 'file' else name}  ['m' = mic, 'q' = quit]"

    def _step(self) -> None:
        try:
            frame = self.source.read_frame()
        except StopIteration:
            return  # non-looping file finished; leave the last frame on screen

        self.result = self.segmenter.push(frame)
        self.classification = classify(self.result)
        self.frame_count += 1

    def _redraw(self, _frame_idx: int):
        for _ in range(FRAMES_PER_UPDATE):
            self._step()

        env = self.segmenter.envelope_buffer
        raw = self.segmenter.raw_buffer
        if env.size == 0:
            return self.wave_line, self.env_line, self.beat_scatter

        t = (np.arange(raw.size) + self.segmenter.buffer_start_sample) / self.segmenter.sample_rate

        self.wave_line.set_data(t, raw)
        self.ax_wave.set_xlim(t[0], t[-1] if t[-1] > t[0] else t[0] + 1)

        self.env_line.set_data(t, env)

        result, cls = self.result, self.classification
        if result and result.beats:
            beat_t = np.array([(b.sample_idx) / self.segmenter.sample_rate for b in result.beats])
            beat_y = np.array([b.amplitude for b in result.beats])
            colors = ["#1f77b4" if b.kind == "S1" else "#9467bd" if b.kind == "S2" else "#7f7f7f"
                      for b in result.beats]
            self.beat_scatter.set_offsets(np.column_stack([beat_t, beat_y]))
            self.beat_scatter.set_color(colors)
        else:
            self.beat_scatter.set_offsets(np.empty((0, 2)))

        self._update_status(result, cls)
        return self.wave_line, self.env_line, self.beat_scatter

    def _update_status(self, result: SegmentResult | None, cls: ClassifyResult | None) -> None:
        if cls is None:
            self.status_text.set_text("warming up...")
            self.status_text.set_color("#444444")
            self.detail_text.set_text("")
        else:
            self.status_text.set_text(cls.label.upper())
            self.status_text.set_color(LABEL_COLOR.get(cls.label, "#000000"))

            bpm_str = f"{result.bpm:.0f} bpm" if result and result.bpm else "-- bpm"
            s1s2_str = "" if (result and result.s1s2_confident) else "  (S1/S2 uncertain)"
            self.detail_text.set_text(f"{bpm_str}{s1s2_str}   {cls.reason}")

        label = self._source_label()
        if getattr(self, "_source_switch_error", None):
            label += f"   [switch failed: {self._source_switch_error}]"
        self.source_text.set_text(label)

    def run(self) -> None:
        self._anim = FuncAnimation(self.fig, self._redraw, interval=200, cache_frame_data=False)
        plt.show()


def build_app(file_path: str | None, use_mic: bool) -> TriageApp:
    file_source = FileSource(file_path, loop=True) if file_path else None

    def mic_factory() -> MicSource:
        return MicSource()

    if use_mic:
        try:
            initial: AudioSource = mic_factory()
        except RuntimeError as exc:
            print(f"Microphone unavailable ({exc}); starting from file instead.", file=sys.stderr)
            if file_source is None:
                raise
            initial = file_source
    else:
        if file_source is None:
            raise ValueError("no --file given and --mic not requested")
        initial = file_source

    return TriageApp(initial, mic_factory=mic_factory, file_source=file_source)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pulso live triage display")
    parser.add_argument("--file", type=str, default=None, help="WAV file to play back")
    parser.add_argument("--mic", action="store_true", help="start from the live microphone")
    args = parser.parse_args()

    if not args.file and not args.mic:
        parser.error("pass --file <wav> and/or --mic")

    app = build_app(args.file, args.mic)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
