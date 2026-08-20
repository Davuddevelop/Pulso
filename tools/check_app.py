"""Headless smoke test for app.py's orchestration loop.

This container has no display and no audio device, so app.py's animation has never
been watched running. This is the closest thing to coverage available here: it drives
the real TriageApp update loop (matplotlib Agg backend, no window) against a synthetic
WAV for a fixed number of frames, and asserts state actually advances -- BPM appears,
beats appear, the status text changes from "warming up" -- rather than just checking
that nothing raised.

It is not a substitute for watching the plot move on a real screen. Run it there
before a demo.

Run:  python tools/check_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app import TriageApp, build_app  # noqa: E402
from check_io import synth_pcg  # noqa: E402
from sources import FileSource, SAMPLE_RATE  # noqa: E402

PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    PASSED += 1
    print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))


def test_loop_advances_on_real_audio(wav_path: Path) -> None:
    print("update loop advances on a clean synthetic recording")
    app = TriageApp(FileSource(wav_path, loop=True))

    check("starts with no result", app.result is None)

    saw_bpm = False
    saw_beats = False
    saw_normal_label = False
    for i in range(40):
        app._redraw(i)
        if app.result and app.result.bpm:
            saw_bpm = True
        if app.result and app.result.beats:
            saw_beats = True
        if app.classification and app.classification.label == "normal":
            saw_normal_label = True

    check("bpm eventually appears", saw_bpm)
    check("beats eventually appear", saw_beats)
    check("classifies a clean 72 bpm signal as normal", saw_normal_label)
    check("status text was set", app.status_text.get_text() != "")
    check("triage label shown in status text", app.status_text.get_text() in
          ("NORMAL", "REVIEW RECOMMENDED", "SIGNAL TOO NOISY"))
    check("disclaimer is present on every render", any(
        "not a medical device" in t.get_text().lower() for t in app.fig.texts
    ))
    check("no action/routing text on a normal reading", app.action_text.get_text() == "")


def test_review_recommended_shows_action(tmp: Path) -> None:
    print("review recommended: routing action appears on screen, names no diagnosis")
    fast_signal, _, _ = synth_pcg(duration_s=15.0, bpm=150.0, systole_s=0.2)
    path = tmp / "fast.wav"
    sf.write(path, fast_signal, SAMPLE_RATE, subtype="PCM_16")

    app = TriageApp(FileSource(path, loop=True))
    saw_review = False
    for i in range(40):
        app._redraw(i)
        if app.classification and app.classification.label == "review recommended":
            saw_review = True

    check("150 bpm eventually flagged for review", saw_review)
    check("status text shows REVIEW RECOMMENDED", app.status_text.get_text() == "REVIEW RECOMMENDED")
    check("action text is populated", app.action_text.get_text() != "")
    check("action text is a routing suggestion, not a diagnosis", app.action_text.get_text().startswith("→"))
    forbidden = ["diagnos", "disease", "arrhythmia", "murmur", "cardiolog", "specialist"]
    hit = [w for w in forbidden if w in app.action_text.get_text().lower()]
    check("no forbidden vocabulary in the on-screen action text", not hit, f"found {hit}")


def test_display_buffers_stay_finite(wav_path: Path) -> None:
    print("no NaN/Inf ever reaches the plot")
    app = TriageApp(FileSource(wav_path, loop=True))
    for i in range(30):
        app._redraw(i)
        raw = app.segmenter.raw_buffer
        env = app.segmenter.envelope_buffer
        check(f"frame {i}: raw buffer finite", bool(np.all(np.isfinite(raw))) if raw.size else True)
        check(f"frame {i}: envelope buffer finite", bool(np.all(np.isfinite(env))) if env.size else True)


def test_source_toggle_falls_back_gracefully(wav_path: Path) -> None:
    """No PortAudio/no device in this container -- 'm' must degrade, not crash."""
    print("'m' key: mic unavailable here, must not crash the app")
    app = TriageApp(FileSource(wav_path, loop=True))
    app._on_key(type("Evt", (), {"key": "m"})())  # simulate a keypress event
    check("still has a usable source after a failed switch", app.source is not None)
    check("error was recorded for the status line", getattr(app, "_source_switch_error", None) is not None)

    for i in range(5):
        app._redraw(i)
    check("still renders after a failed switch attempt", app.result is not None)


def test_silence_shows_too_noisy(tmp: Path) -> None:
    print("silence -> SIGNAL TOO NOISY on screen, not a fabricated reading")
    silence = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)
    path = tmp / "silence.wav"
    sf.write(path, silence, SAMPLE_RATE, subtype="FLOAT")

    app = TriageApp(FileSource(path, loop=True))
    for i in range(20):
        app._redraw(i)

    check("classification is signal too noisy", app.classification is not None and
          app.classification.label == "signal too noisy", f"{app.classification}")
    check("status text shows it", app.status_text.get_text() == "SIGNAL TOO NOISY")
    check("no beats fabricated on silence", app.result is not None and len(app.result.beats) == 0)
    check("no action/routing text on a noisy reading", app.action_text.get_text() == "")


def test_build_app_without_mic_flag(wav_path: Path) -> None:
    print("build_app(file, mic=False) starts from the file, not the mic")
    app = build_app(str(wav_path), use_mic=False)
    check("starts on FileSource", isinstance(app.source, FileSource))
    check("mic_factory is wired for later 'm' presses", app.mic_factory is not None)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        signal, _, _ = synth_pcg(duration_s=15.0, bpm=72.0)
        wav_path = tmp / "demo.wav"
        sf.write(wav_path, signal, SAMPLE_RATE, subtype="PCM_16")

        test_loop_advances_on_real_audio(wav_path)
        test_display_buffers_stay_finite(wav_path)
        test_source_toggle_falls_back_gracefully(wav_path)
        test_review_recommended_shows_action(tmp)
        test_silence_shows_too_noisy(tmp)
        test_build_app_without_mic_flag(wav_path)

    print(f"\n{PASSED} checks passed.")
    print("\nReminder: this is a headless smoke test. The animation has not been")
    print("watched on a real screen in this environment. Verify visually before a demo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
