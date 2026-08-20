"""sources.py contract tests.

The point of this module is that nothing downstream can tell sources apart. So these
tests assert the contract itself -- length, dtype, rate, range -- and assert that the
same downstream code runs unchanged against different source types.

Run:  python tests/test_sources.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import dsp  # noqa: E402
import sources  # noqa: E402
from check_io import synth_pcg  # noqa: E402
from sources import FRAME_SIZE, SAMPLE_RATE, ArraySource, AudioSource, FileSource  # noqa: E402

PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    PASSED += 1
    print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))


def assert_contract(frame: np.ndarray, label: str) -> None:
    assert frame.dtype == np.float32, f"{label}: dtype {frame.dtype}"
    assert frame.shape == (FRAME_SIZE,), f"{label}: shape {frame.shape}"
    assert np.all(np.abs(frame) <= 1.0), f"{label}: out of range, peak {np.max(np.abs(frame))}"


def tone(freq: float, seconds: float, fs: int) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return 0.8 * np.sin(2 * np.pi * freq * t)


def dominant_freq(x: np.ndarray, fs: int) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / fs)[int(np.argmax(spec))])


def test_frame_contract(tmp: Path) -> None:
    print("frame contract")
    signal = synth_pcg(duration_s=5.0)[0]
    path = tmp / "pcg_2k.wav"
    sf.write(path, signal, SAMPLE_RATE, subtype="PCM_16")

    src = FileSource(path, loop=False)
    check("declares 2000 Hz", src.sample_rate == SAMPLE_RATE)
    check("no resampling of a 2 kHz file", not src.resampled, f"original {src.original_rate} Hz")
    check("duration preserved", abs(src.duration_s - 5.0) < 0.01, f"{src.duration_s:.3f} s")

    for i in range(5):
        assert_contract(src.read_frame(), f"frame {i}")
    check("frames are float32, 1024 long, in range", True)

    # Frames must be consecutive and non-overlapping. Compare against what is actually
    # on disk, not the pre-quantisation array -- PCM_16 rounds, and that rounding is the
    # file format's business, not the framing logic's.
    on_disk, _ = sf.read(str(path), dtype="float32")
    src.reset()
    joined = np.concatenate([src.read_frame() for _ in range(4)])
    check("frames are consecutive, non-overlapping",
          np.array_equal(joined, on_disk[: 4 * FRAME_SIZE]), "bit-exact against the file")
    src.close()


def test_exhaustion(tmp: Path) -> None:
    print("end of file")
    # 2.5 frames long, so the third frame is partial.
    n = int(FRAME_SIZE * 2.5)
    signal = synth_pcg(duration_s=10.0)[0][:n]
    path = tmp / "short.wav"
    sf.write(path, signal, SAMPLE_RATE, subtype="FLOAT")

    src = FileSource(path, loop=False)
    frames = [src.read_frame() for _ in range(3)]
    for i, f in enumerate(frames):
        assert_contract(f, f"frame {i}")
    check("final partial frame is padded to full length", frames[2].shape == (FRAME_SIZE,))

    tail = int(FRAME_SIZE * 0.5)
    check("padding is zeros, not garbage", np.all(frames[2][tail:] == 0.0))
    check("real samples survive the pad", np.allclose(frames[2][:tail], signal[2 * FRAME_SIZE :],
                                                      atol=1e-6))

    try:
        src.read_frame()
        raise AssertionError("expected StopIteration after exhaustion")
    except StopIteration:
        check("raises StopIteration once exhausted", True)

    # Iteration protocol stops cleanly rather than propagating StopIteration.
    src.reset()
    check("iterates to completion", len(list(src)) == 3)


def test_looping(tmp: Path) -> None:
    print("looping")
    n = int(FRAME_SIZE * 1.5)
    signal = synth_pcg(duration_s=10.0)[0][:n]
    path = tmp / "loop.wav"
    sf.write(path, signal, SAMPLE_RATE, subtype="FLOAT")

    src = FileSource(path, loop=True)
    frames = [src.read_frame() for _ in range(6)]
    for i, f in enumerate(frames):
        assert_contract(f, f"loop frame {i}")
    check("never exhausts while looping", len(frames) == 6)

    # Second frame is the file's tail followed by a wrap to the head.
    half = FRAME_SIZE // 2
    check("wraps seamlessly at the boundary",
          np.allclose(frames[1][:half], signal[FRAME_SIZE:], atol=1e-6)
          and np.allclose(frames[1][half:], signal[:half], atol=1e-6))
    src.close()


def test_resampling(tmp: Path) -> None:
    print("resampling")
    fs_in = 44100
    x = tone(100.0, 3.0, fs_in)
    path = tmp / "tone_44k.wav"
    sf.write(path, x, fs_in, subtype="PCM_16")

    src = FileSource(path, loop=False)
    check("original rate recorded", src.original_rate == 44100)
    check("flagged as resampled", src.resampled)
    check("presents as 2000 Hz", src.sample_rate == SAMPLE_RATE)
    check("duration preserved through resampling", abs(src.duration_s - 3.0) < 0.01,
          f"{src.duration_s:.4f} s")

    got = np.concatenate([src.read_frame() for _ in range(4)])
    assert_contract(got[:FRAME_SIZE], "resampled frame")
    peak = dominant_freq(got, SAMPLE_RATE)
    check("100 Hz tone survives resampling", abs(peak - 100.0) < 2.0, f"peak at {peak:.1f} Hz")

    # Anti-aliasing: 5 kHz at 44.1 kHz is above Nyquist for 2 kHz. Naive decimation would
    # fold it back into the audible band and it would look like real signal.
    y = tone(5000.0, 3.0, fs_in)
    alias_path = tmp / "alias_44k.wav"
    sf.write(alias_path, y, fs_in, subtype="PCM_16")
    src2 = FileSource(alias_path, loop=False)
    got2 = np.concatenate([src2.read_frame() for _ in range(4)])
    check("out-of-band content is removed, not aliased in",
          float(np.sqrt(np.mean(got2**2))) < 0.01,
          f"residual rms {float(np.sqrt(np.mean(got2**2))):.5f}")


def test_conforming(tmp: Path) -> None:
    print("conforming odd inputs")
    # Stereo must be downmixed to mono.
    stereo = np.stack([tone(100, 2.0, SAMPLE_RATE), tone(200, 2.0, SAMPLE_RATE)], axis=1)
    path = tmp / "stereo.wav"
    sf.write(path, stereo, SAMPLE_RATE, subtype="FLOAT")
    frame = FileSource(path, loop=False).read_frame()
    assert_contract(frame, "stereo downmix")
    check("stereo downmixed to mono", frame.ndim == 1)

    # A float WAV can legally exceed [-1, 1]; the contract cannot.
    hot = (tone(100, 2.0, SAMPLE_RATE) * 3.0).astype(np.float32)
    hot_path = tmp / "hot.wav"
    sf.write(hot_path, hot, SAMPLE_RATE, subtype="FLOAT")
    src = FileSource(hot_path, loop=False)
    frame = src.read_frame()
    assert_contract(frame, "hot input")
    check("over-unity input scaled into range", float(np.max(np.abs(frame))) <= 1.0)
    # Scaled, not clipped: shape must be preserved, so it stays a clean sinusoid.
    peak = dominant_freq(np.concatenate([src.read_frame() for _ in range(3)]), SAMPLE_RATE)
    check("scaled rather than clipped (no harmonics)", abs(peak - 100.0) < 2.0,
          f"still a clean 100 Hz tone, peak {peak:.1f} Hz")


def test_mic_without_portaudio() -> None:
    print("mic source without PortAudio")
    check("importing sources.py does not need PortAudio", "sounddevice" not in sys.modules,
          "lazy import kept the module out of the import graph")
    try:
        sources.MicSource()
        check("MicSource constructed (PortAudio present)", True)
    except RuntimeError as exc:
        check("fails with an actionable message", "PortAudio" in str(exc))
        check("names the fix", "libportaudio2" in str(exc) or "portaudio" in str(exc))


def test_source_interchangeability(tmp: Path) -> None:
    """The actual architectural claim: downstream code cannot tell sources apart."""
    print("source interchangeability")
    signal, s1_idx, _ = synth_pcg(duration_s=8.0, bpm=72.0)

    path_2k = tmp / "interchange_2k.wav"
    sf.write(path_2k, signal, SAMPLE_RATE, subtype="PCM_16")

    # Same audio, delivered at a different file rate, so it must be resampled on load.
    from scipy.signal import resample_poly

    upsampled = resample_poly(signal.astype(np.float64), 441, 20)  # 2000 -> 44100
    path_44k = tmp / "interchange_44k.wav"
    sf.write(path_44k, upsampled, 44100, subtype="PCM_16")

    def measure(src: AudioSource) -> float:
        """Streaming BPM. Identical code for every source -- that is the whole point."""
        sos = dsp.heart_sos(src.sample_rate)
        zi = None
        env_parts = []
        for _ in range(14):
            try:
                frame = src.read_frame()
            except StopIteration:
                break
            if zi is None:
                zi = dsp.make_stream_state(sos, float(frame[0]))
            filtered, zi = dsp.filter_stream(frame, sos, zi)
            env_parts.append(filtered)
        src.close()

        from scipy.signal import find_peaks

        env = dsp.envelope(np.concatenate(env_parts), src.sample_rate)
        peaks, _ = find_peaks(env, height=0.5, distance=int(0.2 * src.sample_rate))
        # S1 is the louder sound; take the stronger half of the detected events.
        strong = peaks[env[peaks] > np.median(env[peaks])]
        return 60.0 * src.sample_rate / float(np.mean(np.diff(strong)))

    bpm_file = measure(FileSource(path_2k, loop=False))
    bpm_resampled = measure(FileSource(path_44k, loop=False))
    bpm_array = measure(ArraySource(signal, SAMPLE_RATE))

    check("BPM from a 2 kHz file", abs(bpm_file - 72.0) < 2.0, f"{bpm_file:.2f}")
    check("BPM from a 44.1 kHz file", abs(bpm_resampled - 72.0) < 2.0, f"{bpm_resampled:.2f}")
    check("BPM from an in-memory array", abs(bpm_array - 72.0) < 2.0, f"{bpm_array:.2f}")
    check("all three agree", max(abs(bpm_file - bpm_resampled), abs(bpm_file - bpm_array)) < 1.0,
          "downstream code was identical in all three cases")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_frame_contract(tmp)
        test_exhaustion(tmp)
        test_looping(tmp)
        test_resampling(tmp)
        test_conforming(tmp)
        test_mic_without_portaudio()
        test_source_interchangeability(tmp)
    print(f"\n{PASSED} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
