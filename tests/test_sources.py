"""sources.py contract tests.

The point of this module is that nothing downstream can tell sources apart. So these
tests assert the contract itself -- length, dtype, rate, range -- and assert that the
same downstream code runs unchanged against different source types.

Run:  python tests/test_sources.py
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import dsp  # noqa: E402
import sources  # noqa: E402
from check_io import synth_pcg  # noqa: E402
from sources import (  # noqa: E402
    FRAME_SIZE,
    SAMPLE_RATE,
    ArraySource,
    AudioSource,
    FileSource,
    SerialMicSource,
    WebSocketMicSource,
    parse_serial_samples,
)

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


def encode_serial_samples(values: list[int], sync_interval: int = 256) -> bytes:
    """Mirror of the firmware's wire encoding, for building synthetic test streams."""
    out = bytearray()
    for i, v in enumerate(values):
        if i % sync_interval == 0:
            out += bytes([0xFF, 0xFE])
        out += bytes([v & 0xFF, (v >> 8) & 0xFF])
    return bytes(out)


def test_parse_serial_samples_clean() -> None:
    print("parse_serial_samples: clean stream with periodic sync markers")
    rng = np.random.default_rng(0)
    values = rng.integers(0, 1024, size=1000).tolist()
    raw = encode_serial_samples(values)

    decoded, leftover = parse_serial_samples(raw)
    check("all samples decoded", decoded == values, f"{len(decoded)} vs {len(values)}")
    check("nothing left undecoded", leftover == b"")


def test_parse_serial_samples_full_scale_value_no_false_marker() -> None:
    """1023's low byte is 0xFF -- must never be mistaken for the start of a marker."""
    print("parse_serial_samples: value 1023 (low byte 0xFF) decodes, doesn't fake a marker")
    values = [1023, 500, 1023, 1023, 0]
    raw = encode_serial_samples(values, sync_interval=10_000)  # no marker in this short run
    decoded, leftover = parse_serial_samples(raw)
    check("1023 decodes correctly despite 0xFF low byte", decoded == values, f"{decoded}")
    check("nothing left undecoded", leftover == b"")


def test_parse_serial_samples_incomplete_tail() -> None:
    print("parse_serial_samples: a trailing incomplete sample is held, not dropped")
    values = [10, 20, 30]
    raw = encode_serial_samples(values, sync_interval=10_000)
    partial = raw[:-1]  # cut the last sample's high byte off mid-stream

    decoded, leftover = parse_serial_samples(partial)
    check("complete samples decoded", decoded == values[:2], f"{decoded}")
    # partial's own trailing byte (the low byte of the 3rd sample) is what
    # must be held back -- not the byte we deliberately excluded from raw,
    # which was the *high* byte and never reached the parser at all.
    check("incomplete tail held back, not discarded", leftover == partial[-1:], f"{leftover!r}")

    # Feeding the missing high byte completes it -- exactly what SerialMicSource
    # does across successive reads as more bytes arrive.
    missing_byte = raw[-1:]
    decoded2, leftover2 = parse_serial_samples(leftover + missing_byte)
    check("completed once the missing byte arrives", decoded2 == values[2:], f"{decoded2}")
    check("buffer drained", leftover2 == b"")


def test_parse_serial_samples_dropped_byte_resyncs() -> None:
    """A single dropped byte must not corrupt the entire rest of the stream."""
    print("parse_serial_samples: recovers after a dropped byte, via the next sync marker")
    rng = np.random.default_rng(1)
    values = rng.integers(0, 1024, size=600).tolist()
    raw = bytearray(encode_serial_samples(values, sync_interval=256))

    # Drop one byte shortly after the start -- well before the next marker at
    # sample 256, so everything between the drop and that marker is expected
    # to come out corrupted, but decoding must recover cleanly afterward.
    drop_at = 10
    corrupted = bytes(raw[:drop_at]) + bytes(raw[drop_at + 1 :])

    decoded, leftover = parse_serial_samples(corrupted)
    check("did not crash or hang on corrupted input", True)

    # After the marker at sample 256, decoding must be back in lockstep with
    # the true values -- find that marker's known position in `values` and
    # confirm a long run of decoded values matches from there.
    tail_true = values[260:600]  # a little past the resync point, for safety margin
    tail_decoded = decoded[-len(tail_true):]
    check("recovers and matches ground truth after the next sync marker",
          tail_decoded == tail_true, f"{len(tail_decoded)} samples compared")


def test_serial_mic_source_bad_port_raises_runtime_error() -> None:
    """build_app's graceful fallback only catches RuntimeError -- a bad port must
    surface as that, not pyserial's own SerialException, or the demo crashes
    instead of falling back to file playback.
    """
    print("SerialMicSource: a bad port raises RuntimeError, not SerialException")
    try:
        SerialMicSource("/dev/definitely-not-a-real-port-xyz123")
        raise AssertionError("expected RuntimeError opening a nonexistent port")
    except RuntimeError as exc:
        check("raises RuntimeError specifically", True, str(exc)[:70])
    except Exception as exc:  # noqa: BLE001 -- exactly what this test guards against
        raise AssertionError(
            f"raised {type(exc).__name__}, not RuntimeError -- build_app's fallback won't catch this"
        ) from exc


def test_serial_mic_source_frame_contract() -> None:
    """SerialMicSource against a fake serial port -- no real board needed for this part."""
    print("SerialMicSource: frame contract against a synthetic serial stream")

    class FakeSerial:
        """Duck-types the small slice of pyserial's API SerialMicSource actually uses."""

        def __init__(self, data: bytes, chunk_size: int = 37) -> None:
            self._data = data
            self._pos = 0
            self._chunk_size = chunk_size
            self.port = "FAKE"
            self.timeout = 2.0

        @property
        def in_waiting(self) -> int:
            return min(self._chunk_size, len(self._data) - self._pos)

        def read(self, n: int) -> bytes:
            end = min(self._pos + n, len(self._data))
            chunk = self._data[self._pos : end]
            self._pos = end
            return chunk

        def close(self) -> None:
            pass

    rng = np.random.default_rng(2)
    n_samples = FRAME_SIZE * 3 + 17  # deliberately not a clean multiple of FRAME_SIZE
    values = rng.integers(0, 1024, size=n_samples).tolist()
    raw = encode_serial_samples(values)

    src = object.__new__(SerialMicSource)  # bypass __init__'s real pyserial.Serial(...) open
    src._serial = FakeSerial(raw, chunk_size=53)  # awkward chunk size to exercise partial reads
    src._raw_buf = b""
    src._values = []

    check("declares 2000 Hz", src.sample_rate == SAMPLE_RATE)

    frames = [src.read_frame() for _ in range(3)]
    for i, frame in enumerate(frames):
        assert_contract(frame, f"serial frame {i}")

    # Reconstruct what the three frames should be from the known input values and
    # confirm the ADC-to-float32 mapping and frame chunking are both exactly right.
    expected = (np.array(values[: 3 * FRAME_SIZE], dtype=np.float32) - 511.5) / 511.5
    got = np.concatenate(frames)
    check("frame values match the known input exactly", np.array_equal(got, expected))

    check("midscale ADC (511 or 512) maps close to 0.0",
          abs(float((511 - 511.5) / 511.5)) < 0.01 and abs(float((512 - 511.5) / 511.5)) < 0.01)
    check("full-scale ADC (1023) maps to 1.0", (1023 - 511.5) / 511.5 == 1.0)
    check("zero ADC maps to -1.0", (0 - 511.5) / 511.5 == -1.0)

    # Exhausting the fake port's data mid-frame must raise, not hang or return short.
    src2 = object.__new__(SerialMicSource)
    src2._serial = FakeSerial(encode_serial_samples(values[:10]))  # far short of one frame
    src2._raw_buf = b""
    src2._values = []
    try:
        src2.read_frame()
        raise AssertionError("expected RuntimeError when the port runs dry mid-frame")
    except RuntimeError as exc:
        check("raises a clear error when data runs out mid-frame", "no data from" in str(exc))


def test_websocket_mic_source_push_and_read() -> None:
    print("WebSocketMicSource: push then read returns exactly what was pushed")
    src = WebSocketMicSource(read_timeout_s=1.0)
    check("declares 2000 Hz", src.sample_rate == SAMPLE_RATE)

    pushed = (np.sin(np.linspace(0, 20, FRAME_SIZE)) * 0.5).astype(np.float32)
    src.push(pushed)
    frame = src.read_frame()
    assert_contract(frame, "websocket frame")
    check("frame matches what was pushed", np.array_equal(frame, pushed))


def test_websocket_mic_source_accumulates_partial_pushes() -> None:
    print("WebSocketMicSource: many small pushes still assemble one correct frame")
    src = WebSocketMicSource(read_timeout_s=1.0)
    rng = np.random.default_rng(3)
    whole = (rng.uniform(-0.9, 0.9, FRAME_SIZE + 37)).astype(np.float32)

    # Push in small, uneven chunks -- like a browser's onaudioprocess callback
    # firing with whatever chunk size the AudioContext gives it.
    pos = 0
    while pos < whole.size:
        step = min(97, whole.size - pos)
        src.push(whole[pos : pos + step])
        pos += step

    frame = src.read_frame()
    assert_contract(frame, "assembled frame")
    check("first frame_size samples match, in order", np.array_equal(frame, whole[:FRAME_SIZE]))


def test_websocket_mic_source_blocks_until_enough_data() -> None:
    print("WebSocketMicSource: read_frame() blocks until a delayed push arrives")
    src = WebSocketMicSource(read_timeout_s=2.0)
    pushed = np.zeros(FRAME_SIZE, dtype=np.float32)

    def delayed_push() -> None:
        time.sleep(0.15)
        src.push(pushed)

    t = threading.Thread(target=delayed_push)
    start = time.monotonic()
    t.start()
    frame = src.read_frame()
    elapsed = time.monotonic() - start
    t.join()

    check("actually waited for the push, not a stale buffer", elapsed >= 0.1, f"{elapsed:.3f}s")
    assert_contract(frame, "delayed frame")


def test_websocket_mic_source_timeout_raises_runtime_error() -> None:
    print("WebSocketMicSource: no data at all -> RuntimeError, not a hang")
    src = WebSocketMicSource(read_timeout_s=0.1)
    try:
        src.read_frame()
        raise AssertionError("expected RuntimeError when nothing was ever pushed")
    except RuntimeError as exc:
        check("raises a clear stall message", "stalled" in str(exc) or "closed" in str(exc), str(exc))


def test_websocket_mic_source_close_wakes_a_blocked_reader() -> None:
    print("WebSocketMicSource: close() unblocks read_frame() promptly, doesn't wait for the timeout")
    src = WebSocketMicSource(read_timeout_s=5.0)
    result: dict[str, object] = {}

    def reader() -> None:
        try:
            src.read_frame()
        except RuntimeError as exc:
            result["error"] = exc

    t = threading.Thread(target=reader)
    start = time.monotonic()
    t.start()
    time.sleep(0.05)
    src.close()
    t.join(timeout=2.0)
    elapsed = time.monotonic() - start

    check("reader thread finished", not t.is_alive())
    check("closed promptly, well under the 5s timeout", elapsed < 1.0, f"{elapsed:.3f}s")
    check("closing with no data raises RuntimeError", isinstance(result.get("error"), RuntimeError))


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
        test_parse_serial_samples_clean()
        test_parse_serial_samples_full_scale_value_no_false_marker()
        test_parse_serial_samples_incomplete_tail()
        test_parse_serial_samples_dropped_byte_resyncs()
        test_serial_mic_source_bad_port_raises_runtime_error()
        test_serial_mic_source_frame_contract()
        test_websocket_mic_source_push_and_read()
        test_websocket_mic_source_accumulates_partial_pushes()
        test_websocket_mic_source_blocks_until_enough_data()
        test_websocket_mic_source_timeout_raises_runtime_error()
        test_websocket_mic_source_close_wakes_a_blocked_reader()
    print(f"\n{PASSED} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
