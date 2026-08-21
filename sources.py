"""Audio capture layer.

This is the only module in the project that knows where audio comes from. Everything
downstream (dsp, segment, classify, app) receives fixed-length float32 frames at
SAMPLE_RATE and cannot tell a WAV file from a microphone from an ESP32.

The contract:

    source.sample_rate == 2000        always, for every source
    source.read_frame() -> np.ndarray  float32, mono, exactly FRAME_SIZE, in [-1, 1]

If adding a hardware source later requires editing any downstream module, the
abstraction is wrong and belongs back here.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

# Project-wide constants. CinC 2016 is distributed at 2 kHz, so there is no resampling
# on the training path. Nyquist is 1 kHz, which covers heart (20-200 Hz) and most lung
# (100-1000 Hz) content.
SAMPLE_RATE = 2000

# ~0.5 s. Enough envelope context for beat detection, still feels real-time.
FRAME_SIZE = 1024


class AudioSource(ABC):
    """A pull-based source of fixed-length audio frames."""

    sample_rate: int = SAMPLE_RATE
    frame_size: int = FRAME_SIZE

    @abstractmethod
    def read_frame(self) -> np.ndarray:
        """Return the next frame.

        Always exactly ``frame_size`` float32 samples, mono, within [-1, 1].
        Raises StopIteration when the source is permanently exhausted.
        """

    def close(self) -> None:
        """Release any held resources. Safe to call more than once."""

    def __enter__(self) -> "AudioSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self):
        while True:
            try:
                yield self.read_frame()
            except StopIteration:
                return


def _to_mono(x: np.ndarray) -> np.ndarray:
    """Average multi-channel audio down to one channel."""
    if x.ndim == 1:
        return x
    return x.mean(axis=1)


def _resample_to(x: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase resample. Exact rational ratio, no accumulating drift.

    resample_poly applies its own anti-aliasing FIR, which is why this is safe when
    downsampling from 44.1 kHz: the content above 1 kHz is filtered out before
    decimation instead of folding back into the heart band.
    """
    if orig_rate == target_rate:
        return x
    ratio = Fraction(target_rate, orig_rate).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator)


def _conform(x: np.ndarray, orig_rate: int) -> np.ndarray:
    """Bring arbitrary decoded audio into contract shape: mono, 2 kHz, float32, [-1, 1]."""
    x = _to_mono(np.asarray(x, dtype=np.float64))
    x = _resample_to(x, orig_rate, SAMPLE_RATE)

    # PCM decodes inside [-1, 1] already; float-subtype WAVs can overshoot. Scale rather
    # than clip -- clipping adds harmonics across the whole spectrum, including straight
    # into the heart band, and would look like signal to the envelope detector.
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        log.warning("input peaked at %.3f, scaling into [-1, 1]", peak)
        x = x / peak * 0.999

    return x.astype(np.float32, copy=False)


class FileSource(AudioSource):
    """Plays a WAV file as if it were arriving live.

    Resamples to SAMPLE_RATE on load if needed. This is deliberate: the ABC declares
    ``sample_rate == 2000``, so a source that could hand back 44.1 kHz would turn that
    attribute into something every caller has to check, and the rate knowledge this
    module exists to contain would leak straight into app.py and segment.py.

    ``original_rate`` is kept so the training path can assert it never resampled a
    record. CinC ships at 2 kHz, so for training data this is always a no-op.
    """

    def __init__(self, path: str | Path, loop: bool = True) -> None:
        self.path = Path(path)
        self.loop = loop

        raw, rate = sf.read(str(self.path), dtype="float64", always_2d=False)
        self.original_rate = int(rate)
        self.resampled = self.original_rate != SAMPLE_RATE
        if self.resampled:
            log.info(
                "%s: resampling %d Hz -> %d Hz", self.path.name, self.original_rate, SAMPLE_RATE
            )

        self._data = _conform(raw, self.original_rate)
        self._pos = 0
        self._exhausted = False

        if self._data.size == 0:
            raise ValueError(f"{self.path} decoded to zero samples")

    @property
    def duration_s(self) -> float:
        return self._data.size / SAMPLE_RATE

    def read_frame(self) -> np.ndarray:
        if self._exhausted:
            raise StopIteration

        end = self._pos + FRAME_SIZE
        frame = self._data[self._pos : end]

        if frame.size == FRAME_SIZE:
            self._pos = end
            return frame.copy()

        # Ran off the end. The contract promises a fixed length, so the tail is padded
        # rather than returned short -- a downstream window of unexpected size is the
        # kind of bug that only shows up on the last frame, on stage.
        if self.loop:
            # Wrap: fill the remainder from the top of the file so playback is seamless.
            need = FRAME_SIZE - frame.size
            frame = np.concatenate([frame, self._data[:need]])
            self._pos = need
            return frame

        padded = np.zeros(FRAME_SIZE, dtype=np.float32)
        padded[: frame.size] = frame
        self._pos = self._data.size
        self._exhausted = True  # next call raises
        return padded

    def reset(self) -> None:
        self._pos = 0
        self._exhausted = False


class MicSource(AudioSource):
    """Live capture from the default input device.

    sounddevice is imported here rather than at module scope on purpose: PortAudio is a
    system library that is simply absent on CI runners, containers, and any laptop that
    has not installed it. An eager import would make ``import sources`` fail everywhere,
    taking file playback down with it.

    Uses the callback API with a bounded buffer that drops the *oldest* frames under
    pressure. For a live triage display that is the correct failure mode -- stale audio
    is worth less than current audio -- but it is a real data loss, so it is counted and
    exposed via ``dropped_frames`` for the quality indicator to report honestly.
    """

    def __init__(self, device: int | str | None = None, buffer_frames: int = 8) -> None:
        try:
            import sounddevice as sd
        except OSError as exc:  # PortAudio missing
            raise RuntimeError(
                "Microphone capture needs the PortAudio system library "
                "(apt: libportaudio2, brew: portaudio). File playback works without it."
            ) from exc

        from collections import deque
        from threading import Lock

        self._sd = sd
        self._buf: deque[np.ndarray] = deque(maxlen=buffer_frames)
        self._lock = Lock()
        self.dropped_frames = 0

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            if status:
                log.debug("input stream status: %s", status)
            block = np.asarray(indata[:, 0], dtype=np.float32).copy()
            with self._lock:
                if len(self._buf) == self._buf.maxlen:
                    self.dropped_frames += 1  # deque discards the oldest
                self._buf.append(block)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            channels=1,
            dtype="float32",
            device=device,
            callback=callback,
        )
        self._stream.start()

    def read_frame(self) -> np.ndarray:
        """Block until a frame is available.

        Never raises StopIteration -- a microphone is not exhaustible. If the device
        stalls, this returns silence rather than hanging the UI thread forever.
        """
        import time

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._buf:
                    frame = self._buf.popleft()
                    break
            time.sleep(0.002)
        else:
            log.warning("input stalled, returning silence")
            return np.zeros(FRAME_SIZE, dtype=np.float32)

        # PortAudio can hand back a short final block on device reconfiguration.
        if frame.size != FRAME_SIZE:
            padded = np.zeros(FRAME_SIZE, dtype=np.float32)
            padded[: min(frame.size, FRAME_SIZE)] = frame[:FRAME_SIZE]
            frame = padded

        return np.clip(frame, -1.0, 1.0)

    def close(self) -> None:
        stream = getattr(self, "_stream", None)
        if stream is not None:
            stream.stop()
            stream.close()
            self._stream = None


class ArraySource(AudioSource):
    """Wraps an in-memory array. For tests and for the synthetic ground-truth signal."""

    def __init__(self, data: np.ndarray, sample_rate: int = SAMPLE_RATE, loop: bool = False) -> None:
        self._data = _conform(np.asarray(data), sample_rate)
        self.loop = loop
        self._pos = 0
        self._exhausted = False

    read_frame = FileSource.read_frame
    reset = FileSource.reset


# ---------------------------------------------------------------------------
# Serial hardware capture (ELEGOO Mega 2560 running hardware/pulso_mic.ino)
# ---------------------------------------------------------------------------
#
# Wire protocol (see the firmware's own docstring for the full rationale):
# two bytes per sample, little-endian, a 10-bit ADC value so the high byte is
# always in [0, 3]. That leaves 0xFF as a byte no real sample's high byte can
# ever take, so the firmware periodically prefixes a sample with the 2-byte
# marker 0xFF 0xFE, and the parser below treats any (0xFF, 0xFE) pair as an
# unambiguous resync point -- it can never collide with real data, because no
# legitimate high byte reaches 0xFE (254 > 3).

SERIAL_SYNC = bytes([0xFF, 0xFE])
SERIAL_BAUD = 115200


def parse_serial_samples(buf: bytes) -> tuple[list[int], bytes]:
    """Decode as many complete ADC samples as possible from a raw byte buffer.

    Pure function, no I/O -- this is what lets the resync logic be tested
    against synthetic corrupted streams without the real board. Self-healing:
    a marker is always skipped, and any byte pair that cannot be a real
    sample (high byte > 3) is treated as a misalignment and walked past one
    byte at a time until back in step, rather than raising or losing the
    whole buffer to one dropped byte.

    Returns (samples, leftover_undecoded_bytes) -- the leftover is always
    handed back to the next call once more bytes have arrived, never dropped.
    """
    samples: list[int] = []
    i = 0
    n = len(buf)
    while i + 1 < n:
        if buf[i] == 0xFF and buf[i + 1] == 0xFE:
            i += 2
            continue
        low, high = buf[i], buf[i + 1]
        if high > 3:
            i += 1  # not a valid sample and not a marker start -- resync by one byte
            continue
        samples.append(low | (high << 8))
        i += 2
    return samples, buf[i:]


class SerialMicSource(AudioSource):
    """Live capture from the Mega 2560 firmware over USB serial.

    UNTESTED against the real board -- there is no hardware in this build
    environment. parse_serial_samples() is unit-tested against synthetic
    (including deliberately corrupted) byte streams in tests/test_sources.py,
    which is real coverage of the protocol logic, but it is not the same as
    having actually read from the serial port. Verify with
    tools/check_serial_source.py before wiring this into app.py.

    pyserial is imported lazily, same reasoning as MicSource's sounddevice
    import: the library is a new dependency this feature needs, and a
    machine without it (or without the board plugged in) must still be able
    to `import sources` and use FileSource.
    """

    def __init__(self, port: str, baud: int = SERIAL_BAUD, timeout_s: float = 2.0) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "Serial capture needs pyserial (pip install pyserial). "
                "File playback works without it."
            ) from exc

        try:
            self._serial = serial.Serial(port, baud, timeout=timeout_s)
        except serial.SerialException as exc:
            # Wrapped into the same RuntimeError MicSource uses for its own
            # open failures, so callers (app.py's build_app) can catch one
            # exception type and fall back to file playback regardless of
            # which live source failed to open.
            raise RuntimeError(f"could not open {port}: {exc}") from exc

        self._raw_buf = b""      # undecoded tail bytes (an incomplete sample/marker)
        self._values: list[int] = []  # decoded samples not yet claimed by a frame

    def read_frame(self) -> np.ndarray:
        while len(self._values) < FRAME_SIZE:
            chunk = self._serial.read(max(1, self._serial.in_waiting or 1))
            if not chunk:
                raise RuntimeError(
                    f"no data from {self._serial.port} within {self._serial.timeout}s "
                    "-- board unplugged, wrong port, or firmware not running"
                )
            self._raw_buf += chunk
            decoded, self._raw_buf = parse_serial_samples(self._raw_buf)
            self._values.extend(decoded)

        frame_values, self._values = self._values[:FRAME_SIZE], self._values[FRAME_SIZE:]

        # 10-bit ADC (0-1023) -> float32 [-1, 1]. 511.5 is the exact midpoint,
        # so a mid-scale reading maps to 0.0 rather than a small fixed offset.
        frame = (np.array(frame_values, dtype=np.float32) - 511.5) / 511.5
        return np.clip(frame, -1.0, 1.0)

    def close(self) -> None:
        serial_conn = getattr(self, "_serial", None)
        if serial_conn is not None:
            serial_conn.close()
            self._serial = None
