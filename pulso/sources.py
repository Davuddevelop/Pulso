"""Audio input boundary.

This module is the ONLY place in the project that knows where audio comes from.
`dsp.py`, `segment.py`, `classify.py` and `app.py` must work identically whether
frames arrive from a WAV file, a laptop microphone, or (later) an ESP32 over
serial. If adding a new source would require editing any of those modules, the
abstraction is wrong -- fix the boundary, do not patch downstream.
"""

from abc import ABC, abstractmethod

import numpy as np

# Fixed project-wide. See CLAUDE.md: CinC 2016 is distributed at 2 kHz, so
# training and inference share a sample rate and nothing on the training path
# is ever resampled.
SAMPLE_RATE = 2000

# ~0.5 s. Enough envelope context for beat detection, still feels live.
FRAME_SIZE = 1024


class AudioSource(ABC):
    """A stream of fixed-length mono audio frames.

    Contract that every implementation must honour:

    - ``sample_rate`` is always ``SAMPLE_RATE`` (2000 Hz).
    - ``read_frame()`` returns float32, mono, 1-D, of a length that does not
      change over the lifetime of the source, with samples in [-1, 1].
    - ``read_frame()`` blocks until a frame is available. It does not return
      partial frames.
    - ``close()`` is idempotent and safe to call on an already-exhausted source.
    """

    sample_rate: int = SAMPLE_RATE

    @abstractmethod
    def read_frame(self) -> np.ndarray:
        """Return the next frame: float32, mono, fixed length, in [-1, 1]."""

    def close(self) -> None:
        """Release any underlying resource. Default: nothing to release."""
