"""Verify the Mega 2560 + hardware/pulso_mic.ino wiring, standalone.

Run this before wiring SerialMicSource into app.py. It reads raw samples,
prints basic stats (mean, range, how much of the ADC's 0-1023 range you're
actually using), and saves a plot -- catches the most common problems (wrong
port, firmware not flashed, no signal, DC bias parked at a rail) without
needing the rest of the pipeline running.

Usage:
    python tools/check_serial_source.py /dev/ttyUSB0
    python tools/check_serial_source.py COM3          (Windows)

Find your port:
    macOS/Linux: ls /dev/tty.* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
    Windows:     Device Manager -> Ports (COM & LPT)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sources import SAMPLE_RATE, SerialMicSource  # noqa: E402

OUT_DIR = ROOT / "out"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    port = sys.argv[1]

    print(f"Opening {port} at 115200 baud...")
    try:
        src = SerialMicSource(port)
    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 -- report whatever pyserial raised, plainly
        print(f"FAILED to open the port: {exc}")
        print("Check: right port name? board plugged in? firmware actually flashed?")
        return 1

    print("Reading 3 seconds of audio...")
    seconds = 3.0
    n_frames = int(seconds * SAMPLE_RATE / src.frame_size) + 1
    chunks = []
    try:
        for i in range(n_frames):
            chunks.append(src.read_frame())
            print(f"  frame {i + 1}/{n_frames}", end="\r")
        print()
    except RuntimeError as exc:
        print(f"\nFAILED mid-read: {exc}")
        return 1
    finally:
        src.close()

    signal = np.concatenate(chunks)
    adc_raw = signal * 511.5 + 511.5  # undo the float32 mapping, back to 0-1023 for diagnostics

    print(f"\n{signal.size} samples read ({signal.size / SAMPLE_RATE:.2f}s at {SAMPLE_RATE} Hz)")
    print(f"  ADC range used   : {adc_raw.min():.0f} - {adc_raw.max():.0f}  (full scale is 0-1023)")
    print(f"  ADC mean (bias)  : {adc_raw.mean():.0f}  (511.5 is dead-center)")
    print(f"  signal peak      : {np.max(np.abs(signal)):.3f}  (of a possible 1.0)")
    print(f"  signal RMS       : {float(np.sqrt(np.mean(signal**2))):.4f}")

    span = adc_raw.max() - adc_raw.min()
    if span < 5:
        print("\n  WARNING: almost no variation. Check: sensor connected? contact made?")
    elif adc_raw.mean() < 30 or adc_raw.mean() > 993:
        print("\n  WARNING: signal is pinned near a rail (0 or 1023). If this is a bare")
        print("  piezo disc, it likely needs the bias-resistor divider described in")
        print("  hardware/pulso_mic.ino's wiring comment -- without it, only one half")
        print("  of the AC swing registers and the rest clips flat.")
    else:
        print("\n  Looks like a live, moving signal. Good sign -- try app.py next.")

    OUT_DIR.mkdir(exist_ok=True)
    t = np.arange(signal.size) / SAMPLE_RATE
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(t, signal, linewidth=0.7, color="#1f77b4")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title(f"Raw serial capture from {port}")
    ax.margins(x=0)
    fig.tight_layout()
    plot_path = OUT_DIR / "check_serial_source.png"
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"\nSaved a plot to {plot_path} -- look for beat-shaped bumps, not just noise.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
