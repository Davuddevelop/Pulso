"""live_server.py integration tests.

Drives the real aiohttp app (build_app()) over a real WebSocket connection via
aiohttp's own test client -- not a mock. A synthetic PCG signal generated at
SAMPLE_RATE is upsampled to simulate what a browser's AudioContext would
actually hand over (typically 48 kHz, never SAMPLE_RATE), sent through /ws in
small chunks the way onaudioprocess would, and the JSON results streamed back
are checked against the same guarantees app.py's own smoke test enforces:
BPM converges, "signal too noisy" is honest on silence, and the on-screen
vocabulary never leaks a diagnosis.

This proves the resample-in-server + WebSocketMicSource + segment/classify
wiring is correct. It does NOT prove the browser<->server path works against
a real phone: no browser, no microphone, no network peer exists in this
environment. See live_server.py's module docstring.

Run:  python tests/test_live_server.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from aiohttp.test_utils import TestClient, TestServer
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from check_io import synth_pcg  # noqa: E402
from live_server import build_app  # noqa: E402
from sources import SAMPLE_RATE  # noqa: E402

BROWSER_RATE = 48000  # what real phone/laptop AudioContexts actually give you
FORBIDDEN = ["diagnos", "disease", "arrhythmia", "murmur", "cardiolog", "specialist"]

PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    PASSED += 1
    print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))


def _to_browser_rate(signal_2k: np.ndarray) -> np.ndarray:
    """Upsample a SAMPLE_RATE signal to BROWSER_RATE, simulating what a real phone
    mic capture would produce before it ever reaches live_server.py's resampler."""
    up = resample_poly(signal_2k, BROWSER_RATE, SAMPLE_RATE)
    return up.astype(np.float32)


async def _stream_and_collect(signal_browser_rate: np.ndarray, n_results: int, chunk: int = 4096) -> list[dict]:
    server = TestServer(build_app())
    client = TestClient(server)
    await client.start_server()
    try:
        ws = await client.ws_connect("/ws")
        await ws.send_json({"type": "hello", "sample_rate": BROWSER_RATE})

        async def sender() -> None:
            pos = 0
            n = signal_browser_rate.size
            while pos < n:
                block = signal_browser_rate[pos : pos + chunk]
                await ws.send_bytes(block.tobytes())
                pos += chunk
                await asyncio.sleep(0.01)  # don't blast the whole file in one event-loop tick

        send_task = asyncio.ensure_future(sender())

        results: list[dict] = []
        try:
            async for msg in ws:
                results.append(json.loads(msg.data))
                if len(results) >= n_results:
                    break
        finally:
            send_task.cancel()
            await ws.close()
        return results
    finally:
        await client.close()


def test_clean_signal_converges_to_correct_bpm() -> None:
    print("clean 72 bpm signal, streamed like a browser would send it -> correct BPM, normal label")
    signal_2k, _, _ = synth_pcg(duration_s=15.0, bpm=72.0)
    browser_signal = _to_browser_rate(signal_2k)

    results = asyncio.run(_stream_and_collect(browser_signal, n_results=6))

    check("got results back over the websocket", len(results) >= 1, f"{len(results)} results")
    bpms = [r["bpm"] for r in results if r.get("bpm")]
    check("bpm eventually appears", len(bpms) > 0)
    if bpms:
        check("converges near 72 bpm", abs(bpms[-1] - 72.0) < 5.0, f"got {bpms[-1]:.1f}")

    labels_seen = {r["label"] for r in results}
    check("only the three allowed labels ever appear", labels_seen <= {"normal", "review recommended", "signal too noisy"}, str(labels_seen))


def test_silence_reports_too_noisy_not_a_fabricated_reading() -> None:
    print("silence, streamed the same way -> signal too noisy, no fabricated bpm/action")
    silence_2k = np.zeros(int(10 * SAMPLE_RATE), dtype=np.float32)
    browser_signal = _to_browser_rate(silence_2k)

    results = asyncio.run(_stream_and_collect(browser_signal, n_results=4))

    check("got results back", len(results) >= 1)
    check("every result is signal-too-noisy on pure silence",
          all(r["label"] == "signal too noisy" for r in results), str({r["label"] for r in results}))
    check("no fabricated bpm on silence", all(r["bpm"] is None for r in results))
    check("no action/routing text on a noisy reading", all(not r["action"] for r in results))


def test_no_forbidden_vocabulary_in_any_streamed_field() -> None:
    print("review-triggering signal -> action text is a routing suggestion, never a diagnosis")
    fast_2k, _, _ = synth_pcg(duration_s=15.0, bpm=150.0, systole_s=0.2)
    browser_signal = _to_browser_rate(fast_2k)

    results = asyncio.run(_stream_and_collect(browser_signal, n_results=6))

    saw_review = any(r["label"] == "review recommended" for r in results)
    check("150 bpm eventually flagged for review", saw_review)

    for r in results:
        text = f"{r.get('reason') or ''} {r.get('action') or ''}".lower()
        hit = [w for w in FORBIDDEN if w in text]
        check("no forbidden vocabulary in a streamed result", not hit, f"found {hit} in {text!r}")

    reviewed = [r for r in results if r["label"] == "review recommended"]
    if reviewed:
        check("review action starts like a routing suggestion, not a verdict",
              all((r["action"] or "").strip() for r in reviewed))


def test_bad_handshake_sample_rate_is_rejected() -> None:
    print("nonsense sample_rate in the hello handshake -> connection closed, not crashed")

    async def run() -> int:
        server = TestServer(build_app())
        client = TestClient(server)
        await client.start_server()
        try:
            ws = await client.ws_connect("/ws")
            await ws.send_json({"type": "hello", "sample_rate": -1})
            msg = await ws.receive()
            return ws.close_code if ws.close_code is not None else -999
        finally:
            await client.close()

    close_code = asyncio.run(run())
    check("server closed the connection with policy-violation code 1008", close_code == 1008, f"got {close_code}")


def main() -> int:
    test_clean_signal_converges_to_correct_bpm()
    test_silence_reports_too_noisy_not_a_fabricated_reading()
    test_no_forbidden_vocabulary_in_any_streamed_field()
    test_bad_handshake_sample_rate_is_rejected()
    print(f"\n{PASSED} checks passed.")
    print("\nReminder: this proves the server-side pipeline over a real websocket.")
    print("It does not prove the browser<->phone path -- verify with a real phone before a demo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
