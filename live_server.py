"""Live demo server: phone mic (browser) -> WebSocket -> the real triage pipeline -> dashboard.

Serves five routes:
    GET  /            the landing page (docs/index.html) -- its "Start Live Check" button
                      links to /live.
    GET  /pricing.html docs/pricing.html -- business model and planned pricing, linked
                      from the landing page nav. Explicit route because Vercel's static
                      hosting serves everything under docs/ automatically, but this
                      aiohttp server only serves what it's told to.
    GET  /live        web/live.html -- captures the browser's mic and renders the live
                      dashboard.
    WS   /ws        receives raw PCM float32 chunks at whatever rate the browser's
                    AudioContext used, resamples to SAMPLE_RATE the same way FileSource
                    resamples a WAV, and streams back JSON triage results.
    WS   /ws-sample plays web/sample_normal.wav (a synthetic, clean 72bpm recording,
                    same generator tests/tools use) through the same segmenter/
                    classifier at real-time pace, for the "Play a sample reading"
                    fallback -- an always-working demo path when a phone mic can't get
                    a clean signal (a real, likely failure mode: phone mics filter out
                    the low frequencies heart sounds live in). Every payload is tagged
                    "sample": true so the UI never confuses it with a live listen.

Uses the exact same dsp/segment/classify modules as app.py -- no signal-processing
or classification logic is duplicated here, only orchestration. The output vocabulary
and safety constraints (normal / review recommended / signal too noisy; generic
routing only, never a diagnosis) are unchanged because classify.py is unchanged.

Run:
    pip install aiohttp   (see requirements.txt)
    python live_server.py
    # in a second terminal, from the venue's internet connection:
    cloudflared tunnel --url http://localhost:8765
    #   or: ngrok http 8765
    # open the printed https://... URL on the phone.

Browsers only grant microphone access on localhost or a page served over HTTPS --
plain http://<laptop-ip>:8765 will silently fail on a phone. The tunnel is what
supplies the HTTPS.

UNTESTED against a real phone/browser in this environment -- no browser, network
peer, or microphone is available here. The resample-and-classify path is exercised
in tests/test_live_server.py against synthetic Float32 chunks pushed the same way a
browser would send them; the actual browser<->server round trip (mic permission
prompts, AudioContext quirks per phone/browser, tunnel behavior) has never been
watched running. Verify with a real phone before a demo.

Not a medical device. Not for clinical use.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from fractions import Fraction
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web
from scipy.signal import resample_poly

from classify import classify
from segment import HeartSegmenter
from sources import SAMPLE_RATE, FRAME_SIZE, FileSource, WebSocketMicSource

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
WEB_DIR = ROOT / "web"
SAMPLE_WAV = WEB_DIR / "sample_normal.wav"

# Sanity bound on the handshake's declared sample_rate -- rejects a malformed or
# hostile handshake before it reaches resample_poly with a nonsense ratio.
MIN_BROWSER_RATE = 4000
MAX_BROWSER_RATE = 192000

# How often the processing loop pushes a result back, independent of how often the
# browser's audio chunks arrive -- keeps the dashboard update rate steady even if
# network delivery is bursty.
RESULT_INTERVAL_S = 0.5


def _resample_chunk(chunk: np.ndarray, orig_rate: int) -> np.ndarray:
    """Same polyphase approach as sources.py's _resample_to, applied per chunk.

    Resampling independently-arriving chunks (rather than one continuous buffer,
    which is what _resample_to assumes) introduces small artifacts at each chunk
    boundary -- a known, accepted tradeoff for live streaming, not something to
    silently hide. HeartSegmenter's bandpass + envelope + refractory logic already
    has to tolerate real-world noise; this is more of that, not a new class of
    problem.
    """
    if orig_rate == SAMPLE_RATE:
        return chunk.astype(np.float32, copy=False)
    ratio = Fraction(SAMPLE_RATE, orig_rate).limit_denominator(1000)
    out = resample_poly(chunk, ratio.numerator, ratio.denominator)
    return out.astype(np.float32, copy=False)


async def index(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(DOCS_DIR / "index.html")


async def pricing_page(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(DOCS_DIR / "pricing.html")


async def live_page(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(WEB_DIR / "live.html")


def _count_new_beats(beats, last_idx: int) -> tuple[int, int]:
    """How many of `beats` are genuinely new since the last push, by absolute sample index.

    HeartSegmenter recomputes beats over a trailing window on every push, so the same
    beat reappears in several consecutive results as the window slides past it -- a
    naive `len(result.beats)` is a windowed snapshot, not a running total, and visibly
    jumps around rather than counting up. Comparing against the highest sample_idx
    already counted gives a real cumulative count that can only grow.
    """
    if not beats:
        return 0, last_idx
    new_count = sum(1 for b in beats if b.sample_idx > last_idx)
    return new_count, max(b.sample_idx for b in beats)


def _build_payload(segmenter: HeartSegmenter, result, cls, *, beats_total: int, sample: bool = False) -> dict:
    payload = {
        "bpm": result.bpm,
        "beats": len(result.beats),
        "beats_total": beats_total,
        "s1s2_confident": result.s1s2_confident,
        "quality": result.quality,
        "quality_reason": result.quality_reason,
        "label": cls.label,
        "reason": cls.reason,
        "urgency": cls.urgency,
        "action": cls.action,
        "waveform": segmenter.raw_buffer[-SAMPLE_RATE * 4 :].tolist(),
    }
    if sample:
        # Lets the client tell a pre-recorded playback apart from a live listen --
        # never fold this into the same UI state as a real reading.
        payload["sample"] = True
    return payload


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=1 * 1024 * 1024)
    await ws.prepare(request)

    source = WebSocketMicSource()
    segmenter = HeartSegmenter(sample_rate=SAMPLE_RATE)
    browser_rate: int | None = None

    async def process_loop() -> None:
        loop = asyncio.get_event_loop()
        beats_total = 0
        last_beat_idx = -1
        while not ws.closed:
            try:
                frame = await loop.run_in_executor(None, source.read_frame)
            except RuntimeError as exc:
                log.info("processing loop stopping: %s", exc)
                return
            result = segmenter.push(frame)
            cls = classify(result)
            new_count, last_beat_idx = _count_new_beats(result.beats, last_beat_idx)
            beats_total += new_count
            payload = _build_payload(segmenter, result, cls, beats_total=beats_total)
            if not ws.closed:
                await ws.send_json(payload)
            await asyncio.sleep(RESULT_INTERVAL_S)

    processor = asyncio.ensure_future(process_loop())

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "hello":
                    rate = int(data.get("sample_rate", 0))
                    if not (MIN_BROWSER_RATE <= rate <= MAX_BROWSER_RATE):
                        await ws.close(code=1008, message=b"bad sample_rate")
                        break
                    browser_rate = rate
                    log.info("client declared sample_rate=%d", rate)
            elif msg.type == WSMsgType.BINARY:
                if browser_rate is None:
                    continue  # browser always sends {"type":"hello"} first; drop stray audio
                chunk = np.frombuffer(msg.data, dtype=np.float32)
                source.push(_resample_chunk(chunk, browser_rate))
            elif msg.type == WSMsgType.ERROR:
                log.warning("ws error: %s", ws.exception())
    finally:
        source.close()
        processor.cancel()
        try:
            await processor
        except asyncio.CancelledError:
            pass

    return ws


async def sample_websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """Plays SAMPLE_WAV through the real segmenter/classifier at real-time pace.

    No audio comes from the client at all here -- connecting is the only signal
    needed. Paced with an explicit sleep because FileSource.read_frame() returns
    instantly (it's not real hardware); without the sleep this would blast through
    the whole file and flood the client in well under a second.

    Plays the recording once, not on a loop -- looping forever meant the demo never
    reached a conclusion and could restart mid-word during a pitch. The final payload
    is tagged "complete" so the client can show a clean end state instead of an
    unexplained disconnect.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    source = FileSource(SAMPLE_WAV, loop=False)
    segmenter = HeartSegmenter(sample_rate=SAMPLE_RATE)
    frame_period_s = FRAME_SIZE / SAMPLE_RATE
    beats_total = 0
    last_beat_idx = -1

    last_payload = None

    try:
        while not ws.closed:
            try:
                frame = source.read_frame()
            except StopIteration:
                break
            result = segmenter.push(frame)
            cls = classify(result)
            new_count, last_beat_idx = _count_new_beats(result.beats, last_beat_idx)
            beats_total += new_count
            last_payload = _build_payload(segmenter, result, cls, beats_total=beats_total, sample=True)
            await ws.send_json(last_payload)
            await asyncio.sleep(frame_period_s)
        else:
            # Loop only exited because the client disconnected, not because the
            # recording finished -- nothing left to tell them.
            last_payload = None

        if last_payload is not None and not ws.closed:
            await ws.send_json({**last_payload, "complete": True})
    except ConnectionResetError:
        pass
    finally:
        source.close()

    return ws


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/pricing.html", pricing_page)
    app.router.add_get("/live", live_page)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/ws-sample", sample_websocket_handler)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Pulso live demo server (phone mic over WebSocket)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("Serving on http://%s:%d -- tunnel this for phone HTTPS access, see module docstring", args.host, args.port)
    web.run_app(build_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
