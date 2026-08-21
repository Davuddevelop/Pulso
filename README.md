# Pulso — auscultation triage MVP

> **Not a medical device. Not for clinical use.**
> Pulso produces a *triage signal* for a community health worker. It does not
> diagnose, name conditions, or suggest treatment.

A low-cost digital stethoscope triage tool. A contact microphone captures heart and
lung sounds, streams them to a laptop, and the software flags patterns worth a second
listen.

Output vocabulary is fixed and deliberately narrow:

| Output | Meaning |
|---|---|
| `normal` | Nothing in the signal met the review threshold. |
| `review recommended` | Something met the threshold. A human should listen. |
| `signal too noisy` | The input is not good enough to say anything at all. |

There is no fourth option, and no accuracy figure is quoted anywhere unless it was
measured on a held-out set.

## Status

Hardware exists now: an ELEGOO Mega 2560 and a sensor front end (see **Hardware**,
below). `SerialMicSource` is written to the same `AudioSource` contract as the file and
mic sources, so nothing downstream had to change — but it has not been verified against
the real board, since this development environment has neither. Everything runs from
recorded audio files by default, and the capture layer is abstracted so a hardware
source drops in without touching any downstream module — which is the actual design
claim `SerialMicSource` is meant to test, not just describe.

## Architecture

Everything downstream of audio capture is ignorant of where the audio came from.

```
sources.py    AudioSource ABC -> FileSource, MicSource, SerialMicSource
dsp.py        pure functions, no state, no I/O -> filters, envelope
segment.py    stateful -> beat detection, S1/S2 labelling, heart rate
classify.py   window in -> {label, confidence} out
app.py        UI / orchestration only. No DSP logic lives here.
```

The contract every source honours:

```python
class AudioSource(ABC):
    sample_rate: int          # always 2000
    @abstractmethod
    def read_frame(self) -> np.ndarray: ...   # float32, mono, fixed length, [-1, 1]
    def close(self) -> None: ...
```

If adding hardware later would require editing `dsp.py`, `segment.py`, `classify.py`
or `app.py`, the abstraction is wrong.

## Fixed technical decisions

| Decision | Value | Reason |
|---|---|---|
| Internal sample rate | 2000 Hz | CinC 2016 ships at 2 kHz; no resampling on the training path. Nyquist 1 kHz covers heart (20–200 Hz) and most lung (100–1000 Hz). |
| Frame size | 1024 samples (~0.5 s) | Enough envelope context for beat detection, still feels real-time. |
| Heart bandpass | 25–200 Hz, Butterworth order 4 | Below 25 Hz is handling noise and DC drift; above 200 Hz is not heart sound. |
| Lung bandpass | 100–**950** Hz, Butterworth order 4 | Standard respiratory band is quoted as 100–1000 Hz, but at fs = 2000 Hz, 1000 Hz *is* Nyquist and is not a realisable digital filter edge (the design needs 0 < Wn < 1). See note below. |
| Filter application | `sosfiltfilt` offline, `sosfilt` + persistent `zi` streaming | Order-4 IIR in `ba` form is numerically unstable. SOS only. |
| Envelope | Shannon energy, then ~50 ms moving average | Standard for PCG; suppresses low-amplitude noise better than rectification. |
| Audio I/O | `soundfile` for files, `sounddevice` for mic | `sounddevice`'s callback API maps cleanly onto `read_frame`. |

### Two deviations from the spec, both deliberate

**Lung band ends at 950 Hz, not 1000 Hz.** 1000 Hz is exactly Nyquist at a 2 kHz sample
rate, so it cannot be a filter edge — `scipy.signal.butter` requires the normalised
cutoff strictly below 1.0 and raises otherwise. The content between 950 and 1000 Hz is
unreachable at this sample rate no matter how the band is written down. `design_bandpass`
rejects any cutoff at or above Nyquist with an explicit error rather than silently
clamping it, so this can never happen by accident.

**`FileSource` resamples internally** when a file is not already at 2 kHz. The
alternative was to refuse the file. `sample_rate` is declared on the `AudioSource` ABC,
so a source that could return 44100 would turn that attribute into something every
caller must *check* rather than rely on — and `app.py` would grow a rate branch, then
`segment.py` would need one for the BPM arithmetic, and the boundary would be gone. The
cost is that a demo file gets a resampling filter the training data never saw, so it is
made visible rather than silent: the resample is logged, and `FileSource.original_rate`
and `.resampled` are kept so the training path can assert it never touched a record.
CinC ships at 2 kHz, so on the training path this is always a no-op.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`sounddevice` additionally needs the PortAudio system library (`libportaudio2` on
Debian/Ubuntu, `brew install portaudio` on macOS). It is imported lazily, so file
playback works without it.

Verify the audio I/O path end to end:

```bash
python tools/check_io.py
```

This synthesises a phonocardiogram-like WAV at a known heart rate, writes it at
2000 Hz, reads it back, asserts the round trip is sample-accurate, and plots it to
`out/check_io.png`. It exists so that a filter or I/O bug surfaces at hour 0 rather
than hour 4, and so the pipeline has a ground-truth signal to test against.

## Tests

No test framework, no new dependency — plain asserts, run directly:

```bash
python tests/test_dsp.py       # 32 checks
python tests/test_sources.py   # 45 checks
python tests/test_segment.py   # 29 checks
python tests/test_classify.py  # 55 checks
```

`dsp.py` is tested only on signals whose correct answer is known in advance — pure
tones, chirps, impulses, and a synthetic PCG with exact S1/S2 positions. Real audio
cannot tell you whether a wrong answer came from the filter or from the recording.

Two of these checks are load-bearing and worth knowing about:

- **Streaming continuity.** Frame-by-frame filtering is asserted bit-identical to
  whole-signal filtering. Filtering each frame with fresh state instead puts a
  discontinuity at every frame boundary, which the envelope detector reads as a beat
  every 1024 samples — a rock-steady, entirely fake heart rate of 117 bpm. The test
  also asserts that the naive version *does* fail, so the check cannot rot into a
  tautology.
- **Source interchangeability.** The same downstream code recovers 71.99 bpm from a
  2 kHz file, a 44.1 kHz file, and an in-memory array. That is the architectural claim
  made executable: if a hardware source ever needs a downstream edit, this test is
  where it should break first.

## Data

Primary dataset: **PhysioNet/CinC Challenge 2016** — 3,126 labelled recordings
(normal/abnormal) across five sub-databases collected by different teams with
different hardware, plus per-record signal-quality annotations and manually corrected
S1/S2 event labels.

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2016/1.0.0/
```

Train on good-quality records; hold out the poor-quality records as a robustness
check. Train across all five sub-databases, never one — they differ in hardware by
design, and that difference is the only cheap proxy available for the domain shift to
a contact mic.

**Do not use the PASCAL dataset.** It is band-limited below 195 Hz, which removes
components this pipeline depends on.

> Liu C, Springer D, et al. *An open access database for the evaluation of heart sound
> algorithms.* Physiol Meas. 2016;37(12):2181–2213.

## Segmentation

`segment.py` implements the deterministic fallback only. CinC 2016 ships manually
corrected S1/S2 labels, which is normally reason to prefer a supervised segmenter — but
that path needs the dataset, and PhysioNet is unreachable from this environment (see
Data, above), so it has not been attempted.

The deterministic rule: systole (S1→S2) is shorter than diastole (S2→S1) at normal
heart rates, so the peak following the longer gap is S1. At a high enough heart rate
that asymmetry collapses on its own — this is real physiology, not a modelling
artefact, since diastole shortens with rate much faster than systole does — and the
segmenter is required to notice and report "beats detected, S1/S2 uncertain" rather
than a confident wrong label. `MIN_ASYMMETRY_RATIO` (1.15) is the gate.

One deviation from the written spec, found by testing against known-answer signals
rather than assumed: the ~200 ms refractory period named in the spec for suppressing
double-counted ringing was measured, on this exact filter/envelope chain, to be about
25x larger than the ringing actually is (~3.5 ms). At that size it was also swallowing
genuine S1–S2 pairs at realistic elevated heart rates — a true pair 190 ms apart was
collapsing into a single detection, corrupting the interval statistics and producing a
badly wrong BPM, not just an imprecise one. It is set to 80 ms instead, still ~20x the
measured ringing. See the comment on `REFRACTORY_S` in `segment.py`.

`HeartSegmenter` (the streaming class) keeps a persistent filter delay-line across
frames but re-runs peak-picking over a trailing buffer on every call rather than doing
true incremental peak detection — a deliberate scope cut, not a silent one: true
streaming peak-picking needs its own state machine to hold a candidate peak until
enough future samples confirm it, which the hour budget did not have room for.
Recomputing over an 8 s buffer once per UI redraw costs nothing measurable at 2 kHz.

## Classification

`classify.py` is a deterministic triage heuristic, not the CNN CLAUDE.md specifies. The
CNN needs the CinC 2016 dataset to train against; PhysioNet remains unreachable from
this environment, so it was never attempted — not stubbed to a fake output, simply not
built. Per CLAUDE.md's own instruction for this exact situation, the deterministic
layer ships alone.

What it actually evaluates is heart rate (outside a typical resting range) and
beat-to-beat rhythm regularity (coefficient of variation of consecutive S1-S1
intervals) from `segment.py`'s output. This is the same cue a community health worker
already uses with two fingers and a watch, timed more precisely — it is **not**
listening to the sound of the heartbeat for murmurs, rubs, or any other morphology, and
must never be described as though it were. `ClassifyResult.confidence` is a heuristic
weight, not a probability and not a measured accuracy figure; nothing has been
evaluated against a held-out set, so nothing may be shown to a user as "N% accurate."

A `review recommended` result also carries `urgency` (`routine` / `prompt` / `urgent`)
and `action`, a plain next step such as "see a clinician within a day or two." This is
routing, not diagnosis: the urgency tier is set purely by how far the measured rate or
rhythm deviates from typical, and `action` always names a generic destination — "a
clinician," "the nearest health facility" — never a specialist type, since the pipeline
has no basis to say which kind of specialist would even be relevant. Two simultaneous
mild findings escalate one tier past either alone, on the reasoning that concurrent
flags are worth a sooner look even when neither looks severe by itself — that is the
only place severity and breadth interact, and it still never asks what the findings
might mean together.

## Running the live demo

```bash
python app.py --file out/check_io_pcm16.wav     # or any WAV
python app.py --mic                              # live microphone
python app.py --serial /dev/ttyUSB0 --file backup.wav   # Mega 2560, file as 'm' fallback
```

Press `m` to switch between file playback and the live input (microphone or serial,
whichever was configured), `q` to quit. `app.py` draws exactly what `segment.py` and
`classify.py` report — it contains no DSP logic of its own.

**This has not been watched running on a real screen.** This session's container has
no display and no audio input device. What exists instead is
`tools/check_app.py` — a headless smoke test (matplotlib's Agg backend) that drives the
real update loop against a synthetic recording and asserts state actually advances
(BPM appears, beats appear, the status text changes), plus a static snapshot rendered
the same way. That is real coverage of the orchestration logic, but a plot that has
never been watched scrolling on a real screen can still surprise you. Run it on your
laptop before it goes in front of a judge.

```bash
python tools/check_app.py   # 86 checks, headless
```

## Live demo (phone microphone, browser)

For a demo where the phone's own mic is the "stethoscope" instead of the desktop app
reading a file or `sounddevice`: `live_server.py` runs the same `segment.py`/`classify.py`
pipeline behind a small aiohttp server, fed by a browser page (`web/live.html`) that
captures the phone's mic via `getUserMedia` and streams it over a WebSocket. The landing
page (`docs/index.html`) has a "Start live check" button that links to `/live` — only
meaningful when this server is what's actually serving the page (dead on the published
static copy of the landing page, by design).

```bash
pip install -r requirements.txt   # now includes aiohttp
python live_server.py             # serves on http://0.0.0.0:8765
```

Phone browsers only grant microphone access on `localhost` or a page served over
**HTTPS** — plain `http://<laptop-ip>:8765` from another device on the WiFi will
silently fail. In a second terminal, tunnel it:

```bash
cloudflared tunnel --url http://localhost:8765     # or: ngrok http 8765
```

Open the printed `https://...` URL on the phone (the QR code either tool prints works
too) and tap "Start live check."

`/live` opens with a role choice — **clinician** (the mic dashboard plus a running,
in-tab session log: timestamp, BPM, triage label, an optional locally-attached file per
listening — nothing persisted anywhere beyond that browser tab) or **patient** (the mic
dashboard alone, unchanged). Both roles also get an optional Bluetooth panel that can
connect a dedicated BLE heart-rate strap (Polar, Wahoo, Garmin chest straps — the
standard Bluetooth SIG Heart Rate Service, `0x180D`). That number is kept strictly
separate from the mic pipeline's triage output — it's a raw pulse rate, not a listening,
and it never produces a normal/review-recommended/too-noisy label on its own. It does
**not** work with AirPods (no heart-rate sensor exists to read) and most smartwatches
don't broadcast this service to arbitrary web pages either; it needs a dedicated strap.
Only available in Chrome on Android or desktop — `navigator.bluetooth` doesn't exist in
Safari/iOS, so the button disables itself with an explanation rather than failing
silently there.

**Demo fallback.** Phone microphones filter out much of the low-frequency range heart
sounds live in, so a live phone-mic reading can plausibly land on "signal too noisy"
mid-demo — the correct behavior, but bad optics live in front of an audience. The start
screen has a second link, "▶ Or preview a sample reading," that runs
`web/sample_normal.wav` (a synthetic, clean 72bpm recording — the same generator
`tests/`/`tools/` use, not a real person's recording) through the exact same
segmenter/classifier at real-time pace over a second route, `/ws-sample`. Every payload
from that route is tagged `"sample": true`; the UI shows "pre-recorded, not a live
listen" for as long as it's playing, and a session a clinician archives from it is
tagged `[SAMPLE]` in the log so it's never mistaken for a real listening.

**Untested against a real phone in this environment** — no browser, microphone, or
network peer exists in this container. `tests/test_live_server.py` drives the real
server over a real WebSocket with synthetic audio upsampled to a realistic browser rate
(48 kHz) and confirms BPM converges, silence honestly reports "signal too noisy," and no
forbidden vocabulary (a diagnosis, a named condition, a specialist) ever appears in a
streamed result — that is real coverage of the server-side pipeline, but the actual
browser round trip (mic permission prompts, per-phone `AudioContext` quirks, tunnel
behavior) has not been watched running. Try it on a real phone before it goes in front
of a judge.

```bash
python tests/test_live_server.py   # 17 checks, real websocket, synthetic audio
```

## Hardware

An ELEGOO Mega 2560 (ATmega2560, no wireless) plus a sensor front end became available
mid-hackathon — CLAUDE.md's original "hardware does not exist yet" is now out of date.
`hardware/pulso_mic/pulso_mic.ino` samples one analog input at exactly 2 kHz on a Timer1
CTC interrupt and streams it to the laptop over USB serial. `SerialMicSource` in
`sources.py` decodes that stream on the host and presents it as an ordinary
`AudioSource` — the point being that this is a real test of the "hardware drops in with
zero downstream changes" design claim, not just a restatement of it.

**What is and isn't verified.** The wire protocol's parser
(`parse_serial_samples`) is unit-tested against synthetic byte streams, including
deliberately corrupted ones (a dropped byte mid-stream, a truncated trailing sample) —
real coverage of the resync logic. What is *not* verified is the firmware or the
physical sensor, because there is no board in this environment. Run
`tools/check_serial_source.py <port>` first — it reads a few seconds of raw ADC data,
reports the range and DC bias, and flags the most likely wiring problems (no signal,
signal pinned at a rail) before you touch `app.py`.

**Sensor: confirmed as a Keyes KY-038/KY-037-style sound sensor module** (electret
microphone capsule + LM393 comparator + onboard amplifier). Wiring is simple — VCC→5V,
GND→GND, AO→A0, DO unused — no bias circuit needed, since the module's AO output is
already amplified and DC-biased. The two-resistor bias circuit and diode clamp in
`pulso_mic.ino`'s wiring comment are kept as a documented fallback for a bare piezo
disc, in case the sensor changes or the KY-038's response turns out too voice-band-
limited (see below), not because the current sensor needs them.

**Open question, not assumed away: does this module have enough bass response for
heart sound?** KY-038-style modules are built for airborne sound — voice, claps — and
their amplifier stage is usually tuned for that range, not necessarily flat down into
the 20–200 Hz band `dsp.py`'s `HEART_BAND` filters for. Whether there's enough usable
low-frequency signal is genuinely unknown until measured. `check_serial_source.py`'s
plot is the way to find out; placing the capsule directly against skin, or inside a
small acoustic coupling cup the way a DIY stethoscope head is built, will matter more
here than it would for a true contact transducer.

```bash
python tools/check_serial_source.py /dev/ttyUSB0   # or COM3 on Windows
```

## Known threat: domain shift

The model would train on clinical stethoscope audio and run on a cheap contact mic
with a completely different frequency response. Mitigations, in priority order:

1. Train across all five sub-databases.
2. Augment aggressively — pink/white noise, random EQ tilt, random gain, clipping,
   time shift.
3. `DegradedFileSource`, an `AudioSource` wrapper applying the same corruptions.
   Validate against it, not against clean files.
4. Report confidence, and refuse to emit a label below threshold.

The deterministic layer (beat detection, envelope, heart rate, quality indicator) is
what carries the demo and works with no model involved. The classifier is an
additional flag on top, not the product.
