# Pulso

A low-cost digital stethoscope triage tool. A contact microphone captures heart and
lung sounds, streams them to a laptop, and software flags abnormal patterns as a
**triage signal for a community health worker**.

> **Not a medical device. Not for clinical use.**
> Pulso does not diagnose. It outputs one of three triage states —
> *normal*, *review recommended*, or *signal too noisy* — and nothing else.

## Status

Skeleton only. The `AudioSource` boundary is defined; no DSP, segmentation,
classification, or UI yet.

## Architecture

Everything downstream of audio capture is ignorant of where the audio came from.

| Module | Role | State |
|---|---|---|
| `pulso/sources.py` | `AudioSource` ABC → `FileSource`, `MicSource`, later `ESP32Source` | ABC defined |
| `pulso/dsp.py` | Pure functions: filters, envelope. No state, no I/O. | not written |
| `pulso/segment.py` | Stateful: beat detection, S1/S2 labelling, heart rate | not written |
| `pulso/classify.py` | Window in → `{label, confidence}` out | not written |
| `pulso/app.py` | UI and orchestration only. No DSP logic. | not written |

Adding hardware later must not require editing `dsp.py`, `segment.py`,
`classify.py`, or `app.py`.

## Fixed technical decisions

| Decision | Value | Reason |
|---|---|---|
| Internal sample rate | 2000 Hz | CinC 2016 ships at 2 kHz; zero resampling on the training path. Nyquist 1 kHz covers heart (20–200 Hz) and most lung (100–1000 Hz). |
| Frame size | 1024 samples (~0.5 s) | Enough envelope context for beat detection, still feels real-time. |
| Heart bandpass | 25–200 Hz, Butterworth order 4 | Below 25 Hz is handling noise and DC drift; above 200 Hz is not heart sound. |
| Lung bandpass | 100–1000 Hz, Butterworth order 4 | Standard respiratory band. |
| Filter application | `sosfiltfilt` offline, `sosfilt` + persistent `zi` streaming | Order-4 IIR in `ba` form is numerically unstable; use SOS. |
| Envelope | Shannon energy → ~50 ms moving average | Standard for PCG; suppresses low-amplitude noise better than rectification. |
| Audio I/O | `soundfile` for files, `sounddevice` for mic | `sounddevice`'s callback API maps cleanly onto `read_frame`. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Primary dataset: PhysioNet/CinC Challenge 2016 — 3,126 labelled recordings
(normal/abnormal) across five sub-databases collected with different hardware,
with per-record signal-quality annotations and manually-corrected S1/S2 labels.

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2016/1.0.0/
```

Train on good-quality records; hold out the poor-quality ones as a robustness
check. The PASCAL dataset is **not** used — it is band-limited below 195 Hz,
which removes components this pipeline depends on.

Liu C, Springer D, et al. *An open access database for the evaluation of heart
sound algorithms.* Physiol Meas. 2016;37(12):2181–2213.

## Safety constraints

- No diagnosis, no disease name, no treatment suggestion — ever.
- Triage language only: "normal" / "review recommended" / "signal too noisy".
- No accuracy figure is stated unless it was measured on a held-out set.
