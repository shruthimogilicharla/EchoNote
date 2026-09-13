# EchoNote — Offline Meeting Transcription & Summarization Assistant

EchoNote is an on-device meeting assistant built for Snapdragon-powered HP PCs. It listens to a meeting, transcribes the audio in real time, and produces a clean summary with action items — entirely on-device, using the Hexagon NPU for acceleration. No audio, transcript, or summary ever leaves the machine.

## Why this matters

Most meeting-assistant tools today route audio through the cloud. That's a non-starter for legal, healthcare, enterprise, and education settings where confidentiality is non-negotiable, and it also means the tool stops working the moment you lose internet access. EchoNote is built to solve both problems at once by running the full pipeline — speech recognition, summarization, and action-item extraction — locally, taking advantage of the NPU on Snapdragon X-series silicon for speed and battery efficiency.

## How it works

1. **Capture** — audio is recorded locally from the system microphone (`src/audio_capture.py`).
2. **Transcribe** — the audio stream is transcribed using a Whisper model, run through Qualcomm AI Hub's optimized build so inference runs on the Hexagon NPU instead of the CPU/GPU (`src/transcriber.py`).
3. **Summarize** — the transcript is passed to a small quantized LLM (e.g., Llama 3.2 1B/3B, QNN-optimized) which produces a summary and a list of action items (`src/summarizer.py`).
4. **Present** — a lightweight desktop UI shows the live transcript alongside the running summary (`src/app.py`).

## Project structure

```
EchoNote/
├── src/
│   ├── audio_capture.py     # Records/streams microphone audio
│   ├── transcriber.py       # Whisper-based on-device transcription
│   ├── summarizer.py        # LLM-based summarization + action items
│   └── app.py                # Desktop UI entry point
├── models/                   # Local model weights (not committed — see below)
├── docs/                     # Architecture notes and pitch materials
├── tests/                    # Unit tests for each pipeline stage
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd EchoNote
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Model weights are not committed to this repo due to size. Download instructions are in `docs/MODEL_SETUP.md`.

## Status

This is a hackathon submission for the Snapdragon® AI Lab Build & Present Challenge. Current state: [fill in — e.g., "core pipeline functional, UI in progress"].

## Hardware target

Optimized for Snapdragon X Elite / X Plus-powered HP PCs, using the Hexagon NPU via Qualcomm's QNN execution provider for ONNX Runtime.

## License

MIT (or your choice) — see `LICENSE`.
