# Model Setup

EchoNote relies on two models, both intended to run through the QNN execution
provider on Snapdragon-powered HP PCs.

## 1. Whisper (speech-to-text)

- Source: Qualcomm AI Hub's Whisper model listing.
- Export target: ONNX, quantized for QNN.
- Steps:
  1. Visit Qualcomm AI Hub and locate the Whisper (base or small) model entry.
  2. Follow the "Export" flow to produce a QNN-optimized ONNX build.
  3. Place the resulting `.onnx` file at `models/whisper-base-qnn.onnx`.
  4. Confirm input/output tensor names match what `src/transcriber.py` expects,
     and update `_decode()` accordingly — the exact decoding step depends on
     how the model was exported.

## 2. Summarization LLM

- Source: Llama 3.2 1B or 3B (or another small open-source model), quantized
  and exported via Qualcomm AI Hub or the QNN toolchain.
- Steps:
  1. Export or download a QNN-compatible build of the chosen model.
  2. Place model files under `models/llm/`.
  3. Update `src/app.py` to load the model and tokenizer, and pass them into
     `Summarizer(...)`.

## Dev-machine fallback

Both `Transcriber` and the LLM loading code fall back to CPU execution when
the QNN provider isn't available, so the pipeline can be developed and
smoke-tested on a non-Snapdragon machine before final validation on the
target hardware.
