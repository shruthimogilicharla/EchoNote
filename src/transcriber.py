"""
transcriber.py

Wraps a Whisper model for on-device speech-to-text. Designed to run
through ONNX Runtime with the QNN execution provider so inference is
offloaded to the Hexagon NPU on Snapdragon-powered devices. Falls back
to the default CPU execution provider on machines without QNN support,
so the pipeline is still runnable during development on non-Snapdragon
hardware.
"""

import numpy as np
import onnxruntime as ort


class Transcriber:
    def __init__(self, model_path: str, use_npu: bool = True):
        """
        Args:
            model_path: path to the exported Whisper ONNX model
                        (see docs/MODEL_SETUP.md for how to obtain this
                        from Qualcomm AI Hub).
            use_npu: attempt to use the QNN execution provider; falls
                     back to CPU automatically if unavailable.
        """
        providers = ["CPUExecutionProvider"]
        if use_npu and "QNNExecutionProvider" in ort.get_available_providers():
            providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
        else:
            print("[Transcriber] QNN provider not available, using CPU")

        self.session = ort.InferenceSession(model_path, providers=providers)

    def transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """
        Runs a single audio chunk through the model and returns the
        decoded text.

        NOTE: this is a structural stub. The exact input/output tensor
        names and post-processing (token decoding) depend on how the
        Whisper model was exported from Qualcomm AI Hub — fill these in
        once the model is downloaded, per docs/MODEL_SETUP.md.
        """
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: audio_chunk})

        # Placeholder decoding logic — replace with the real tokenizer
        # decode step once the model's output format is confirmed.
        text = self._decode(outputs[0])
        return text

    def _decode(self, model_output) -> str:
        raise NotImplementedError(
            "Implement decoding once the Whisper ONNX model's output "
            "format is confirmed (see docs/MODEL_SETUP.md)."
        )
