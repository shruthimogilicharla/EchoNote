"""
audio_capture.py

Handles microphone input for EchoNote. Streams audio in fixed-length
chunks so the transcription pipeline can process meetings incrementally
rather than waiting for the whole recording to finish.
"""

import queue
import numpy as np
import sounddevice as sd


class AudioCapture:
    """Streams microphone audio in chunks for downstream transcription."""

    def __init__(self, samplerate: int = 16000, chunk_seconds: float = 5.0):
        self.samplerate = samplerate
        self.chunk_samples = int(samplerate * chunk_seconds)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[AudioCapture] stream status: {status}")
        self._queue.put(indata.copy())

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_samples,
            callback=self._callback,
        )
        self._stream.start()
        print("[AudioCapture] recording started")

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            print("[AudioCapture] recording stopped")

    def chunks(self):
        """Yield audio chunks as they arrive. Blocks until stop() is called
        and the queue is drained."""
        while self._stream is not None and self._stream.active:
            try:
                yield self._queue.get(timeout=1.0)
            except queue.Empty:
                continue


if __name__ == "__main__":
    # Quick manual test: record for ~10 seconds and report chunk shapes.
    import time

    cap = AudioCapture()
    cap.start()
    start = time.time()
    for chunk in cap.chunks():
        print(f"Got chunk: {chunk.shape}")
        if time.time() - start > 10:
            break
    cap.stop()
