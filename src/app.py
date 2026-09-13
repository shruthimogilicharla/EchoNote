"""
app.py

Entry point for EchoNote. Wires together audio capture, transcription,
and summarization, and displays a live transcript + summary in a
simple desktop UI.

Run with:  python src/app.py
"""

import sys
import threading

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit
from PyQt6.QtCore import Qt

from audio_capture import AudioCapture
from transcriber import Transcriber
from summarizer import Summarizer, MeetingState

# Paths — update these once models are downloaded (see docs/MODEL_SETUP.md)
WHISPER_MODEL_PATH = "models/whisper-base-qnn.onnx"


class EchoNoteWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EchoNote — Offline Meeting Assistant")
        self.resize(700, 500)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Live Transcript"))
        self.transcript_box = QTextEdit(readOnly=True)
        layout.addWidget(self.transcript_box)

        layout.addWidget(QLabel("Summary & Action Items"))
        self.summary_box = QTextEdit(readOnly=True)
        layout.addWidget(self.summary_box)

        self.setLayout(layout)

    def append_transcript(self, text: str):
        self.transcript_box.append(text)

    def update_summary(self, summary: str, action_items: list):
        items = "\n".join(f"• {item}" for item in action_items)
        self.summary_box.setPlainText(f"{summary}\n\nAction Items:\n{items}")


def pipeline_worker(window: EchoNoteWindow):
    """Runs capture -> transcribe -> summarize in a background thread
    so the UI stays responsive."""
    capture = AudioCapture()
    transcriber = Transcriber(WHISPER_MODEL_PATH)
    # summarizer = Summarizer(model=..., tokenizer=...)  # load your LLM here
    state = MeetingState()

    capture.start()
    for chunk in capture.chunks():
        text = transcriber.transcribe_chunk(chunk)
        if text:
            window.append_transcript(text)
            # state = summarizer.update(state, text)
            # window.update_summary(state.summary, state.action_items)


def main():
    app = QApplication(sys.argv)
    window = EchoNoteWindow()
    window.show()

    worker = threading.Thread(target=pipeline_worker, args=(window,), daemon=True)
    worker.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
