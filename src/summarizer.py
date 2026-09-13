"""
summarizer.py

Takes an accumulating meeting transcript and produces a running summary
plus a list of action items, using a small quantized LLM (e.g. Llama
3.2 1B/3B) optimized for the Hexagon NPU via QNN.
"""

from dataclasses import dataclass, field


SUMMARY_PROMPT_TEMPLATE = """You are taking notes for a meeting. Given the transcript so far, \
produce:
1. A short summary (3-5 sentences) of what has been discussed.
2. A bullet list of any action items mentioned, with who owns each one if stated.

Transcript so far:
{transcript}

Respond in this format:
Summary:
<summary>

Action Items:
- <item 1>
- <item 2>
"""


@dataclass
class MeetingState:
    transcript: str = ""
    summary: str = ""
    action_items: list = field(default_factory=list)


class Summarizer:
    def __init__(self, model, tokenizer):
        """
        Args:
            model: a loaded, QNN-optimized causal LLM (e.g. via
                   optimum's ONNX Runtime integration).
            tokenizer: the matching tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer

    def update(self, state: MeetingState, new_text: str) -> MeetingState:
        """Append new transcript text and regenerate the summary."""
        state.transcript += " " + new_text
        prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=state.transcript)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        output_ids = self.model.generate(**inputs, max_new_tokens=256)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        state.summary, state.action_items = self._parse_response(response)
        return state

    def _parse_response(self, response: str):
        summary = ""
        action_items = []

        if "Action Items:" in response:
            summary_part, items_part = response.split("Action Items:", 1)
            summary = summary_part.replace("Summary:", "").strip()
            action_items = [
                line.strip("- ").strip()
                for line in items_part.strip().splitlines()
                if line.strip().startswith("-")
            ]
        else:
            summary = response.strip()

        return summary, action_items
