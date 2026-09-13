"""
Unit tests for the response-parsing logic in Summarizer.
These don't require a loaded model — they test the text parsing directly.
"""

import unittest
from src.summarizer import Summarizer


class TestSummarizerParsing(unittest.TestCase):
    def setUp(self):
        # model/tokenizer not needed for parsing tests
        self.summarizer = Summarizer(model=None, tokenizer=None)

    def test_parses_summary_and_action_items(self):
        response = (
            "Summary:\n"
            "The team discussed the Q3 roadmap and agreed to prioritize "
            "the offline transcription feature.\n\n"
            "Action Items:\n"
            "- Priya to draft the technical spec by Friday\n"
            "- Sam to set up the CI pipeline\n"
        )
        summary, items = self.summarizer._parse_response(response)

        self.assertIn("Q3 roadmap", summary)
        self.assertEqual(len(items), 2)
        self.assertIn("Priya to draft the technical spec by Friday", items)

    def test_handles_missing_action_items(self):
        response = "Summary:\nJust a general discussion, no concrete next steps yet."
        summary, items = self.summarizer._parse_response(response)

        self.assertIn("general discussion", summary)
        self.assertEqual(items, [])


if __name__ == "__main__":
    unittest.main()
