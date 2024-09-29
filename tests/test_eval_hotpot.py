import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import freezegun
from datasets import Split, load_dataset

from araft.evaluate_hotpot import evaluate_agent, filter_levels

from .data.mock_agent import get_mock_agent
from .utils import DATA_DIR, TEST_TIME, dircmp

freezegun.configure(extend_ignore_list=["transformers"])


@freezegun.freeze_time(TEST_TIME)
class TestEvalHotpot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_agent = get_mock_agent()
        cls.dataset = load_dataset(
            "json", data_files=str(DATA_DIR / "test_dataset.json"), split=Split.TRAIN
        )
        cls.expected_dir = DATA_DIR / "expected_eval_hotpot"

    def test_filter_111(self):
        filtered_dataset = filter_levels(self.dataset, [1, 1, 1])
        self.assertEqual(
            filtered_dataset["_id"],
            [
                "5a875ce15542993e715abf16",
                "5a81a60455429903bc27b990",
                "5a70fb2d5542994082a3e482",
            ],
        )

    def test_filter_101(self):
        filtered_dataset = filter_levels(self.dataset, [1, 0, 1])
        self.assertEqual(
            filtered_dataset["_id"],
            ["5a875ce15542993e715abf16", "5a70fb2d5542994082a3e482"],
        )

    def test_filter_too_many(self):
        self.assertRaises(IndexError, filter_levels, self.dataset, [1, 0, 4])

    def test_evaluate(self):
        with TemporaryDirectory() as report_dir:
            evaluate_agent(self.mock_agent, self.dataset, Path(report_dir))
            cmp = dircmp(report_dir, self.expected_dir)
            self.assertTrue(cmp.is_same(), cmp.msg())
