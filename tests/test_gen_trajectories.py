import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import freezegun
from datasets import Split, load_dataset

from araft.generate_trajectories import generate_trajectories

from .data.mock_agent import get_mock_agent
from .utils import DATA_DIR, TEST_TIME, dircmp

freezegun.configure(extend_ignore_list=["transformers"])


@freezegun.freeze_time(TEST_TIME)
class TestGenTrajectories(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_agent = get_mock_agent()
        cls.dataset = load_dataset(
            "json",
            data_files=str(DATA_DIR / "test_dataset.json"),
            split=Split.TRAIN,
        )
        cls.expected_dir = DATA_DIR / "expected_gen_traj"

    def test_high_threshold(self):
        with TemporaryDirectory() as sampling_dir:
            self.assertRaises(
                ValueError,
                generate_trajectories,
                self.mock_agent,
                self.dataset,
                Path(sampling_dir),
                f1_threshold=1.5,
            )

    def test_negative_threshold(self):
        with TemporaryDirectory() as sampling_dir:
            self.assertRaises(
                ValueError,
                generate_trajectories,
                self.mock_agent,
                self.dataset,
                Path(sampling_dir),
                f1_threshold=-0.1,
            )

    def test_full(self):
        with TemporaryDirectory() as sampling_dir:
            generate_trajectories(
                self.mock_agent, self.dataset, Path(sampling_dir), f1_threshold=0.5
            )
            cmp = dircmp(sampling_dir, self.expected_dir / "full")
            self.assertTrue(cmp.is_same(), cmp.msg())

    def test_start_1_stop_3(self):
        with TemporaryDirectory() as sampling_dir:
            generate_trajectories(
                self.mock_agent,
                self.dataset,
                Path(sampling_dir),
                start=1,
                stop=3,
                f1_threshold=0.5,
            )
            cmp = dircmp(sampling_dir, self.expected_dir / "start_1_stop_3")
            self.assertTrue(cmp.is_same(), cmp.msg())

    def test_traj_qty_1(self):
        with TemporaryDirectory() as sampling_dir:
            generate_trajectories(
                self.mock_agent,
                self.dataset,
                Path(sampling_dir),
                traj_qty=1,
                f1_threshold=0.5,
            )
            cmp = dircmp(sampling_dir, self.expected_dir / "traj_qty_1")
            self.assertTrue(cmp.is_same(), cmp.msg())
