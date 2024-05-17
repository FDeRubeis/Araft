import filecmp
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Split, load_dataset
from transformers import AutoTokenizer

from araft.traj_to_DPO import traj_to_DPO

from .utils import DATA_DIR


class TestTrajToDPO(unittest.TestCase):

    def test_traj_to_DPO(self):
        with TemporaryDirectory() as outdir:
            dataset = load_dataset(
                "json",
                data_files=str(DATA_DIR / "test_trajectories.json"),
                split=Split.TRAIN,
            )
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
            outfile = Path(outdir) / "DPOdata.csv"
            traj_to_DPO(dataset, tokenizer, (DATA_DIR / "prompt_templates"), outfile)

            expected_file = DATA_DIR / "expected_dpo.csv"
            self.assertTrue(
                filecmp.cmp(outfile, expected_file, shallow=False),
                f"following files are different:\n{outfile}\n{expected_file}",
            )
