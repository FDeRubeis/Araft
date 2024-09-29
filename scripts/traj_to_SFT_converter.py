import argparse
from pathlib import Path

from datasets import Split, load_dataset
from transformers import AutoTokenizer

from araft.traj_to_SFT import traj_to_SFT
from araft.utils import CONFIG_DIR, OUTPUT_DIR


def main(args):

    if args.outfile.suffix != ".csv":
        raise ValueError("Output file must have csv extension")

    dataset = load_dataset("json", data_files=args.dataset, split=Split.TRAIN)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    traj_to_SFT(dataset, tokenizer, args.templates_dir, args.outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Converts trajectories to SFT training data"
    )

    # define cli arguments
    parser.add_argument("model", help="model id for which you want to format the data")
    parser.add_argument("dataset", help="path to trajectories dataset in json format")
    parser.add_argument(
        "-o",
        "--outfile",
        help="File path to output the SFT training data. Must have csv extension",
        type=Path,
        default=OUTPUT_DIR / "SFTdata.csv",
    )
    parser.add_argument(
        "--templates-dir",
        help="directory with prompt templates",
        type=Path,
        default=CONFIG_DIR / "prompt_templates",
    )

    main(parser.parse_args())
