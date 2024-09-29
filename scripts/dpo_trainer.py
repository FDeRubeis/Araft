import argparse
import json
from pathlib import Path

import torch
from datasets import Split, load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer

from araft.utils import CONFIG_DIR, OUTPUT_DIR


def main(args):

    ref_adapter_name = "reference"
    dpo_adapter_name = "dpo_trained"

    with open(args.config, "r") as jsonfile:
        config = json.load(jsonfile)

    dataset = load_dataset("csv", data_files=args.dataset, split=Split.TRAIN)
    if args.samples:
        dataset = dataset.select(range(args.samples))

    # load model
    bnb_config = BitsAndBytesConfig(**config["bnb_config"])
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, "float16"),
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)

    # load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # load adapters
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        is_trainable=True,
        adapter_name=dpo_adapter_name,
    )
    model.load_adapter(args.adapter, adapter_name=ref_adapter_name)

    # initialize trainer
    training_arguments = TrainingArguments(
        **config["training_args"], output_dir=args.output_dir
    )
    dpo_trainer = DPOTrainer(
        model,
        args=training_arguments,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=0.1,
        model_adapter_name=dpo_adapter_name,
        ref_adapter_name=ref_adapter_name,
    )

    # run training and save results
    dpo_trainer.train()
    dpo_trainer.save_model()
    dpo_trainer.create_model_card()
    if args.push_to_hub:
        dpo_trainer.push_to_hub()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train model with DPO")

    # define cli arguments
    parser.add_argument("base_model", help="base model of the SFT training")
    parser.add_argument("adapter", help="adapter from the SFT training")
    parser.add_argument("dataset", help="path to dataset in csv format")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="path to store results",
        type=Path,
        default=OUTPUT_DIR / "araft_trained_dpo",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="path to config file",
        type=Path,
        default=CONFIG_DIR / "DPO_trainer.json",
    )
    parser.add_argument(
        "--samples",
        help="only take the first %(metavar)s samples from dataset",
        type=int,
        metavar="N",
    )
    parser.add_argument(
        "--push-to-hub",
        help="push to hub the model at the end of the training. Defaults to False",
        action="store_true",
    )

    main(parser.parse_args())
