import argparse
import json
from pathlib import Path

from datasets import Split, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from araft.utils import CONFIG_DIR, OUTPUT_DIR


def main(args):

    with open(args.config, "r") as jsonfile:
        config = json.load(jsonfile)

    dataset = load_dataset("csv", data_files=args.dataset, split=Split.TRAIN)
    if args.samples:
        dataset = dataset.select(range(args.samples))

    # load base model
    bnb_config = BitsAndBytesConfig(**config["bnb_config"])
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # initialize trainer
    lora_config = LoraConfig(**config["lora_config"])
    training_arguments = TrainingArguments(
        **config["training_args"], output_dir=args.output_dir
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # run training and save results
    trainer.train()
    trainer.save_model()
    trainer.create_model_card()
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train model with SFT")

    # define cli arguments
    parser.add_argument("model", help="model id from the Huggingface hub")
    parser.add_argument("dataset", help="path to dataset in csv format")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="path to store results",
        type=Path,
        default=OUTPUT_DIR / "araft_trained_sft",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="path to config file",
        type=Path,
        default=CONFIG_DIR / "SFT_trainer.json",
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
