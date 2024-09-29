import argparse
import json
from pathlib import Path

import torch
from datasets import Split, load_dataset
from langchain.agents import AgentExecutor
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda
from peft import PeftConfig, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from araft.evaluate_hotpot import evaluate_agent, filter_levels
from araft.utils import (
    CONFIG_DIR,
    OUTPUT_DIR,
    AraftParser,
    chat_prompt_to_string,
    format_steps_to_sp,
    generate_prompt_template,
    get_wikipedia_tool,
)


def main(args):

    with open(args.bnb_config, "r") as jsonfile:
        bnb_config_values = json.load(jsonfile)

    dataset = load_dataset("json", data_files=args.dataset, split=Split.TRAIN)
    if args.samples:
        dataset = filter_levels(dataset, args.samples)

    # define hf pipeline
    bnb_config = BitsAndBytesConfig(**bnb_config_values)
    peft_config = PeftConfig.from_pretrained(args.model, subfolder=args.subfolder)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, "float16"),
    )
    peft_model = PeftModelForCausalLM.from_pretrained(
        base_model, args.model, subfolder=args.subfolder
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    # token ] has two different encodings depending on
    # whether it's preceeded by a space character or not.
    # The line below takes both encodings.
    stop_tokens = (
        tokenizer.encode("a]", add_special_tokens=False)[1],
        tokenizer.encode(" ]", add_special_tokens=False)[1],
    )
    pipe = pipeline(
        "text-generation",
        model=peft_model,
        tokenizer=tokenizer,
        eos_token_id=stop_tokens,
        return_full_text=False,
    )

    # initialize agent executor
    to_pipe_input = RunnableLambda(
        lambda cpv: chat_prompt_to_string(cpv, tokenizer.apply_chat_template)
    )
    wikipedia = get_wikipedia_tool()
    hf_pipe = HuggingFacePipeline(pipeline=pipe)
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_steps_to_sp(x["intermediate_steps"]),
        }
        | generate_prompt_template(args.templates_dir)
        | to_pipe_input
        | hf_pipe
        | AraftParser()
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[wikipedia],
        return_intermediate_steps=True,
        verbose=args.agent_verbose,
        max_iterations=args.agent_max_iter,
    )

    evaluate_agent(agent_executor, dataset, args.output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate model on hotpotQA datasets")

    # define cli arguments
    parser.add_argument("model", help="model (peft adapter) id or path")
    parser.add_argument("dataset", help="path to dataset in json format")
    parser.add_argument(
        "--subfolder", help="subfolder where the peft adapter is stored"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="path to store results",
        type=Path,
        default=OUTPUT_DIR / "evaluation",
    )
    parser.add_argument(
        "--templates-dir",
        help="directory with prompt templates",
        type=Path,
        default=CONFIG_DIR / "prompt_templates",
    )
    parser.add_argument(
        "--bnb-config",
        help="path to bnb config file",
        type=Path,
        default=CONFIG_DIR / "evaluator_hotpot_bnb.json",
    )
    parser.add_argument(
        "--samples",
        help="number of easy, medium and hard samples to to take from the dataset",
        type=int,
        nargs=3,
        metavar=("EASY", "MEDIUM", "HARD"),
    )
    parser.add_argument(
        "--agent-verbose", help="make agent verbose", action="store_true"
    )
    parser.add_argument(
        "--agent-max-iter",
        help="max iterations before agent will go to the next question (default: %(default)s)",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--f1-threshold",
        help="f1 threshold to accept answers (default: %(default)s)",
        type=float,
        default=1.0,
    )

    main(parser.parse_args())
