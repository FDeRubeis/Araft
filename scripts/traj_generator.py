import argparse
import os
from pathlib import Path

from datasets import Split, load_dataset
from langchain.agents import AgentExecutor
from langchain_community.llms import HuggingFaceEndpoint
from langchain_experimental.chat_models import Llama2Chat

from araft.generate_trajectories import generate_trajectories
from araft.utils import (
    CONFIG_DIR,
    OUTPUT_DIR,
    AraftParserStopSequence,
    format_steps_to_sp,
    generate_prompt_template,
    get_wikipedia_tool,
)


def main(args):

    dataset = load_dataset("json", data_files=args.dataset, split=Split.TRAIN)

    # initialize agent executor
    stop_sequence = "]"
    wikipedia = get_wikipedia_tool()
    llm = HuggingFaceEndpoint(
        repo_id=args.model, huggingfacehub_api_token=args.hf_token
    )
    model = Llama2Chat(llm=llm)
    model_with_stop = model.bind(stop=[stop_sequence])
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_steps_to_sp(x["intermediate_steps"]),
        }
        | generate_prompt_template(args.templates_dir)
        | model_with_stop
        | AraftParserStopSequence(stop_sequence=stop_sequence)
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[wikipedia],
        return_intermediate_steps=True,
        verbose=args.agent_verbose,
        max_iterations=args.agent_max_iter,
    )

    generate_trajectories(
        agent_executor,
        dataset,
        args.output_dir,
        f1_threshold=args.f1_threshold,
        start=args.start,
        stop=args.stop,
        traj_qty=args.traj_qty,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate trajectories for a ReAct agent"
    )

    # define cli arguments
    parser.add_argument("model", help="model id from the Huggingface hub")
    parser.add_argument("dataset", help="path to dataset in json format")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="path to store the folder with the collected trajectories",
        type=Path,
        default=OUTPUT_DIR / "trajectories",
    )
    parser.add_argument(
        "--templates-dir",
        help="directory with prompt templates",
        type=Path,
        default=CONFIG_DIR / "prompt_templates",
    )
    parser.add_argument(
        "--start",
        help="dataset row from which the sampling starts (inclusive) (default: %(default)s)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--stop", help="dataset row where the sampling ends (exclusive)", type=int
    )
    parser.add_argument(
        "--traj-qty", help="maximum number of trajectories to collect", type=int
    )
    parser.add_argument(
        "--hf-token",
        help="Huggingface access token. If not specified, the value from env variable HF_TOKEN is taken",
        default=os.environ.get("HF_TOKEN"),
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
