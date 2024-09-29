from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from langchain.agents import AgentExecutor

from .utils import answer_question, format_steps_to_sp


def generate_trajectories(
    agent_exec: AgentExecutor,
    dataset: Dataset,
    sampling_dir: Path,
    start: int = 0,
    stop: int = None,
    traj_qty: int = None,
    f1_threshold: float = 1.0,
):
    """
    runs agent on the dataset to collect trajectories

    Args:
        sampling_dir: directory where the results are stored.
        start: dataset row from which the sampling starts (inclusive)
        stop: dataset row where the sampling ends (exclusive)
        traj_qty: the maximum number of trajectories to collect. After
        traj_qty trajectories have been collected, the function will return
    """

    if f1_threshold <= 0 or f1_threshold > 1.0:
        raise ValueError(
            f"f1_threshold must be strictly positive and not greater than 1.0. Given threshold: {f1_threshold}"
        )

    if stop is None:
        stop = dataset.num_rows
    if traj_qty is None:
        traj_qty = dataset.num_rows

    with open(sampling_dir / "trajectories.json", "w") as traj_fd:
        with open(sampling_dir / "logs.txt", "w") as log_fd:
            traj_count = 0
            i = start - 1

            for i in range(start, stop):
                if traj_count == traj_qty:
                    i -= 1
                    break

                # process question
                response, f1_score = answer_question(agent_exec, dataset[i], i, log_fd)
                if (not f1_score) or (f1_score < f1_threshold):
                    continue

                # write trajectory
                traj_count += 1
                sample = {
                    "index": i,
                    "id": dataset[i]["_id"],
                    "question": dataset[i]["question"],
                    "label": dataset[i]["answer"],
                    "prediction": response["output"],
                    "f1_score": f1_score,
                    "trajectory": format_steps_to_sp(response["intermediate_steps"])
                    + response["final_step"]
                    + "\n",
                }
                json.dump(sample, traj_fd)
                print("", file=traj_fd, flush=True)  # add newline and flush

            # write summary
            i += 1
            summary_str = f"Start index (inclusive): {start}. Stop index (exclusive): {i}. Successful trajectories count: {traj_count}."
            with open(sampling_dir / "summary.txt", "w") as sum_fd:
                print(summary_str, file=sum_fd)

            print(summary_str)
            return
