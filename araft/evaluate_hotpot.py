from __future__ import annotations

from pathlib import Path

from datasets import Dataset, concatenate_datasets
from langchain.agents import AgentExecutor

from araft.utils import answer_question


def filter_levels(dataset: Dataset, levels: list[int]) -> Dataset:

    easy_dataset = dataset.filter(lambda x: x["level"] == "easy").select(
        range(levels[0])
    )
    medium_dataset = dataset.filter(lambda x: x["level"] == "medium").select(
        range(levels[1])
    )
    hard_dataset = dataset.filter(lambda x: x["level"] == "hard").select(
        range(levels[2])
    )

    return concatenate_datasets([easy_dataset, medium_dataset, hard_dataset])


def evaluate_agent(agent: AgentExecutor, dataset: Dataset, report_dir: Path):

    result = {
        "easy": [0, 0],  # f1_score, total
        "medium": [0, 0],  # f1_score, total
        "hard": [0, 0],  # f1_score, total
    }

    # answer questions
    with open(report_dir / "logs.txt", "w") as log_fd:
        for i, data_row in enumerate(dataset):
            level = data_row["level"]
            result[level][1] += 1

            _, f1_score = answer_question(agent, data_row, i, log_fd)
            if not f1_score:
                continue

            result[level][0] += f1_score

    # write report
    with open(report_dir / "report.txt", "w") as log_fd:
        total_f1 = result["easy"][0] + result["medium"][0] + result["hard"][0]
        total = result["easy"][1] + result["medium"][1] + result["hard"][1]
        report_string = (
            f"EASY\nf1_score: {result['easy'][0]:.{2}f}, total: {result['easy'][1]}\n"
            f"MEDIUM\nf1_score: {result['medium'][0]:.{2}f}, total: {result['medium'][1]}\n"
            f"HARD\nf1_score: {result['hard'][0]:.{2}f}, total: {result['hard'][1]}\n"
            f"ALL\nf1_score: {total_f1:.{2}f}, total: {total}\n"
        )
        print(report_string, file=log_fd)
