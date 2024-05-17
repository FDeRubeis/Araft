from __future__ import annotations

import re
import string
import time
import warnings
from collections import Counter
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Callable, Union

from bs4 import GuessedAtParserWarning
from langchain.agents import AgentExecutor
from langchain.agents.agent import AgentOutputParser
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    OutputParserException,
    SystemMessage,
)
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompt_values import ChatPromptValue

WIKIPEDIA_ACTION_STRING = "Wikipedia"
ANSWER_ACTION_STRING = "Answer"
WIKIPEDIA_TOOL_NAME = "wikipedia"
ARAFT_ROOT_DIR = Path(__file__).parent.parent
CONFIG_DIR = ARAFT_ROOT_DIR / "config"
OUTPUT_DIR = ARAFT_ROOT_DIR / "output"


def generate_prompt_template(prompt_templates_dir: Path) -> ChatPromptTemplate:
    """Build Langchain template from template files"""

    system_file = prompt_templates_dir / "system.txt"
    with open(system_file, mode="r") as f:
        system_template = f.read()

    human_file = prompt_templates_dir / "human.txt"
    with open(human_file, mode="r") as f:
        human_template = f.read()

    messages = [
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ]
    return ChatPromptTemplate.from_messages(messages)


class AraftParser(AgentOutputParser):
    """
    Parse the output from the LLM to identify the next action.
    The action can be either 'Wikipedia' or 'Answer'. Format expected:
    Action: [action: ACTION_NAME, action_input: ACTION_INPUT]
    """

    pattern = re.compile(
        rf"Action: \[action: ({WIKIPEDIA_ACTION_STRING}|{ANSWER_ACTION_STRING}), action_input: (.*)\]"
    )

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:

        found = self.pattern.search(text)
        if not found:
            raise OutputParserException(f"no action found: {text}")
        if len(found.groups()) != 2:
            raise OutputParserException(f"no action_input found: {text}")

        action, action_input = found.groups()

        if action == WIKIPEDIA_ACTION_STRING:
            return AgentAction(WIKIPEDIA_TOOL_NAME, action_input, text)

        return AgentFinish({"output": action_input, "final_step": text.lstrip()}, text)

    @property
    def _type(self) -> str:
        return "araft-parser"


class AraftParserStopSequence(AgentOutputParser):
    """Add stop sequence back to output, then call the AraftParser"""

    stop_sequence: str

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:

        text += self.stop_sequence
        return AraftParser().parse(text)

    @property
    def _type(self) -> str:
        return "araft-parser-stop-sequence"


def get_wikipedia_tool() -> WikipediaQueryRun:
    warnings.filterwarnings(
        "ignore",
        category=GuessedAtParserWarning,
        message="No parser was explicitly specified",
    )  # ignore known unfixable warning
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


def format_steps_to_sp(
    intermediate_steps: list[tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
) -> str:
    """Construct the scratchpad containing the current trajectory"""

    scratchpad = ""
    for action, observation in intermediate_steps:
        scratchpad += action.log.lstrip() + "\n" + observation_prefix
        scratchpad += _format_obs_to_sp(observation)

    return scratchpad


def answer_question(
    agent: AgentExecutor,
    data_row: dict[Any],
    index: int,  # row index of the data_row in the dataset
    log_fd: TextIOWrapper,
) -> tuple[dict[Any], float]:

    time_string = datetime.now().strftime("%Y-%m-%d %H:%M")
    question = data_row["question"]
    label = data_row["answer"]
    log_string = f"[{time_string}][id: {data_row['_id']}][ordinal: {str(index)}]: "
    max_iter_error_string = "Agent stopped due to iteration limit or time limit."

    # answer question
    print(f"processing question {index}")
    try:
        response = agent.invoke({"input": question})
    except Exception as error:
        print(f"{log_string}error answering: {repr(error)}", file=log_fd, flush=True)
        return None, None

    # parse response
    if response["output"] == max_iter_error_string:
        print(f"{log_string}max iterations exceeded", file=log_fd, flush=True)
        return None, None

    f1_score = compute_f1_score(label, response["output"])[0]
    print(
        f"{log_string}Agent answer: {response['output']}. Label: {label}. f1_score: {f1_score:.{2}f}",
        file=log_fd,
        flush=True,
    )
    return response, f1_score


def chat_prompt_to_string(
    cpv: ChatPromptValue, apply_chat_template: Callable[[list[Any]], str]
) -> str:
    """
    Convert a Langchain chat prompt into an input string for the
    Huggingface model
    """

    # placholder to prevent trailing newlines from being stripped
    # by tokenizer
    ph = "[NEWLINE]"

    # perform conversion
    lc_to_hf_role = {"system": "system", "human": "user"}
    hf_messages = [
        {
            "role": lc_to_hf_role[lc_msg.type],
            "content": re.sub(r"(\n+)$", ph, lc_msg.content),
        }
        for lc_msg in cpv.messages
    ]
    formatted_text = apply_chat_template(hf_messages, tokenize=False)
    return formatted_text.replace(ph, "\n")


# define f1_score function. Shamelessly stolen from hotpotQA evaluation utilities:
# https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py#L8
def compute_f1_score(prediction, ground_truth):
    normalized_prediction = _normalize_answer(prediction)
    normalized_ground_truth = _normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def compute_em_score(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _format_obs_to_sp(observation: str) -> str:

    # only take the first returned page
    scracthpad = observation.split("\n\nPage: ", 1)[0]

    # only take the page's content
    scracthpad = scracthpad.replace("\n", "")
    scracthpad = scracthpad.split("Summary: ", 1)[1] + "\n"

    return scracthpad


def _normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
