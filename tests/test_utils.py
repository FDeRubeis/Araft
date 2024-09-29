import unittest
from io import StringIO
from pathlib import Path

import freezegun
from datasets import Split, load_dataset
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from transformers import AutoTokenizer

from araft.utils import (
    ANSWER_ACTION_STRING,
    WIKIPEDIA_ACTION_STRING,
    WIKIPEDIA_TOOL_NAME,
    AraftParser,
    AraftParserStopSequence,
    answer_question,
    chat_prompt_to_string,
    compute_f1_score,
    format_steps_to_sp,
    generate_prompt_template,
)

from .data.mock_agent import AGENT_ANSWER, AGENT_SALZBURG_ERROR, get_mock_agent
from .utils import DATA_DIR, TEST_TIME

freezegun.configure(extend_ignore_list=["transformers"])


class TestUtils(unittest.TestCase):

    def test_format_steps_sp_empty(self):
        thought = format_steps_to_sp([])
        self.assertEqual("", thought)

    def test_format_steps_sp_action(self):
        action_input = "foo"
        action_string = (
            f"Action: [action: {WIKIPEDIA_ACTION_STRING}, action_input: {action_input}]"
        )
        tho_act = f"Thought: bar\n{action_string}"
        first_summary = r"This is a list of airports in Maine."
        first_page = rf"Page: List of airports in Maine\nSummary: {first_summary}"
        second_summary = r"Presque Isle International Airport"
        second_page = (
            rf"Page: Presque Isle International Airport\nSummary: {second_summary}"
        )
        step = [
            AgentAction(
                tool=WIKIPEDIA_TOOL_NAME,
                tool_input=action_input,
                log=f" {tho_act}",
            ),
            f"{first_page}\n\n{second_page}",
        ]
        thoughts = format_steps_to_sp([step])
        self.assertEqual(thoughts, f"{tho_act}\nObservation: {first_summary}\n")

    def test_compute_f1_score(self):
        # empty strings
        self.assertEqual(compute_f1_score("", ""), (0, 0, 0))
        # non-normalized string
        self.assertEqual(compute_f1_score("the FoO.? bAr", "foo bar"), (1.0, 1.0, 1.0))
        # different strings
        self.assertEqual(compute_f1_score("foo", "bar"), (0, 0, 0))
        # same strings
        self.assertEqual(compute_f1_score("foo", "foo"), (1.0, 1.0, 1.0))
        # partially similar strings
        self.assertEqual(compute_f1_score("foo bar", "bar baz"), (0.5, 0.5, 0.5))

    def test_chat_prompt_to_string(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
        templates_dir = DATA_DIR / "prompt_templates"
        prompt_template = generate_prompt_template(templates_dir).format_prompt(
            input="foo", agent_scratchpad="bar"
        )
        prompt_string = chat_prompt_to_string(
            prompt_template, tokenizer.apply_chat_template
        )
        self.assertEqual(
            prompt_string,
            '<s>[INST] <<SYS>>\nAnswer the given question completing the given trajectory. A trajectory is a cycle of Thought/Action/Observation steps.\n\nIn the Thought step you reflect on your internal knowledge and the knowledge obtained so far from the observation to decide on the next action. In the Action step you will perform the action "Answer" if you know the answer, or "Wikipedia" if you need further information. The observation step contains the returned information from the Wikipedia action.\n\n\nQuestion: What cricket batting stroke, which hits the cricket ball over the keeper, was invented by Tillakaratne Dilshan during the ICC World Twenty20 in 2009?\nThought: I don\'t know it. I need to find the batting stroke invented by Tillakaratne Dilshan.\nAction: [action: Wikipedia, action_input: Tillakaratne Dilshan batting stroke]\nObservation: Tillakaratne Dilshan is an aggressive right-hand batsman who invented the scoop, which has come to be known as the Dilscoop, a shot that hits the ball over the keeper. \nThought: the observation reports that the batting stroke invented by Tillakaratne Dilshan was the Dilscoop. So now I can provide the final answer.\nAction: [action: Answer, action_input: the Dilscoop]\n\n\nQuestion: What is the most famous monument in Rome?\nThought: I already know the answer. The most famous monument in Rome is the Colosseum.\nAction: [action: Answer, action_input: the Colosseum]\n\n\nQuestion: What is the birth name of the actor that co-starred with Gilda Radner in the film "Hanky Panky"?\nThought: I don\'t know it. I need to search the movie Hanky Panky on Wikipedia.\nAction: [action: Wikipedia, action_input: Hanky Panky movie]\nObservation: Hanky Panky is a 1982 American comedy thriller Metrocolor film directed by Sidney Poitier, starring Gene Wilder and Gilda Radner. Wilder and Radner met during filming and later married.\nThought: the actor that co-starred with Gilda Radner was Gene Wilder. I need to find his birth name.\nAction: [action: Wikipedia, action_input: Gene Wilder]\nObservation: Gene Wilder (born Jerome Silberman, June 11, 1933 - August 29, 2016) was an American actor, comedian, writer and filmmaker known mainly for his comedic roles, but also for his portrayal of Willy Wonka in Willy Wonka & the Chocolate Factory (1971).\nThought: the observation reports that the birth name of Gene Wilder is Jerome Silberman. So now I can provide the final answer.\nAction: [action: Answer, action_input: Jerome Silberman]\n\nNote that your job is only to provide the Thought and Action steps, your answer needs to be ALWAYS in the following format:\nThought: <YOUR_THOUGHT>\nAction: [action: <ACTION_NAME>, action_input: <ACTION_INPUT>]\n\nDon\'t add anything else! Your answer MUST ALWAYS start with "Thought: " and end with "]" followed by a newline.\n<</SYS>>\n\nQuestion: foo\nbar [/INST]',
        )

    def test_generate_prompt_no_template(self):
        self.assertRaises(FileNotFoundError, generate_prompt_template, Path("foo"))

    def test_generate_prompt_template(self):

        templates_dir = DATA_DIR / "prompt_templates"
        template = generate_prompt_template(templates_dir)

        system_message, human_message = template.messages

        self.assertEqual(type(system_message), SystemMessage)
        self.assertEqual(type(human_message), HumanMessagePromptTemplate)
        self.assertEqual(human_message.input_variables, ["agent_scratchpad", "input"])

    def test_araft_parser_stop_seq(self):
        stop_sequence = "]"
        parser = AraftParserStopSequence(stop_sequence=stop_sequence)
        action_input = "foo"
        action_string = (
            f"Action: [action: {WIKIPEDIA_ACTION_STRING}, action_input: {action_input}"
        )
        parsed_action = parser.parse(action_string)
        self.assertEqual(
            parsed_action,
            AgentAction(
                tool=WIKIPEDIA_TOOL_NAME,
                tool_input=action_input,
                log=f"{action_string}{stop_sequence}",
            ),
        )


class TestAraftParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = AraftParser()

    def test_no_action(self):
        action_string = "foo"
        self.assertRaises(OutputParserException, self.parser.parse, action_string)

    def test_wikipedia_action(self):
        action_input = "foo"
        action_string = (
            f"Action: [action: {WIKIPEDIA_ACTION_STRING}, action_input: {action_input}]"
        )
        parsed_action = self.parser.parse(action_string)
        self.assertEqual(
            parsed_action,
            AgentAction(
                tool=WIKIPEDIA_TOOL_NAME, tool_input=action_input, log=action_string
            ),
        )

    def test_answer_action(self):
        answer = "foo"
        answer_string = (
            f"Action: [action: {ANSWER_ACTION_STRING}, action_input: {answer}]"
        )
        parsed_action = self.parser.parse(answer_string)
        self.assertEqual(
            parsed_action,
            AgentFinish(
                return_values={"output": answer, "final_step": answer_string},
                log=answer_string,
            ),
        )


@freezegun.freeze_time(TEST_TIME)
class TestAnswerQuestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_agent = get_mock_agent()
        cls.dataset = load_dataset(
            "json",
            data_files=str(DATA_DIR / "test_dataset.json"),
            split=Split.TRAIN,
        )

    def test_right_answer(self):
        log_fd = StringIO()
        index = 0
        response = answer_question(self.mock_agent, self.dataset[index], index, log_fd)
        log_fd.seek(0)
        response_log = log_fd.read()

        self.assertEqual(
            response_log,
            "[2024-04-04 00:00][id: 5a81a60455429903bc27b990][ordinal: 0]: Agent answer: Yes. Label: yes. f1_score: 1.00\n",
        )
        self.assertEqual(response, (AGENT_ANSWER[self.dataset[index]["question"]], 1.0))

    def test_almost_right(self):
        log_fd = StringIO()
        index = 1
        response = answer_question(self.mock_agent, self.dataset[index], index, log_fd)
        log_fd.seek(0)
        response_log = log_fd.read()

        self.assertEqual(
            response_log,
            "[2024-04-04 00:00][id: 5a70fb2d5542994082a3e482][ordinal: 1]: Agent answer: Cabo Saint Lucas. Label: Cabo San Lucas. f1_score: 0.67\n",
        )
        self.assertEqual(
            response,
            (AGENT_ANSWER[self.dataset[index]["question"]], 0.6666666666666666),
        )

    def test_iteration_limit(self):
        log_fd = StringIO()
        index = 3
        response = answer_question(self.mock_agent, self.dataset[index], index, log_fd)
        log_fd.seek(0)
        response_log = log_fd.read()

        self.assertEqual(
            response_log,
            "[2024-04-04 00:00][id: 5a875ce15542993e715abf16][ordinal: 3]: max iterations exceeded\n",
        )
        self.assertEqual(response, (None, None))

    def test_value_error(self):
        log_fd = StringIO()
        index = 4
        response = answer_question(self.mock_agent, self.dataset[index], index, log_fd)
        log_fd.seek(0)
        response_log = log_fd.read()

        self.assertEqual(
            response_log,
            f"[2024-04-04 00:00][id: 5a760b725542994ccc91869a][ordinal: 4]: error answering: {repr(AGENT_SALZBURG_ERROR)}\n",
        )
        self.assertEqual(response, (None, None))
