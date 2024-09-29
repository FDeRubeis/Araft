import re
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from .utils import chat_prompt_to_string, generate_prompt_template


def traj_to_DPO(dataset : Dataset, 
                 tokenizer : AutoTokenizer, 
                 templates_dir : Path,
                 outfile : Path
                 ):
  
    obs_regexp = r'(Observation:.*?\n)'
    prompt_template = generate_prompt_template(templates_dir)

    with open(outfile, 'x', encoding="utf-8") as fd:
        print("prompt,chosen,rejected", file=fd)

        for row in dataset:
            question, traj = row["question"], row["trajectory"]
            scratchpad = ""
            rejected = "Sure! Here are the steps for answering the question:"
            while traj:
                # split the trajectory into:
                # - the first thought/action cycle, 
                # - the observation that follows it
                # - the remaining messages 
                sections = re.split(obs_regexp, traj, 1)
                
                # write entry for current action
                tho_act = sections[0]
                chat_prompt = prompt_template.format_prompt(input=question, agent_scratchpad=scratchpad)
                prompt_string = chat_prompt_to_string(chat_prompt, tokenizer.apply_chat_template)
                data_entry=(
                           f"\"{prompt_string.replace('"','""')}\"," # escape " for csv
                           f"\"{tho_act.replace('"','""')}{tokenizer.eos_token}\","
                           f"\"{rejected.replace('"','""')}{tokenizer.eos_token}\""
                )
                print(data_entry, file=fd)

                # update variables for next loop
                sections += [''] * (3 - len(sections))
                obs, traj = sections[1:3]
                scratchpad += f"{tho_act}{obs}"
                rejected=tho_act
