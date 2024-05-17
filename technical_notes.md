# Technical Notes

This page explores the details about the technical implementation and the reasoning behind the decisions I made throughout the project's development. Additionally, it contains an overview of the project's results and some qualitative observations about the final model's capabilities. I hope that my experience can be helpful for people who want to extend my project or are working on something similar.

For an introduction to the Araft project itself, please refer to the [README.md](./README.md) page.

## Overview

In order to build the agent, I performed multiple steps, all of which can be reproduced using the scripts in this directory.

DISTILLATION  
At first I used a 10 times bigger model, Llama2-70B, to generate a dataset of correct trajectories. The model was prompted to answer questions from the HotpotQA set using the ReAct pattern and, if the final answer was correct, the trajectory was taken as valid and stored. For further details about this step, please see script [traj_generator.py](scripts/traj_generator.py)

SFT TRAINING  
I used the generated trajectories to train a smaller model, Llama2-7B, using Supervised Fine-Tuning (SFT) as suggested by Chen et al. (2023) [[1]](#1). This "taught" the model how to perform a correct ReAct trajectory using Wikipedia. I called the resulting model `araft_trained_sft` and it is available on [Huggingface](https://huggingface.co/FDeRubeis/araft_trained_sft). araft_trained_sft achieves a 16% performance on the HotpotQA dataset. The script I used to convert the trajectories to SFT data is [traj_to_SFT_converter.py](scripts/traj_to_SFT_converter.py) and the one to actually train the model is [sft_trainer.py](scripts/sft_trainer.py).

DPO TRAINING  
araft_trained_sft presented a clear tendency to repeat the same Wikipedia query over and over, often forcing the agent to stop due to reaching the iteration limit. In order to correct this behavior, I used Direct Preference Optimization (DPO) training. In DPO, araft_trained_sft was fine-tuned using the correct steps as desired response, whereas repeating the same query as the previous iteration was presented as undesired response. I called the model resulting from the DPO training `araft_trained_dpo`. araft_trained_dpo achieves a 26% performance. This is an improvement, compared to the 16% of araft_trained_sft. araft_trained_dpo is available on [Huggingface](https://huggingface.co/FDeRubeis/araft_trained_dpo). The script used to convert the trajectories to DPO data is [traj_to_DPO_converter.py](scripts/traj_to_DPO_converter.py) and the one to actually train the model is [dpo_trainer.py](scripts/dpo_trainer.py).

EVALUATION  
I evaluated the models on 300 questions from the HotpotQA dataset (100 easy, 100 medium and 100 hard). The F1 scores, averaged over 5 measurements, are presented as percentages in the table below.:  

|                   | easy | medium | hard | all  |
|-------------------|------|--------|------|------|
| araft_trained_sft | 16.2 | 13.0   | 18.3 | 15.8 |
| araft_trained_dpo | 25.8 | 26.7   | 25.6 | 26.0 |
| Llama2-70B        | 33.0 | 27.8   | 22.6 | 27.8 |

I used the script [evaluator_hotpot.py](scripts/evaluator_hotpot.py) for the evaluation. 

## Insights

The Araft steps and their parameters are a result of various trials and observations I made throughout the project. In this section, I will elaborate on these steps and provide further details about the rationale behind the decisions.

### Distillation 

For the distillation step, I chose Llama2-70B to generate the correct trajectories, since it's the only model of the Llama2 family that is able to do it through prompting only. I provided it with a 3-shot prompt that can be found in the [config folder](config/prompt_templates/).

The original idea was to use Llama2-7B to learn on its own data, taking insipiration from Zelikman et al. (2022) [[2]](#2) and Gulchere et al. (2022) [[3]](#3). Unfortunately the method did not work with the ReAct pattern because it is too complex for Llama2-7B to learn and produce a correct trajectory.

### SFT

The format of the SFT data is based on the indications provided by the [Huggingface's Llama2 release page](https://huggingface.co/blog/llama2#how-to-prompt-llama-2). Llama2 uses the following chat format:

---
\<s>[INST]\<\<SYS>>  
<span style="color:blue">{{ system_prompt }}</span>  
\<\</SYS>>

<span style="color:blue">{{ usr_msg }}</span> [/INST] <span style="color:green"> {{ model_answer }} </span>\</s>

---

I used the 3-shot prompt as the system_prompt and the current trajectory as the user_msg. The model_answer was the next step taken from the correct trajectory. This led to generate a data sample for each step of a trajectory. An example of the generated data can be found in the test's [expected output](tests/data/expected_sft.csv).

In order to fit the training procedure on a T4 GPU, I loaded the model in 4-bit quantization. I used Low-Rank Adaptation (LoRA) [[4]](#4) with r=16 and alpha=32. For further details on the parameters used, please see the [Huggingface model](https://huggingface.co/FDeRubeis/araft_trained_sft) or the script's [config file](config/SFT_trainer.json).

I used a dataset of 600 samples, because Qiao et al. (2023) [[5]](#5) show that after 200 samples the performance plateaus. I used a batch size of 1, although with 4 steps of gradient accumulation. According to my experiments, 400 samples are needed before the performance reaches its peak.

### DPO

As I mentioned previously, the model reaches a performance of 16%, but it has a tendency to repeat the same Wikipedia query over and over. Here is an example of this behavior:

---
<span style="color:blue">Question</span>: Peter Hobbs founded the company that is based in what town in Manchester?   
<span style="color:green">Thought</span>: I don't know it. I need to find the town where Peter Hobbs founded his company.  
<span style="color:green">Action</span>: [action: Wikipedia, action_input: Peter Hobbs]     
<span style="color:green">Observation</span>: Peter Hobbs (January 19, 1918 – January 2, 2011) was a French-born American actor, known for roles on Broadway, television and film...  
<span style="color:green">Thought</span>: I don't know it. I need to find the town where Peter Hobbs founded his company.   
<span style="color:green">Action</span>: [action: Wikipedia, action_input: Peter Hobbs]    
<span style="color:green">Observation</span>: Peter Hobbs (January 19, 1918 – January 2, 2011) was a French-born American actor, known for roles on Broadway, television and film...  
...

---
For this reason, I decided to perform a DPO training to discourage this behavior. In this training, repeating the same question was explictily provided as undesired behavior. For example, given the following intermediate trajectory:

---
<span style="color:blue">Question</span>: What city, other than San Jose del Cabo, is at the the southern tip of Baja California Peninsula and is also encompassed in the Los Cabos Municipality?  
<span style="color:blue">Thought</span>: I'm not sure. I think it's Cabo Saint Lucas, but I'm not certain.  
<span style="color:blue">Action</span>: [action: Wikipedia, action_input: Los Cabos Municipality]  
<span style="color:blue">Observation</span>: Los Cabos (Spanish pronunciation: [los ˈkaβos]) is a municipality located ...

---
This output was shown as desired behavior:

---
<span style="color:green">Thought</span>: I'm still not sure. The observation mentions Cabo Saint Lucas, but it also mentions San José del Cabo. I need to find out which one is not part of the Los Cabos Municipality.    
<span style="color:green">Action</span>: [action: Wikipedia, action_input: San José del Cabo]  

---
whereas this output (i.e. the repetition of the previous step) was shown as undersired:

---
<span style="color:red">Thought</span>: I'm not sure. I think it's Cabo Saint Lucas, but I'm not certain.  
<span style="color:red">Action</span>: [action: Wikipedia, action_input: Los Cabos Municipality]

---

The formatted dataset was generated following [Huggingface's documentation](https://huggingface.co/docs/trl/en/dpo_trainer#expected-dataset-format). An example of the generated data can be found in the test's [expected output](tests/data/expected_dpo.csv).

For the training the model was loaded in 4-bit quantization. The LoRA adapter from the SFT training was used as a reference and a new adapter with the same dimensions was produced. For further details on the parameters used, please see the [Huggingface model](https://huggingface.co/FDeRubeis/araft_trained_dpo) or the script's [config file](config/DPO_trainer.json).

Also in this case the model was trained on 600 samples with a batch size of 1 (gradient accumulation over 4 steps). According to my experiments, however, the performance reached its peak after only 100 samples. The checkpoints to test the model at different stages of the fine-tuning can be found on Huggingface.

## Results

The DPO trained model showed a significant performance improvement compared to the SFT trained model, meaning that the DPO training worked. Here is an example of a correct trajectory:

---
<span style="color:blue">Question</span>: Who did Honora Edgeworth decline the hand of, even as she had a romantic engagement with a British Army officer hanged as a spy by the Continental Army?  
<span style="color:green">Thought</span>: I don't know the answer. I need to search for Honora Edgeworth and her romantic engagement.  
<span style="color:green">Action</span>: [action: Wikipedia, action_input: Honora Edgeworth]  
<span style="color:green">Observation</span>: Honora Edgeworth (née Sneyd; 1751 – 1 May 1780) was an eighteenth-century English writer ... and having declined the hand of Thomas Day ...  
<span style="color:green">Thought</span>: The observation reports that Honora Edgeworth declined the hand of Thomas Day.  
<span style="color:green">Action</span>: [action: Answer, action_input: Thomas Day]

---

However, the model still often shows limited reasoning capabilities. For example, in the following trajectory, it shows that it does not understand the question and gives a trivial answer:

---
<span style="color:blue">Question</span>: "Hollywood" was a rock song made famous by which band headed by the Irish musician and singer songwriter Dolores O'Riordan?"  
<span style="color:green">Thought</span>: I don't know it. I need to search the information about Dolores O'Riordan and the song "Hollywood".  
<span style="color:green">Action</span>: [action: Wikipedia, action_input: Dolores O'Riordan]  
<span style="color:green">Observation</span>: Dolores Mary Eileen O'Riordan ( oh-REER-dən; 6 September 1971 – 15 January 2018) was an Irish singer, songwriter, ...   
<span style="color:green">Thought</span>: The song "Hollywood" was made famous by the Irish musician and singer-songwriter Dolores O'Riordan.  
<span style="color:green">Action</span>: [action: Answer, action_input: Dolores O'Riordan]  

---

As an other examples of flawed reasoning, we can see the following question:

---
<span style="color:blue">Question</span>: Shuozhou and Aral, Xinjiang are both located where?  
<span style="color:green">Thought</span>: I don't know it. I need to search Shuozhou and Aral on Wikipedia.  
<span style="color:green">Action</span>: [action: Wikipedia, action_input: Shuozhou and Aral]  
<span style="color:green">Observation</span>: Shuozhou is a prefecture-level city in northern Shanxi province, China, bordering Inner Mongolia...  
<span style="color:green">Thought</span>: The observation reports that Shuozhou is located in northern Shanxi province, China.  
<span style="color:green">Action</span>: [action: Answer, action_input: northern Shanxi province]  

--- 

In the trajectory above, the model seems to understand that we are looking for a "location" in the context of Shouzhou and Aral, and it stops as soon as it finds something that resembles an answer.

Both trajectories show that the reasoning capabilities of the model are still limited. In particular, they show that the model falls easily into the trap of "spurious correlation", i.e. information that looks like the right answer because of its context, but is not.

Also, according to my observations, the model is only able to answer single-hop questions, meaning that it's not able to actually plan a trajectory.

The limited reasoning capabilities of the model could already be observed when it failed to use the ReAct pattern through prompting alone. However, this problem could be overcome by fine-tuning the model. In general, fine-tuning is extremeley helpful when it comes to teach the model to think a certain way. For this reason, I suggest anyone who wants to use the Araft model or scripts to keep in mind the limitations and to prefer fine-tuning to prompting when it comes to instruct the model.

## Conclusion

In the Araft project, I have built a ReAct agent for question-answering that retrieves infromation from Wikipedia. The agent is small, so it can be loaded onto a single T4 (for example, from Google Colab), which I hope can be helpful to any enthusiast/researcher who is looking for an agile way of evaluating a ReAct agent and test new enhancement/ideas onto it. However, it is important to always keep the model's limitations in mind: its cognitive capabilities are inferior to those of bigger models and sometimes fine-tuning might be the only way to instruct the model. 

## References

<a id="1">[1]</a> 
Chen, Baian et al. “FireAct: Toward Language Agent Fine-tuning.” ArXiv abs/2310.05915 (2023)

<a id="2">[2]</a> 
Zelikman, E. et al. “STaR: Bootstrapping Reasoning With Reasoning.” (2022).

<a id="3">[3]</a> 
Gulcehre, Caglar et al. “Reinforced Self-Training (ReST) for Language Modeling.” ArXiv abs/2308.08998 (2023)

<a id="4">[4]</a> 
Hu, J. Edward et al. “LoRA: Low-Rank Adaptation of Large Language Models.” ArXiv abs/2106.09685 (2021)

<a id="5">[5]</a> 
Qiao, Shuofei et al. “AUTOACT: Automatic Agent Learning from Scratch via Self-Planning.” ArXiv abs/2401.05268 (2024)