This folder contains the scrips that were used to fine-tune the Araft model. 
- [traj_generator](traj_generator.py): generate trajectories for distillation
- [traj_to_SFT_converter](traj_to_SFT_converter.py): convert trajectories to SFT training data
- [sft_trainer](sft_trainer.py): perform SFT training of the model
- [traj_to_DPO_converter](traj_to_DPO_converter.py): convert trajectories to DPO training data
- [dpo_trainer](dpo_trainer.py): perform DPO training of the model
- [evaluator_hotpot](evaluator_hotpot.py): evaluate model on hotpotQA dataset