# Self-Correcting Human Motion Synthesis with Video Analysis

**For trajectory_guidance, please check my repository
[trajectory_guidance](https://github.com/HuangZiheng-o-O/trajectory_guidance_pipeline_and_llm_enhanced_human_motion_generation)**

This project tackles the challenge of generating detailed and context-rich 3D human motions from textual descriptions beyond standard training data. The framework integrates a multi-agent system—powered by large language models and a vision-language module—to segment, synthesize, and refine motion outputs in an iterative loop. By employing a mask-transformer architecture with body part-specific encoders and codebooks, we achieve granular control over both short and extended motion sequences. After initial generation, an automated review process uses video-based captioning to identify discrepancies and generate corrective instructions, allowing each body region to be accurately adjusted. Experimental results on the HumanML3D benchmark demonstrate that this approach not only attains competitive performance against recent methods but excels in handling long-form prompts and multi-step motion compositions. Comprehensive user studies further indicate significant improvements in realism and fidelity for complex scenarios.

A person does Bruce Lee's classic kicks, and runs forward with right arm extending forward, and trying to avoid sphere obstacles in his way.

https://github.com/user-attachments/assets/10640905-bd8e-4d5b-b001-95ed92313004

A woman picks up speed from a walk to a run, holding the T-pose.

https://github.com/user-attachments/assets/247d33cb-9aad-49d8-a102-3b15b0a8117e

A person sits on the floor with hands resting on their knees, then reaches forward with their right arm trying to grab something.

https://github.com/user-attachments/assets/01a8b24b-6203-4485-8d39-2580cb61856a

An angry midfielder performs a slide tackle on another player.

https://github.com/user-attachments/assets/b6032f26-95f4-4459-b38e-7b7857b0214a



![Model](https://github.com/user-attachments/assets/62b2f604-4c53-4d04-8a66-247082ae4746)


An illustrative example of the workflow：
![framework](https://github.com/user-attachments/assets/d68e084f-d152-4659-9210-c5dcb7ea43f0)


![example](https://github.com/user-attachments/assets/b7f98307-b35a-44cf-b29c-6bff7aace599)

## Acknowlegements

Sincerely thank the open-sourcing of these works where the code is based on: 
[momask-codes](https://github.com/EricGuo5513/momask-codes/), [deep-motion-editing](https://github.com/DeepMotionEditing/deep-motion-editing), [Muse](https://github.com/lucidrains/muse-maskgit-pytorch), [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MDM](https://github.com/GuyTevet/motion-diffusion-model/tree/main) and [MLD](https://github.com/ChenFengYe/motion-latent-diffusion/tree/main)  

