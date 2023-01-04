import os
import torch
from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2",
                                             revision="fp16",
                                             torch_dtype=torch.float16)
pipeline.save_pretrained("./diffusion_model")
os.system('cd diffusion_model && zip -r ../diffusion_model.zip * && cd .. && rm -r diffusion_model')
prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist", torch_dtype=torch.float16)
prompter_model.save_pretrained("./promptist")
os.system('cd promptist && zip -r ../promptist.zip * && cd .. && rm -r promptist')
