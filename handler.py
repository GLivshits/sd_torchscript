import logging
import zipfile
from abc import ABC
import diffusers
import numpy as np
import torch
import base64
import os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logger.info("Diffusers version %s", diffusers.__version__)


def modifty_text(prompter_model, prompter_tokenizer, plain_text):
    input_ids = prompter_tokenizer(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids
    eos_id = prompter_tokenizer.eos_token_id
    outputs = prompter_model.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = prompter_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res



class DiffusersHandler(BaseHandler, ABC):
    """
    Diffusers handler class for text to image generation.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the Stable Diffusion model is loaded and
        initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device( "cuda:" + str(properties.get("gpu_id"))
                                    if torch.cuda.is_available() and properties.get("gpu_id") is not None
                                    else "cpu")
        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        diffusion_path = os.path.join(model_dir, 'diffusion_model')
        promptist_path = os.path.join(model_dir, 'promptist')
        with zipfile.ZipFile(diffusion_path + '.zip', "r") as zip_ref:
            zip_ref.extractall(diffusion_path)
        with zipfile.ZipFile(promptist_path + '.zip', "r") as zip_ref:
            zip_ref.extractall(promptist_path)
        self.diffusion_model = StableDiffusionPipeline.from_pretrained(diffusion_path)
        self.diffusion_model.to(self.device)
        logger.info("Diffusion model from path %s loaded successfully", model_dir)

        self.prompter_model = AutoModelForCausalLM.from_pretrained(promptist_path)
        self.prompter_model.to(self.device)
        self.prompter_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.prompter_tokenizer.pad_token = self.prompter_tokenizer.eos_token
        self.prompter_tokenizer.padding_side = "left"

        logger.info("Promptist model from path %s loaded successfully", model_dir)

        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, of the user's prompt.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of prompts.
        """
        data = requests[0]
        prompt = str(data.get('prompt'))
        negative_prompt = str(data.get('negative_prompt', ''))
        cfg_scale = float(data.get('cfg_scale', 7.5))
        ddim_eta = float(data.get('ddim_eta', 0.0))
        num_iterations = int(data.get('num_iterations', 50))
        modify_prompt = data.get('modify_prompt', False)
        num_images_per_prompt = int(data.get('num_images_per_prompt', 1))

        metadata = {'prompt': prompt,
                    'num_inference_steps': num_iterations,
                    'guidance_scale': cfg_scale,
                    'negative_prompt': negative_prompt,
                    'eta': ddim_eta,
                    'modify_prompt': modify_prompt,
                    'num_images_per_prompt': num_images_per_prompt}
        return metadata

    def inference(self, inputs):
        """Generates the image relevant to the received text.
        Args:
            input_batch (list): List of Text from the pre-process function is passed here
        Returns:
            list : It returns a list of the generate images for the input text
        """
        # Handling inference for sequence_classification.
        outs = []
        total_number_of_images = int(inputs.pop('num_images_per_prompt'))
        if inputs['modify_prompt']:
            inputs['prompt'] = modifty_text(self.prompter_model, self.prompter_tokenizer, inputs['prompt'])
        inputs.pop('modify_prompt')
        bs = 3
        n = 0
        while n < total_number_of_images:
            cur_bs = min(total_number_of_images-n, bs)
            n += cur_bs
            inferences = self.diffusion_model(**inputs, num_images_per_prompt=cur_bs).images
            outs.extend([np.array(item.convert('RGB')) for item in inferences])
        outs = np.stack(outs, axis=0)
        return outs

    def postprocess(self, inference_output):
        out_bytes = base64.b64encode(inference_output.tobytes()).decode('ascii')
        return [out_bytes]