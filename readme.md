## Installation

1) Make sure you have Nvidia driver and Cuda>=11.7 installed.
2) pip install -r requirements.txt
3) python download_models.py
4) torch-model-archiver --model-name stable-diffusion --version 1.0 --handler handler.py --extra-files "diffusion_model.zip, promptist.zip"
5) mkdir model_store && mv stable-diffusion.mar model_store/stable-diffusion.mar
6) Install Java: sudo apt-get install -y openjdk-17-jdk
---
## Startup
- To start: torchserve --start --ts-config config.properties --models all --model-store model_store 
- To stop: torchserve --stop
- Working example: query.py
---
## Request fields:
- "prompt": a text for generation.
- "num_iterations": number of iterations to perform (tradeoff between speed and quality)
- "guidance scale": a classifier-free guidance scale (tradeoff between diversity and quality)
- "negative prompt": a description of what should not be present on an image.
- "eta": eta for DDIM sampling. 0 - faster. 1 - original implementation via Langevin dynamics. Note: larger eta implies more iterations.
- "num_images_per_prompt": number of images to generate.
- "modify_prompt": use Promptist prompt modifier (which probably will result in better image quality).