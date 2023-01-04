import numpy as np
import requests
from PIL import Image
import base64
import torch
import json

data = {'prompt': 'a photo of an astronaut from the Returnal game riding a horse',
        'num_iterations': 50,
        'guidance_scale': 10,
        'negative_prompt': '',
        'eta': 0.0,
        'num_images_per_prompt': 6,
        'modifty_prompt': True}


response = requests.post('http://127.0.0.1:8080/predictions/stable-diffusion/', data=data)
out_arr = np.frombuffer(base64.b64decode(response.content), dtype=np.uint8).reshape(-1, 768, 768, 3)
for i, item in enumerate(out_arr):
    Image.fromarray(item).save(f'{i+1}.jpg')