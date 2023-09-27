import sys

sys.path.append(".")

from heron.models.video_blip import VideoBlipForConditionalGeneration
from heron.models.utils import (
    load_pretrained_weight,
)

import requests
from PIL import Image

import torch
from transformers import AutoProcessor

device_id = 0

model_config = {
    "pretrained_path": "/raid/moriy/model/heron/video_blip_soccernet/video_blip_soccernet/exp001/video_blip_soccernet/exp001_final",
    "language_model_name": "stabilityai/japanese-stablelm-base-alpha-7b",
    "num_frames": 1,
    "max_length": 256,
    "temperature": 0.0,
}

img_path = "data/SoccerNet/raw_images/         6/frame_0001.jpg"
gold_caption = "[PLAYER] ([TEAM]) volleys the ball from just outside the box, but his effort goes narrowly over the bar."  # https://www.soccer-net.org/

# prepare a pretrained model
model = VideoBlipForConditionalGeneration.create(
    model_config["language_model_name"],
    model_config["num_frames"],
    torch_dtype=torch.float16,
)
load_pretrained_weight(model, model_config["pretrained_path"])
model.eval()
model.to(f"cuda:{device_id}")

# prepare a processor
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

# prepare inputs
image = Image.open(img_path)

text = ""

# do preprocessing
inputs = processor(
    images=image,
    text=text,
    return_tensors="pt",
    max_length=model_config["max_length"],
    padding="max_length",
    truncation=True,
)
inputs = {k: v.to(f"cuda:{device_id}") for k, v in inputs.items()}

# set eos token
eos_token_id_list = [
    processor.tokenizer.pad_token_id,
    processor.tokenizer.eos_token_id,
]

# do inference
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_length=model_config["max_length"],
        do_sample=False,
        temperature=model_config["temperature"],
        eos_token_id=eos_token_id_list,
    )

# print result
print(processor.tokenizer.batch_decode(out)[0])
print(gold_caption)
