import sys

from PIL import Image

import torch
from transformers import AutoProcessor
import numpy as np
import cv2
from loguru import logger
import yaml

try:
    from heron.models.utils import (
        load_pretrained_weight,
        load_model,
    )
except ModuleNotFoundError:
    sys.path.append(".")
    from heron.models.utils import (
        load_pretrained_weight,
        load_model,
    )


logger.add("log/demo.log", rotation="500 MB")

device_id = 1

config_file_path = "projects/video_blip_soccernet/exp001.yml"
with open(config_file_path, "r") as i_:
    config = yaml.safe_load(i_)
    model_config = config["model_config"]
    training_config = config["training_config"]

model_config[
    "pretrained_path"
] = "/raid/moriy/model/heron/video_blip_soccernet/video_blip_soccernet/exp001/video_blip_soccernet/exp001_final"
model_config["temperature"] = 0.0

img_path = "data/SoccerNet/raw_images/         6/frame_0001.jpg"
gold_caption = "[PLAYER] ([TEAM]) volleys the ball from just outside the box, but his effort goes narrowly over the bar."  # https://www.soccer-net.org/

# prepare a pretrained model
model = load_model(model_config)
load_pretrained_weight(model, model_config["pretrained_path"])
model.eval()
model.to(f"cuda:{device_id}")

# prepare a processor
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

# prepare inputs
img = Image.open(img_path).convert("RGB")
img = np.array(img)
if img.shape[2] != 3:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
imgs = [img]
text = ""

# do preprocessing
inputs = processor(
    images=imgs,
    text=text,
    return_tensors="pt",
    max_length=model_config["max_length"],
    padding="max_length",
    truncation=True,
)
inputs = {k: v.to(f"cuda:{device_id}") for k, v in inputs.items()}
inputs["pixel_values"] = inputs["pixel_values"].half()

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
logger.info(processor.tokenizer.batch_decode(out)[0])
logger.info(gold_caption)
