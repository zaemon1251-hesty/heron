import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoProcessor
from loguru import logger
import yaml
from datetime import datetime

# Ensure heron.models.utils can be imported
try:
    from heron.models.utils import load_pretrained_weight, load_model
except ModuleNotFoundError:
    sys.path.append(".")
    from heron.models.utils import load_pretrained_weight, load_model


# Logger setup
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"log/demo_{current_time}.log"

logger.add("log/demo.log", rotation="500 MB")


def load_config(config_file_path):
    with open(config_file_path, "r") as i_:
        config = yaml.safe_load(i_)
    return config["model_config"], config["training_config"]


def prepare_model(model_config, device_id):
    model = load_model(model_config)

    # load pretrained weight
    if model_config.get("pretrained_path") is not None:
        print("load pretrained")
        load_pretrained_weight(model, model_config["pretrained_path"])
        print(
            f'Successfully loading pretrained weights from {model_config["pretrained_path"]}'
        )

    model.eval()
    model.to(f"cuda:{device_id}")
    return model


def prepare_inputs(img_path, processor, model_config, device_id):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    if img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    imgs = [img]
    prompt = "This is soccer video frames. Make a caption with emotion and detail information."
    text = f"##Instrcution:{prompt} \n##Caption:"

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

    return inputs


def generate_caption(model, inputs, processor, model_config, generation_kwargs):
    eos_token_id_list = [
        processor.tokenizer.pad_token_id,
        processor.tokenizer.eos_token_id,
    ]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=model_config["max_length"],
            eos_token_id=eos_token_id_list,
            **generation_kwargs,
        )
    return processor.tokenizer.batch_decode(out)[0]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate captions based on given parameters."
    )

    parser.add_argument(
        "--device_id", type=int, required=True, help="Device ID for CUDA execution."
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--img_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--gold_caption", type=str, required=True, help="Expected gold caption."
    )

    return parser.parse_args()


def main(args):
    model_config, training_config = load_config(args.config_file_path)
    model = prepare_model(model_config, args.device_id)

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    inputs = prepare_inputs(args.img_path, processor, model_config, args.device_id)

    generation_kwargs = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 0,
        "repetition_penalty": 1.1,
    }

    generated_caption = generate_caption(
        model, inputs, processor, model_config, generation_kwargs
    )

    logger.info(generated_caption)
    logger.info(args.gold_caption)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
