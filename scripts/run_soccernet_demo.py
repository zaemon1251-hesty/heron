import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer
from loguru import logger
import yaml
from datetime import datetime
import os
from itertools import product

# Ensure heron.models.utils can be imported
try:
    from heron.models.utils import load_pretrained_weight, load_model
except ModuleNotFoundError:
    sys.path.append(".")
    from heron.models.utils import load_pretrained_weight, load_model


# Logger setup
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"log/demo_{current_time}.log"

logger.add(log_filename, rotation="500 MB")


def load_config(config_file_path):
    with open(config_file_path, "r") as i_:
        config = yaml.safe_load(i_)
    return config["model_config"], config["training_config"]


def prepare_model(model_config, tokenizer, device_id):
    model = load_model(model_config)

    # spacyの特別なトークンをリストとしてまとめる
    special_tokens_list = ["[PLAYER]", "[COACH]", "[TEAM]", "([TEAM])", "[REFEREE]"]
    # transformersのトークナイザに特別なトークンを追加
    special_tokens_dict = {"additional_special_tokens": special_tokens_list}
    _ = tokenizer.add_special_tokens(special_tokens_dict)
    model.language_model.resize_token_embeddings(len(tokenizer))

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


def prepare_inputs(img_dir_path, processor, model_config, device_id):
    # 画像の枚数が16枚になるようにする
    img_path_list = os.listdir(img_dir_path)
    while len(img_path_list) < model_config["num_image_with_embedding"]:
        img_path_list.append(img_path_list[-1])
        if len(img_path_list) == model_config["num_image_with_embedding"]:
            break
        img_path_list.insert(0, img_path_list[0])

    imgs = []
    for img_filename in img_path_list:
        exact_img_path = os.path.join(img_dir_path, img_filename)
        img = Image.open(exact_img_path).convert("RGB")
        img = np.array(img)
        if img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        imgs.append(img)

    print(f"Number of images: {len(imgs)}")
    logger.info(f"Number of images: {len(imgs)}")
    logger.info(f"Image shape: {imgs[0].shape}")

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
    inputs["pixel_values"] = inputs["pixel_values"].half().unsqueeze(0)

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
        "--weights_path", type=str, required=True, help="Expected gold caption."
    )
    parser.add_argument(
        "--gold_caption", type=str, required=True, help="Expected gold caption."
    )

    return parser.parse_args()


def main(args):
    model_config, _ = load_config(args.config_file_path)
    logger.info(model_config)

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    if args.weights_path is not None:
        processor.tokenizer = AutoTokenizer.from_pretrained(args.weights_path)

    model = prepare_model(model_config, processor.tokenizer, args.device_id)

    inputs = prepare_inputs(args.img_path, processor, model_config, args.device_id)
    logger.info(
        {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    )

    temperature_params = [0.01, 0.3, 0.7, 1.0]
    top_p_params = [0.8, 0.9, 0.95, 1.0]
    repetition_penalty_params = [0.9, 1.0, 1.1, 1.2]

    generation_kwargs = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 0,
        "repetition_penalty": 1.1,
    }
    generated_captions = {}

    for temperature, top_p, repetition_penalty in product(
        temperature_params, top_p_params, repetition_penalty_params
    ):
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
        generation_kwargs["repetition_penalty"] = repetition_penalty

        generated_caption = generate_caption(
            model, inputs, processor, model_config, generation_kwargs
        )
        generated_captions[
            f"temperature: {temperature}, top_p: {top_p}, repetition_penalty: {repetition_penalty}"
        ] = generated_caption

    logger.info(args.gold_caption)
    logger.info(generated_captions)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
