# Copyright 2023 Turing Inc. Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any

import deepspeed
import fire
import torch
import yaml
from transformers import TrainingArguments

from heron.datasets.utils import get_dataset
from heron.models.utils import (
    apply_lora_model,
    load_model,
    load_pretrained_weight,
    set_trainable_params,
    unload_and_merge_lora,
)
from heron.models.vision_language_trainer import VisionLanguageTrainer as Trainer

from evaluator import Evaler  # noqa
from loguru import logger
from datetime import datetime

GitLLMForCausalLM = Any

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"log/train_{current_time}.log"

logger.add(log_filename, rotation="500 MB")


def main(config_file: str, local_rank: int = 0):
    with open(config_file, "r") as i_:
        config = yaml.safe_load(i_)
        model_config = config["model_config"]
        training_config = config["training_config"]

    if os.environ.get("WANDB_NAME") is not None:
        training_config["output_dir"] = os.path.join(
            training_config["output_dir"], os.environ["WANDB_NAME"]
        )

    # distributed learning
    deepspeed.init_distributed()

    # configの割り当て
    keys_to_finetune = config["model_config"]["keys_to_finetune"]
    keys_to_freeze = config["model_config"]["keys_to_freeze"]

    # DatasetのLoad
    train_dataset, val_dataset = get_dataset(config)

    # 訓練に関するconfig
    training_args = TrainingArguments(**training_config)

    # load model
    model = load_model(model_config)

    if model_config["use_lora"]:
        model = apply_lora_model(model, model_config)

    # load pretrained weight
    if model_config.get("pretrained_path") is not None:
        logger.info("load pretrained")
        load_pretrained_weight(model, model_config["pretrained_path"])
        logger.info(
            f'Successfully loading pretrained weights from {model_config["pretrained_path"]}'
        )

    # Set trainable params
    trainable_list, untrainable_list = set_trainable_params(
        model, keys_to_finetune, keys_to_freeze, train_lora=model_config["use_lora"]
    )
    if model_config["use_lora"]:
        model = apply_lora_model(model, model_config)

    logger.info(f"trainable_list\n{trainable_list}")
    logger.info(f"untrainable_list\n{untrainable_list}")

    tokenizer = train_dataset.datasets[0].processor.tokenizer
    # spacyの特別なトークンをリストとしてまとめる
    special_tokens_list = ["[PLAYER]", "[COACH]", "[TEAM]", "([TEAM])", "[REFEREE]"]
    # transformersのトークナイザに特別なトークンを追加
    special_tokens_dict = {"additional_special_tokens": special_tokens_list}
    _ = tokenizer.add_special_tokens(special_tokens_dict)
    model.language_model.resize_token_embeddings(len(tokenizer))

    # evaler = Evaler(tokenizer)
    model.summary()

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    with torch.autocast("cuda"):
        trainer.train()

    # Save the finel checkpoint
    if os.environ.get("WANDB_NAME") is not None:
        final_save_path = os.path.join(
            training_config["output_dir"], os.environ["WANDB_NAME"] + "_final"
        )
    else:
        final_save_path = os.path.join(training_config["output_dir"], "final_model")

    if model_config["use_lora"]:
        model = unload_and_merge_lora(model, model_config)
    model.save_pretrained(final_save_path)
    train_dataset.datasets[0].processor.save_pretrained(final_save_path)


if __name__ == "__main__":
    fire.Fire(main)
