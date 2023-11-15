# Project Name

A brief description of the project.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Instructions on how to install the project.

## Usage

demo soccernet

```bash
TRANSFORMERS_CACHE=/raid/moriy/.cache/huggingface CUDA_VISIBLE_DEVISES=0 python scripts/run_soccernet_demo.py --device_id 0 --config_file_path "projects/video_blip_soccernet/exp003.yml" --weights_path /raid/moriy/model/heron/video_blip_soccernet/exp003/video_blip_soccernet/exp003_final --img_path "data/SoccerNet/raw_images/0000000002/" --gold_caption "[PLAYER] ([TEAM]) fails to find a teammate across the pitch with a long ball." --log_wandb

TRANSFORMERS_CACHE=/raid/moriy/.cache/huggingface CUDA_VISIBLE_DEVISES=0 python scripts/run_soccernet_demo.py --device_id 0 --config_file_path "projects/video_blip_soccernet/exp004.yml" --weights_path /raid/moriy/model/heron/video_blip_soccernet/exp004/video_blip_soccernet/exp004_final --img_path "data/SoccerNet/raw_images/0000000002/" --gold_caption "[PLAYER] ([TEAM]) fails to find a teammate across the pitch with a long ball." --log_wandb

TRANSFORMERS_CACHE=/raid/moriy/.cache/huggingface CUDA_VISIBLE_DEVISES=3 python scripts/run_soccernet_demo.py --device_id 3 --config_file_path "projects/video_blip_soccernet/exp004.yml" --weights_path /raid/moriy/model/heron/video_blip_soccernet/exp004/video_blip_soccernet/exp004_final --img_path "data/SoccerNet/raw_images/         6/" --gold_caption "[PLAYER] ([TEAM]) volleys the ball from just outside the box, but his effort goes narrowly over the bar." --log_wandb

```

