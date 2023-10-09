#!/bin/bash
export TRANSFORMERS_CACHE=/raid/moriy/.cache/huggingface
export WANDB_PROJECT=heron
export PROJECT_NAME=video_blip_soccernet/exp003
export WANDB_NAME=$PROJECT_NAME

deepspeed train.py --config_file projects/$PROJECT_NAME.yml
