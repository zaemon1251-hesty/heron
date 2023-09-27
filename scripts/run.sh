#!/bin/bash
export WANDB_PROJECT=heron
export PROJECT_NAME=video_blip_soccernet/exp002
export WANDB_NAME=$PROJECT_NAME

deepspeed train.py --config_file projects/$PROJECT_NAME.yml
