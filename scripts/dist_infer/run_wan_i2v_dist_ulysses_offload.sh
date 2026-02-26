#!/bin/bash

# set path firstly
lightx2v_path=/usr/games/LightX2V-storyicon
model_path=/data/docker/Wan2.1-I2V-14B-480P

export CUDA_VISIBLE_DEVICES=0,1,2,3

source /usr/games/LightX2V-storyicon/.venv/bin/activate

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# 使用I2V任务，需要提供输入图像
torchrun --nproc_per_node=4 -m lightx2v.infer \
--model_cls wan2.1 \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/dist_infer/wan_i2v_dist_ulysses_offload.json \
--image_path ${lightx2v_path}/assets/img_lightx2v.png \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan_i2v_offload.mp4
