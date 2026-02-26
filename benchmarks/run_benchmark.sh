#!/bin/bash

# ==============================================================================
# 完整参数对齐的 Benchmark 脚本
# 参考 lightx2v-main0918wan22/benchmarks/run_default.sh
# ==============================================================================

export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=0
export TOKENIZERS_PARALLELISM=false
export lightx2v_path=/usr/games/LightX2V-storyicon
export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None
export PROFILING_DEBUG_LEVEL=2
export ENABLE_PROFILING_DEBUG=true
export ENABLE_GRAPH_MODE=${ENABLE_GRAPH_MODE:-false}

export TORCH_PROFILE=${TORCH_PROFILE:-0}
export USE_CHANNELS_LAST_3D=1
export VAE_BATCH_MODE=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Model configuration
model_path=/data/docker/Wan2.1-I2V-14B-480P
CONFIG_JSON="${lightx2v_path}/configs/dist_infer/wan_i2v_dist_ulysses_offload.json"

# Input configuration (允许命令行覆盖)
PROMPT="${1:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.}"
IMAGE_PATH="${2:-${lightx2v_path}/assets/img_lightx2v.png}"
OUTPUT_PATH="${3:-${lightx2v_path}/save_results/output_$(date +%Y%m%d_%H%M%S).mp4}"

NEGATIVE_PROMPT="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# 推理参数（对齐 main0918wan22）
SEED=${SEED:-233}
NF=${NF:-5.0}             # 视频秒数
MOTION=${MOTION:-1.3}     # 运动幅度
SHIFT=${SHIFT:-5}         # sample_shift
STEP=${STEP:-8}           # 推理步数
SIZE_STR=${SIZE_STR:-360P}  # 分辨率

# Warmup configuration
WARMUP_ITERATIONS=${WARMUP_ITERATIONS:-1}
WARMUP_FLAG=""
if [ "$WARMUP_ITERATIONS" -gt 0 ]; then
    WARMUP_FLAG="--enable_warmup --warmup_iterations $WARMUP_ITERATIONS"
fi

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Display configuration
echo "======================================"
echo "🚀 LightX2V I2V Benchmark"
echo "======================================"
echo "Model:        $model_path"
echo "Config:       $CONFIG_JSON"
echo "------------------------------------"
echo "Input Image:  $IMAGE_PATH"
echo "Prompt:       $PROMPT"
echo "Output:       $OUTPUT_PATH"
echo "------------------------------------"
echo "Seed:         $SEED"
echo "Video Seconds: $NF s"
echo "Motion:       $MOTION"
echo "Shift:        $SHIFT"
echo "Step:         $STEP"
echo "Resolution:   $SIZE_STR"
echo "------------------------------------"
echo "DTYPE:              $DTYPE"
echo "TORCH_PROFILE:      $TORCH_PROFILE"
echo "USE_CHANNELS_LAST_3D: $USE_CHANNELS_LAST_3D"
echo "WARMUP_ITERATIONS:  $WARMUP_ITERATIONS"
echo "======================================"

# Activate virtual environment
source ${lightx2v_path}/.venv/bin/activate

# Run inference
torchrun --nproc_per_node=4 --master_port 29500 \
  ${lightx2v_path}/benchmarks/inference_dist.py \
  --model_cls wan2.1 \
  --task i2v \
  --model_path $model_path \
  --config_json "$CONFIG_JSON" \
  --image_path "$IMAGE_PATH" \
  --prompt "$PROMPT" \
  --negative_prompt "$NEGATIVE_PROMPT" \
  --save_result_path "$OUTPUT_PATH" \
  --seed $SEED \
  --nf $NF \
  --motion $MOTION \
  --shift $SHIFT \
  --step $STEP \
  --size_str $SIZE_STR \
  $WARMUP_FLAG

INFERENCE_EXIT_CODE=$?

echo ""
echo "======================================"
if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    echo "✅ Benchmark completed successfully!"
    if [ "$TORCH_PROFILE" = "1" ]; then
        echo "📊 Profiling results saved in: ${lightx2v_path}/benchmarks/"
    fi
else
    echo "❌ Benchmark failed with exit code $INFERENCE_EXIT_CODE"
fi
echo "======================================"

exit $INFERENCE_EXIT_CODE
