#!/usr/bin/env python3
# coding=utf-8
"""
分布式 I2V 推理脚本（完整参数对齐版本）
参考 lightx2v-main0918wan22/benchmarks/inference.py
"""

import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import csv
import gc
import json
import random
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from loguru import logger

# 导入推理相关模块
from lightx2v.common.ops import *  # noqa: F403
from lightx2v.models.runners.wan.wan_runner import WanRunner  # noqa: F401
from lightx2v.utils.envs import *  # noqa: F403
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
from lightx2v.utils.utils import seed_all, validate_config_paths, validate_task_arguments
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

warnings.filterwarnings("ignore")

# 分辨率映射（与前端一致）
STR2SIZE = {
    "1080P": (1920, 1080),
    "720P": (1280, 720),
    "540P": (960, 544),
    "480P": (832, 480),
    "360P": (640, 360),
}


def resize_with_max_area(img_pil, max_area, vae_stride=(4, 8, 8), patch_size=(1, 2, 2)):
    """
    根据最大面积和VAE步长调整图像尺寸
    与前端 dis-webui-file-refine.py 的 resize_with_max_area 函数保持一致
    """
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]
    aspect_ratio = h / w

    # 计算latent空间的尺寸
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) //
        vae_stride[1] // patch_size[1] * patch_size[1]
    )
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) //
        vae_stride[2] // patch_size[2] * patch_size[2]
    )

    # 转换回原图空间的尺寸
    new_h = lat_h * vae_stride[1]
    new_w = lat_w * vae_stride[2]

    # 使用bicubic resize
    img_resized = img_pil.resize((new_w, new_h), Image.BICUBIC)

    return img_resized, new_w, new_h


def trace_handler(p):
    """Torch profiler trace handler，保存统计信息"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    metadata = {
        "TIMESTAMP": timestamp,
        "COMMAND_LINE": sys.argv,
        "ENVIRONMENT_VARIABLES": dict(os.environ),
    }
    metadata_path = os.path.join(script_dir, f"{timestamp}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Torch Profiler: Metadata saved to {metadata_path}")

    key_averages = p.key_averages()
    events = []
    for item in key_averages:
        events.append({
            'use_device': item.use_device,
            'self_device_time_total_str': item.self_device_time_total_str,
            'self_device_memory_usage': item.self_device_memory_usage,
            'self_cpu_time_total_str': item.self_cpu_time_total_str,
            'self_cpu_memory_usage': item.self_cpu_memory_usage,
            'scope': item.scope,
            'key': item.key,
            'device_time_total_str': item.device_time_total_str,
            'cpu_time_total_str': item.cpu_time_total_str,
            'count': item.count,
        })

    if events:
        fieldnames = sorted(events[0].keys())
        csv_filename = os.path.join(script_dir, f"{timestamp}_events.csv")
        with open(csv_filename, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)
        logger.info(f"✓ Torch Profiler: Events saved to {csv_filename}")

    trace_path = os.path.join(script_dir, f"{timestamp}_trace.json")
    p.export_chrome_trace(trace_path)
    logger.info(f"✓ Torch Profiler: Trace saved to {trace_path}")


class MockProfiler:
    """Mock profiler for when profiling is disabled"""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    def step(self):
        pass


def create_profiler(rank):
    """创建 torch profiler（仅在 rank 0 且 TORCH_PROFILE=1 时启用）"""
    if rank == 0 and os.environ.get("TORCH_PROFILE", "0") == "1":
        logger.info("Starting torch profiler...")
        return torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
            on_trace_ready=trace_handler,
        )
    return MockProfiler()


def init_runner(config):
    """初始化推理runner"""
    torch.set_grad_enabled(False)
    runner = RUNNER_REGISTER[config.model_cls](config)
    runner.init_modules()
    return runner


def main():
    parser = argparse.ArgumentParser(description="分布式 I2V 推理脚本")

    # 输入输出参数
    parser.add_argument("--prompt", type=str, required=True, help="输入prompt")
    parser.add_argument("--image_path", type=str, required=True, help="输入图像路径")
    parser.add_argument("--save_result_path", type=str, required=True, help="输出视频路径")
    parser.add_argument("--negative_prompt", type=str, default="", help="负向prompt")

    # 推理参数（对齐 main0918wan22）
    parser.add_argument("--seed", type=int, default=233, help="随机种子，-1为随机生成")
    parser.add_argument("--nf", type=float, default=5.0, help="视频秒数（3-8之间）")
    parser.add_argument("--motion", type=float, default=1.3, help="运动幅度（0.5-10之间）")
    parser.add_argument("--shift", type=int, default=5, help="sample_shift参数（1-17之间）")
    parser.add_argument("--step", type=int, default=8, help="推理步数（3-12之间）")
    parser.add_argument("--size_str", type=str, default="360P",
                        choices=["360P", "480P", "540P", "720P", "1080P"],
                        help="分辨率")

    # 模型配置参数
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--config_json", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model_cls", type=str, default="wan2.1", help="模型类别")
    parser.add_argument("--task", type=str, default="i2v", help="任务类型")

    # Warmup & Profiling
    parser.add_argument("--enable_warmup", action="store_true",
                        help="启用预热推理（首次运行dummy推理来预热CUDA kernels）")
    parser.add_argument("--warmup_iterations", type=int, default=1,
                        help="预热迭代次数（默认1次）")

    args = parser.parse_args()
    validate_task_arguments(args)

    # 验证输入文件
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"图像文件不存在: {args.image_path}")

    # 处理随机种子
    if args.seed < 0:
        args.seed = random.randint(1, 999999)

    # 分辨率限制
    if args.size_str == "720P" and args.nf > 6:
        logger.warning("720P目前最多支持到生成6s，超过则以6s生成")
        args.nf = 6
    if args.size_str == "1080P" and args.nf > 5:
        logger.warning("1080P目前最多支持到生成5s，超过则以5s生成")
        args.nf = 5

    # 创建输出目录
    output_dir = os.path.dirname(args.save_result_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 加载并处理图像
    logger.info(f"加载图像: {args.image_path}")
    img_pil = Image.open(args.image_path).convert("RGB")
    w, h = STR2SIZE[args.size_str]
    resized_img, new_w, new_h = resize_with_max_area(img_pil, max_area=w * h)

    # 保存调整后的图像到临时目录
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_input")
    os.makedirs(temp_dir, exist_ok=True)
    temp_image_path = os.path.join(temp_dir, f"temp_{int(time.time())}.png")
    resized_img.save(temp_image_path)
    logger.info(f"调整后的图像尺寸: {new_w}x{new_h}")

    # 计算视频长度（帧数）
    target_video_length = int(args.nf * 16) + 1

    # 更新 args 对象以包含所有推理参数（在 set_config 之前）
    args.image_path = temp_image_path
    args.save_video_path = args.save_result_path  # 注意：内部使用 save_video_path
    args.target_video_length = target_video_length
    args.sample_shift = args.shift
    args.infer_steps = args.step
    args.target_width = new_w
    args.target_height = new_h

    seed_all(args.seed)

    # 加载配置
    logger.info("="*80)
    logger.info("加载配置并初始化...")
    config = set_config(args)

    # 初始化分布式环境（必须在参数覆盖之前）
    if config.get("parallel"):
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        platform_device.init_parallel_env()
        set_parallel_config(config)

    # 强制覆盖配置文件中的参数（必须在 set_parallel_config 之后）
    config.target_width = new_w
    config.target_height = new_h
    config.target_video_length = target_video_length
    config.sample_shift = args.shift
    config.infer_steps = args.step

    # 确保 warmup 需要的属性存在于 config 中
    config.prompt = args.prompt
    config.image_path = args.image_path
    config.seed = args.seed

    # 打印配置
    logger.info("="*80)
    logger.info("推理配置:")
    logger.info(f"  - 分辨率: {args.size_str} ({new_w}x{new_h})")
    logger.info(f"  - 视频长度: {args.nf}秒 ({target_video_length}帧)")
    logger.info(f"  - 推理步数: {args.step}")
    logger.info(f"  - Sample Shift: {args.shift}")
    logger.info(f"  - 运动幅度: {args.motion}")
    logger.info(f"  - 随机种子: {args.seed}")
    logger.info(f"  - 输出路径: {args.save_result_path}")
    logger.info("="*80)

    print_config(config)
    validate_config_paths(config)

    # 初始化runner
    logger.info("初始化模型...")
    runner = init_runner(config)

    # 创建 torch profiler
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch_profiler = create_profiler(rank=rank)
    if rank == 0 and os.environ.get("TORCH_PROFILE", "0") == "1":
        logger.info(f"✓ Torch Profiler enabled on rank {rank}")

    is_rank0 = not dist.is_initialized() or dist.get_rank() == 0

    # Warmup 逻辑
    if args.enable_warmup:
        warmup_iterations = args.warmup_iterations
        if is_rank0:
            logger.info("="*80)
            logger.info(f"🔥 开始预热推理（Warmup x {warmup_iterations}）...")
            logger.info("="*80)

        # 保存原始配置（config 是 LockableDict，支持属性访问）
        original_prompt = config.prompt
        original_image_path = config.image_path
        original_save_video_path = config.save_video_path
        original_seed = config.seed

        # 创建 warmup 用的 dummy 图像
        warmup_output_dir = os.path.join(os.path.dirname(__file__), "temp_warmup")
        os.makedirs(warmup_output_dir, exist_ok=True)
        dummy_img = Image.new('RGB', (new_w, new_h), color=(128, 128, 128))
        warmup_image_path = os.path.join(warmup_output_dir, f"warmup_img_{int(time.time())}.png")
        dummy_img.save(warmup_image_path)

        # Warmup 配置
        warmup_prompt = "A dummy warmup video for preheating CUDA kernels."
        warmup_seed = 999999

        # 创建 input_info 用于 warmup
        warmup_input_info = init_empty_input_info(args.task)

        warmup_times = []
        for i in range(warmup_iterations):
            warmup_output_path = os.path.join(warmup_output_dir, f"warmup_{int(time.time())}_{i}.mp4")

            # 更新配置
            config.prompt = warmup_prompt
            config.image_path = warmup_image_path
            config.save_video_path = warmup_output_path
            config.seed = warmup_seed

            # 更新 warmup_input_info
            warmup_data = {
                "prompt": warmup_prompt,
                "image_path": warmup_image_path,
                "seed": warmup_seed,
                "save_result_path": warmup_output_path,
            }
            update_input_info_from_dict(warmup_input_info, warmup_data)

            if is_rank0:
                logger.info(f"  预热 [{i+1}/{warmup_iterations}] 开始...")

            warmup_start = time.time()
            runner.run_pipeline(warmup_input_info)
            warmup_end = time.time()
            warmup_time = warmup_end - warmup_start
            warmup_times.append(warmup_time)

            if is_rank0:
                logger.info(f"  预热 [{i+1}/{warmup_iterations}] 完成，耗时: {warmup_time:.2f}秒")
                if os.path.exists(warmup_output_path):
                    os.remove(warmup_output_path)

            torch.cuda.empty_cache()

        if is_rank0:
            avg_time = sum(warmup_times) / len(warmup_times)
            min_time = min(warmup_times)
            max_time = max(warmup_times)
            logger.info("="*80)
            logger.info(f"✓ 预热完成 ({warmup_iterations}次)")
            logger.info(f"  平均: {avg_time:.2f}s | 最小: {min_time:.2f}s | 最大: {max_time:.2f}s | 波动: {(max_time-min_time)/avg_time*100:.1f}%")
            logger.info("="*80)
            if os.path.exists(warmup_image_path):
                os.remove(warmup_image_path)

        # 恢复原始配置
        config.prompt = original_prompt
        config.image_path = original_image_path
        config.save_video_path = original_save_video_path
        config.seed = original_seed

        torch.cuda.empty_cache()
        gc.collect()

    # 正式推理
    if is_rank0:
        logger.info("="*80)
        if args.enable_warmup:
            logger.info("🚀 开始正式推理（已预热）...")
        else:
            logger.info("🚀 开始正式推理（未预热）...")
        logger.info("="*80)

    # 准备正式推理的 input_info
    input_info = init_empty_input_info(args.task)
    data = args.__dict__
    update_input_info_from_dict(input_info, data)

    start_time = time.time()
    with torch_profiler:
        runner.run_pipeline(input_info)
        torch_profiler.step()
    end_time = time.time()

    # 检查输出
    if is_rank0:
        max_wait = 10
        for i in range(max_wait * 2):
            if os.path.exists(args.save_result_path):
                logger.info("="*80)
                logger.info(f"✓ 视频生成成功!")
                logger.info(f"  - 输出路径: {args.save_result_path}")
                logger.info(f"  - 耗时: {end_time - start_time:.2f}秒")
                logger.info("="*80)
                break
            time.sleep(0.5)
        else:
            logger.error(f"✗ 视频生成失败: 输出文件不存在")

    # 清理临时文件
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    # 清理分布式环境
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
