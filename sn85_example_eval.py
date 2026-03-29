"""
replay_buffer/example_eval.py

Example showing how to wire your upscaling or compression model into
the local test framework and run a before/after comparison.

Replace the placeholder functions with your actual model calls.
"""
import sys
import os
import subprocess
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from replay_buffer.local_test_framework import (
    load_clip_registry,
    evaluate_upscaling,
    evaluate_compression,
    save_result,
    compare_models,
    print_summary,
    VMAF_THRESHOLDS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# REPLACE THESE WITH YOUR ACTUAL MODEL CALLS
# ═══════════════════════════════════════════════════════════════════════════════

def my_upscaler(input_path: str, task_type: str, output_path: str) -> None:
    """
    Call your upscaling model here.

    task_type is one of: HD24K, SD24K, SD2HD
    HD24K: input is 1080p, output should be 4K (3840x2160)
    SD24K: input is SD (~480p), output should be 4K (3840x2160)

    This example calls the video2x upscaler used in the base miner code.
    Replace with your actual model.
    """
    # Example: FFmpeg bicubic upscale (baseline, not AI)
    # Replace with: Real-ESRGAN, BSRGAN, Video2X, etc.
    resolution_map = {
        "HD24K": "3840x2160",
        "SD24K": "3840x2160",
        "SD2HD": "1920x1080",
    }
    target_res = resolution_map.get(task_type, "3840x2160")
    w, h = target_res.split("x")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"scale={w}:{h}:flags=bicubic",
        "-c:v", "libx264", "-crf", "18",
        "-an",
        output_path, "-hide_banner", "-loglevel", "error"
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg upscaling failed for {input_path}")


def my_compressor(
    input_path: str,
    vmaf_threshold: float,
    target_codec: str,
    codec_mode: str,
    target_bitrate: float,
    output_path: str,
) -> None:
    """
    Call your compression model here.

    The validator sends these exact parameters:
      vmaf_threshold: 85, 89, or 93 (randomly chosen)
      target_codec: "av1" or "hevc"
      codec_mode: "CRF" or "VBR"
      target_bitrate: 5.0, 8.0, or 10.0 Mbps

    Your compressor must achieve compression_rate < 0.80 (>1.25x) AND
    VMAF >= vmaf_threshold to earn a non-zero score.
    """
    # Codec name mapping
    codec_map = {"av1": "libaom-av1", "hevc": "libx265"}
    encoder = codec_map.get(target_codec, "libx265")

    if codec_mode == "CRF":
        # CRF mode: constant quality (ignores target_bitrate)
        # AV1: CRF 28-35 typically gives good compression with VMAF 85+
        # HEVC: CRF 23-28
        crf = 28 if target_codec == "av1" else 25
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", encoder, "-crf", str(crf),
            "-an",
            output_path, "-hide_banner", "-loglevel", "error"
        ]
    else:
        # VBR mode: variable bitrate
        bitrate_kbps = int(target_bitrate * 1000)
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", encoder, "-b:v", f"{bitrate_kbps}k",
            "-an",
            output_path, "-hide_banner", "-loglevel", "error"
        ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg compression failed for {input_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION RUNS
# ═══════════════════════════════════════════════════════════════════════════════

def run_upscaling_eval(model_tag: str = "baseline", content_seconds: int = 5):
    """Run upscaling evaluation across all buffer clips."""
    clips = load_clip_registry()
    if not clips:
        print("No clips in buffer. Run:")
        print("  python replay_buffer/local_test_framework.py --download --n-clips 10")
        return

    print(f"Evaluating upscaling model '{model_tag}' on {len(clips)} clips...")
    results = []
    for i, clip in enumerate(clips):
        print(f"  [{i+1}/{len(clips)}] {clip.clip_id} ({clip.task_type})...")
        result = evaluate_upscaling(
            clip=clip,
            upscaler_fn=my_upscaler,
            content_seconds=content_seconds,
            model_tag=model_tag,
        )
        save_result(result, "upscaling", model_tag)
        results.append(result)

        # Quick print
        if result.vmaf_score is not None:
            vmaf_str = f"VMAF={result.vmaf_score:.1f}"
        else:
            vmaf_str = "VMAF=N/A (install ffmpeg+libvmaf)"
        if result.s_f is not None:
            sf_str = f"S_F={result.s_f:.4f}"
        else:
            sf_str = "S_F=N/A (PieAPP not available)"
        frame_str = "frames:OK" if result.frame_count_match else "frames:MISMATCH"
        print(f"    {vmaf_str} {sf_str} {frame_str} ({result.processing_time_s:.1f}s)")

    print_summary([r.__dict__ if hasattr(r, '__dict__') else r for r in results],
                  "upscaling", model_tag)


def run_compression_eval(model_tag: str = "baseline"):
    """Run compression evaluation across all buffer clips with random validator params."""
    import random
    clips = load_clip_registry()
    if not clips:
        print("No clips in buffer.")
        return

    print(f"Evaluating compression model '{model_tag}' on {len(clips)} clips...")
    results = []
    for i, clip in enumerate(clips):
        # Randomise params as validator does
        threshold = random.choice(VMAF_THRESHOLDS)
        print(f"  [{i+1}/{len(clips)}] {clip.clip_id} (threshold={threshold})...")
        result = evaluate_compression(
            clip=clip,
            compressor_fn=my_compressor,
            vmaf_threshold=threshold,
            model_tag=model_tag,
        )
        save_result(result, "compression", model_tag)
        results.append(result)

        vmaf_str  = f"VMAF={result.vmaf_score:.1f}" if result.vmaf_score else "VMAF=N/A"
        ratio_str = f"{result.compression_ratio:.1f}x" if result.compression_ratio else "N/A"
        score_str = f"score={result.final_score:.4f}" if result.final_score else "score=0"
        frame_str = "frames:OK" if result.frame_count_match else "frames:MISMATCH"
        print(f"    {vmaf_str} ratio={ratio_str} {score_str} {frame_str}")

    print_summary([r.__dict__ if hasattr(r, '__dict__') else r for r in results],
                  "compression", model_tag)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["upscaling", "compression"],
                        default="upscaling")
    parser.add_argument("--model-tag", default="baseline")
    parser.add_argument("--content-seconds", type=int, default=5)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--before-tag", default="before")
    parser.add_argument("--after-tag", default="after")
    args = parser.parse_args()

    if args.compare:
        compare_models(args.task, args.before_tag, args.after_tag)
    elif args.task == "upscaling":
        run_upscaling_eval(args.model_tag, args.content_seconds)
    else:
        run_compression_eval(args.model_tag)
