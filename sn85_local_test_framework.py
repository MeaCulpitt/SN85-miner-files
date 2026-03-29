"""
replay_buffer/local_test_framework.py

Local test clip framework for SN85 Vidaio -- the replay buffer equivalent.

WHY THIS EXISTS
===============
SN44 has a deterministic scoring function that can be exactly replicated locally.
SN85 uses VMAF (requires FFmpeg + libvmaf) and PieAPP (requires model weights + GPU)
computed by the validator against reference videos you don't have.

However, you CAN run your upscaling/compression model on local test clips and
measure the SAME metrics the validator uses, which gives you:
  - A directionally accurate prediction of your score before deployment
  - A consistent test set to compare mutations against each other
  - Detection of regressions (model change hurt quality vs helped)

This is not an exact replay -- it is the best local approximation available.

VALIDATOR CLIP DISTRIBUTION (from worker.py calculate_task_thresholds())
=========================================================================
Task type probabilities (from VideoSchedulerConfig defaults):
  HD24K (40%): Source 1080p → output 4K (3840x2160). Pexels landscape clips.
  SD2HD (30%): Currently INACTIVE in production (commented out in video_utils.py).
  SD24K (30%): Source SD → output 4K (3840x2160). Pexels landscape clips.
  4K28K (10%): Future -- 8K output. Not currently active.

Active task types: HD24K (57%) and SD24K (43%) of actual traffic.

Clip duration: 5s chunks (default) or 10s chunks (if miner declared ContentLength.TEN).
Clip source: Pexels API (landscape video, min 15s raw, chunked to 5s/10s segments).
Content: nature, landscapes, city footage, sports -- high motion and static mix.

Compression clips: same Pexels source, no downscaling. VMAF threshold varies
per task (85/89/93 randomly chosen from VMAF_QUALITY_THRESHOLD enum).
Target codec: AV1 or HEVC (randomly chosen). Mode: CRF or VBR. Bitrate: 5/8/10 Mbps.

SETUP
=====
1. Install ffmpeg with libvmaf support:
   Ubuntu: sudo apt install ffmpeg
   Verify: ffmpeg -filters | grep vmaf

2. Install PieAPP (optional, upscaling only):
   pip install pieapp  (or use the validator's pieapp_metric.py)

3. Download test clips:
   python replay_buffer/local_test_framework.py --download --n-clips 10

4. Run evaluation:
   python replay_buffer/local_test_framework.py --eval --task-type upscaling
   python replay_buffer/local_test_framework.py --eval --task-type compression

5. Compare mutations:
   python replay_buffer/local_test_framework.py --compare \
     --before-model /path/to/old/model \
     --after-model /path/to/new/model
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Paths ─────────────────────────────────────────────────────────────────────
BUFFER_DIR   = Path(__file__).parent
ROOT         = BUFFER_DIR.parent
CLIPS_DIR    = BUFFER_DIR / "clips"
RESULTS_DIR  = BUFFER_DIR / "results"
REGISTRY_DIR = ROOT / "registry"

CLIPS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── Validator distribution constants (from source) ────────────────────────────
# worker.py calculate_task_thresholds() with VideoSchedulerConfig defaults
TASK_WEIGHTS = {
    "HD24K": 0.40,   # 1080p source → 4K output
    "SD24K": 0.30,   # SD source   → 4K output
    # "SD2HD": 0.30  # INACTIVE in production (commented out in video_utils.py)
    # "4K28K": 0.10  # Not currently active
}

# Normalised weights for active task types
_total = sum(TASK_WEIGHTS.values())
TASK_WEIGHTS_NORMALISED = {k: v/_total for k, v in TASK_WEIGHTS.items()}

# Validator codec/quality constants (from neurons/validator.py)
VMAF_THRESHOLDS   = [85, 89, 93]   # VMAF_QUALITY_THRESHOLD enum
TARGET_CODECS     = ["av1", "hevc"]
CODEC_MODES       = ["CRF", "VBR"]
TARGET_BITRATES   = [5.0, 8.0, 10.0]

# Resolution map per task type (from video_utils.py DOWNSCALE_HEIGHTS + TARGET_RESOLUTIONS)
TASK_RESOLUTIONS = {
    "HD24K": {"source_height": 1080, "output_w": 3840, "output_h": 2160},
    "SD24K": {"source_height": 480,  "output_w": 3840, "output_h": 2160},
    "SD2HD": {"source_height": 480,  "output_w": 1920, "output_h": 1080},
}

# Scoring formula replicas (from scoring_function.py and video_similarity_score_sythetic.py)
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def pieapp_to_final_score(pieapp_score: float) -> float:
    """Exact replica of calculate_final_score() in video_similarity_score_sythetic.py"""
    s = sigmoid(pieapp_score)
    at_zero = (1 - (math.log10(sigmoid(0) + 1) / math.log10(3.5))) ** 2.5
    at_two  = (1 - (math.log10(sigmoid(2.0) + 1) / math.log10(3.5))) ** 2.5
    val     = (1 - (math.log10(s + 1) / math.log10(3.5))) ** 2.5
    return 1 - ((val - at_zero) / (at_two - at_zero))

def length_score(content_seconds: float) -> float:
    """S_L = log(1 + t) / log(321)"""
    return math.log(1 + content_seconds) / math.log(321)

def upscaling_final_score(s_q: float, s_l: float) -> tuple[float, float]:
    """Returns (S_pre, S_F)"""
    s_pre = s_q * 0.5 + s_l * 0.5
    s_f   = 0.1 * math.exp(6.979 * (s_pre - 0.5))
    return s_pre, s_f

def compression_score(vmaf: float, rate: float, threshold: float) -> float:
    """
    Exact replica of calculate_compression_score() from scoring_function.py.
    vmaf: 0-100. rate: compressed/original (0-1). threshold: 85/89/93.
    """
    hard_cutoff = threshold - 5.0
    if rate >= 0.80:
        return 0.0
    if vmaf < hard_cutoff:
        return 0.0
    ratio = 1 / rate
    if vmaf < threshold:
        pos = (vmaf - hard_cutoff) / 5.0
        qf  = 0.7 * (pos ** 2)
        cc  = ((ratio - 1) / 19) ** 1.5 if ratio <= 20 else 1.0 + 0.3 * math.log(ratio / 20)
        return min(1.0, (cc * qf) / 1.12)
    excess = vmaf - threshold
    qc = 0.7 + 0.3 * min(1.0, excess / (100 - threshold))
    cc = ((ratio - 1.25) / 18.75) ** 0.9 if ratio <= 20 else 1.0 + 0.1 * math.log(ratio / 20)
    return min(1.0, (0.70 * cc + 0.30 * qc) / 1.12)


# ── Test clip management ──────────────────────────────────────────────────────
@dataclass
class TestClip:
    clip_id: str
    path: str
    task_type: str          # HD24K or SD24K
    source_height: int
    output_width: int
    output_height: int
    duration_seconds: float
    source: str             # pexels / local / synthetic
    added_timestamp: str

    def to_dict(self):
        return asdict(self)


def load_clip_registry() -> list[TestClip]:
    path = CLIPS_DIR / "registry.jsonl"
    if not path.exists():
        return []
    clips = []
    for line in path.read_text().split("\n"):
        if line.strip():
            try:
                clips.append(TestClip(**json.loads(line)))
            except Exception:
                continue
    return clips


def save_clip(clip: TestClip):
    path = CLIPS_DIR / "registry.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(clip.to_dict()) + "\n")
    print(f"  Registered clip: {clip.clip_id} ({clip.task_type}, {clip.duration_seconds}s)")


def download_pexels_clip(pexels_api_key: str, task_type: str,
                          duration_s: int = 5) -> Optional[TestClip]:
    """
    Download a random Pexels clip matching the task type.
    Mirrors the validator's Pexels selection logic from video_utils.py.
    """
    if not HAS_REQUESTS:
        print("  requests not installed. pip install requests")
        return None
    if not pexels_api_key:
        print("  PEXELS_API_KEY not set. Set env var or pass --pexels-key.")
        return None

    res = TASK_RESOLUTIONS[task_type]
    min_height = res["source_height"]

    # Search for landscape videos matching resolution
    headers = {"Authorization": pexels_api_key}
    queries = ["nature landscape", "city timelapse", "ocean waves", "mountains",
               "traffic", "forest", "aerial drone", "sports"]
    query = random.choice(queries)

    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers,
            params={"query": query, "per_page": 15, "orientation": "landscape"},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  Pexels API error: {e}")
        return None

    # Filter videos by resolution and duration
    candidates = []
    for video in data.get("videos", []):
        dur = video.get("duration", 0)
        if dur < 15:  # need at least 15s to extract a clean chunk
            continue
        for vf in video.get("video_files", []):
            h = vf.get("height", 0)
            w = vf.get("width", 0)
            if h >= min_height and w > h:  # landscape
                candidates.append((video["id"], vf["link"], h, dur))
                break

    if not candidates:
        print(f"  No suitable Pexels clips found for {task_type}. Try a different query.")
        return None

    vid_id, link, height, source_dur = random.choice(candidates)
    clip_id = f"{task_type}_{vid_id}_{int(time.time())}"
    raw_path = CLIPS_DIR / f"{clip_id}_raw.mp4"
    chunk_path = CLIPS_DIR / f"{clip_id}_{duration_s}s.mp4"

    print(f"  Downloading Pexels video {vid_id} ({height}p, {source_dur}s)...")
    try:
        with requests.get(link, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(raw_path, "wb") as f:
                for chunk in resp.iter_content(1024 * 1024):
                    f.write(chunk)
    except Exception as e:
        print(f"  Download failed: {e}")
        return None

    # Extract a chunk (starting at 5s to avoid intros)
    start = random.randint(5, max(5, source_dur - duration_s - 5))
    cmd = [
        "ffmpeg", "-y", "-i", str(raw_path),
        "-ss", str(start), "-t", str(duration_s),
        "-c:v", "libx264", "-crf", "18",  # high quality for reference
        "-an",  # no audio
        str(chunk_path), "-hide_banner", "-loglevel", "error"
    ]
    result = subprocess.run(cmd)
    raw_path.unlink(missing_ok=True)

    if result.returncode != 0 or not chunk_path.exists():
        print(f"  FFmpeg chunking failed")
        return None

    clip = TestClip(
        clip_id=clip_id,
        path=str(chunk_path),
        task_type=task_type,
        source_height=height,
        output_width=res["output_width"],
        output_height=res["output_height"],
        duration_seconds=duration_s,
        source="pexels",
        added_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    save_clip(clip)
    return clip


def add_local_clip(local_path: str, task_type: str, duration_s: float) -> TestClip:
    """Register a locally-obtained clip into the buffer."""
    res = TASK_RESOLUTIONS.get(task_type, TASK_RESOLUTIONS["HD24K"])
    clip_id = f"{task_type}_local_{int(time.time())}"
    dest = CLIPS_DIR / f"{clip_id}.mp4"
    import shutil
    shutil.copy(local_path, dest)
    clip = TestClip(
        clip_id=clip_id,
        path=str(dest),
        task_type=task_type,
        source_height=res["source_height"],
        output_width=res["output_width"],
        output_height=res["output_height"],
        duration_seconds=duration_s,
        source="local",
        added_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    save_clip(clip)
    return clip


# ── VMAF computation ──────────────────────────────────────────────────────────
def compute_vmaf(reference_path: str, distorted_path: str,
                 n_samples: int = 10) -> Optional[float]:
    """
    Compute VMAF score using FFmpeg. Mirrors validate's vmaf_metric.py.
    Requires FFmpeg with libvmaf support.
    """
    # Check ffmpeg availability
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  ffmpeg not found. Install with: sudo apt install ffmpeg")
        return None

    # Check libvmaf support
    result = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True)
    if "vmaf" not in result.stdout:
        print("  ffmpeg does not have libvmaf support.")
        print("  Install: sudo apt install ffmpeg libvmaf-dev")
        return None

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        log_path = f.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", distorted_path,
            "-i", reference_path,
            "-lavfi", f"[0:v][1:v]libvmaf=log_path={log_path}:log_fmt=json:n_subsample={max(1, n_samples)}",
            "-f", "null", "-",
            "-hide_banner", "-loglevel", "error"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  VMAF computation failed: {result.stderr[:200]}")
            return None

        with open(log_path) as f:
            data = json.load(f)
        vmaf = data.get("pooled_metrics", {}).get("vmaf", {}).get("harmonic_mean")
        if vmaf is None:
            vmaf = data.get("pooled_metrics", {}).get("vmaf", {}).get("mean")
        return float(vmaf) if vmaf is not None else None
    except Exception as e:
        print(f"  VMAF error: {e}")
        return None
    finally:
        Path(log_path).unlink(missing_ok=True)


def get_file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def get_frame_count(path: str) -> Optional[int]:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_streams",
         "-select_streams", "v:0", path],
        capture_output=True, text=True
    )
    for line in result.stdout.split("\n"):
        if line.startswith("nb_frames="):
            try:
                return int(line.split("=")[1])
            except ValueError:
                return None
    return None


# ── Upscaling evaluation ──────────────────────────────────────────────────────
@dataclass
class UpscalingResult:
    clip_id: str
    task_type: str
    content_seconds: float
    vmaf_score: Optional[float]
    vmaf_passes_gate: bool    # VMAF/100 > 0.5
    pieapp_score: Optional[float]
    s_q: Optional[float]      # PieAPP → final quality score
    s_l: float                # length score
    s_pre: Optional[float]
    s_f: Optional[float]      # final upscaling score
    frame_count_match: bool
    output_size_mb: Optional[float]
    processing_time_s: float
    timestamp: str
    model_tag: str


def evaluate_upscaling(clip: TestClip, upscaler_fn,
                        content_seconds: int = 5,
                        model_tag: str = "default") -> UpscalingResult:
    """
    Evaluate an upscaling model against a test clip.

    upscaler_fn: callable(input_path: str, task_type: str, output_path: str) -> None
      Your upscaling model wrapped to match this signature.

    The validator workflow:
      1. Takes a 1080p (HD24K) or SD (SD24K) clip
      2. Sends it to miner (your model upscales it)
      3. Compares miner output against the ORIGINAL high-res using VMAF + PieAPP
    """
    ts = datetime.now(timezone.utc).isoformat()
    s_l = length_score(content_seconds)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        output_path = f.name

    t0 = time.time()
    try:
        upscaler_fn(clip.path, clip.task_type, output_path)
    except Exception as e:
        print(f"  Upscaler failed: {e}")
        return UpscalingResult(
            clip_id=clip.clip_id, task_type=clip.task_type,
            content_seconds=content_seconds,
            vmaf_score=None, vmaf_passes_gate=False, pieapp_score=None,
            s_q=None, s_l=s_l, s_pre=None, s_f=None,
            frame_count_match=False, output_size_mb=None,
            processing_time_s=time.time()-t0, timestamp=ts, model_tag=model_tag
        )
    processing_time = time.time() - t0

    # Frame count check (critical -- mismatch → score=0)
    ref_frames  = get_frame_count(clip.path)
    dist_frames = get_frame_count(output_path)
    frame_match = (ref_frames is not None and dist_frames is not None
                   and ref_frames == dist_frames)
    if not frame_match:
        print(f"  WARNING: Frame count mismatch! ref={ref_frames}, output={dist_frames}")

    output_size = get_file_size_mb(output_path) if Path(output_path).exists() else None

    # VMAF (requires ffmpeg + libvmaf)
    vmaf = compute_vmaf(clip.path, output_path)
    vmaf_gate = (vmaf is not None and vmaf / 100 > 0.5)

    # PieAPP (skip if not available)
    pieapp = None
    s_q = None
    s_pre = None
    s_f = None
    if vmaf_gate:
        pieapp = _try_compute_pieapp(clip.path, output_path)
        if pieapp is not None:
            s_q = pieapp_to_final_score(pieapp)
            s_pre, s_f = upscaling_final_score(s_q, s_l)

    Path(output_path).unlink(missing_ok=True)

    return UpscalingResult(
        clip_id=clip.clip_id, task_type=clip.task_type,
        content_seconds=content_seconds,
        vmaf_score=vmaf, vmaf_passes_gate=vmaf_gate,
        pieapp_score=pieapp, s_q=s_q, s_l=s_l, s_pre=s_pre, s_f=s_f,
        frame_count_match=frame_match,
        output_size_mb=output_size,
        processing_time_s=processing_time,
        timestamp=ts, model_tag=model_tag,
    )


def _try_compute_pieapp(ref_path: str, dist_path: str,
                         n_frames: int = 4) -> Optional[float]:
    """
    Attempt to compute PieAPP score. Returns None if PieAPP not available.
    The validator uses 4 randomly sampled frames (pieapp_sample_count=4).

    If you have the validator's pieapp_metric.py:
      - Copy services/scoring/pieapp_metric.py to replay_buffer/
      - Set VIDAIO_REPO_PATH env var
    """
    # Try to import PieAPP metric from validator source
    vidaio_path = os.environ.get("VIDAIO_REPO_PATH",
                                  str(ROOT.parent / "vidaio-subnet"))
    pieapp_path = Path(vidaio_path) / "services" / "scoring" / "pieapp_metric.py"

    if pieapp_path.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("pieapp_metric", pieapp_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.calculate_pieapp_score(ref_path, dist_path, n_frames=n_frames)
        except Exception as e:
            pass  # Fall through to alternative

    # Alternative: use lpips as a proxy if available
    try:
        import torch
        import lpips
        import cv2
        import numpy as np
        fn = lpips.LPIPS(net='vgg')
        ref = cv2.imread(ref_path)   # This won't work for video -- placeholder
        if ref is None:
            return None
        # Real implementation would sample frames -- see pieapp_metric.py for reference
        return None
    except ImportError:
        pass

    print("  PieAPP not available. Install validator stack or set VIDAIO_REPO_PATH.")
    print("  S_Q cannot be computed. VMAF and frame-match checks still work.")
    return None


# ── Compression evaluation ────────────────────────────────────────────────────
@dataclass
class CompressionResult:
    clip_id: str
    vmaf_threshold: int
    target_codec: str
    codec_mode: str
    target_bitrate: float
    vmaf_score: Optional[float]
    compression_rate: Optional[float]   # output_size / input_size
    compression_ratio: Optional[float]  # input_size / output_size
    final_score: Optional[float]
    frame_count_match: bool
    input_size_mb: float
    output_size_mb: Optional[float]
    processing_time_s: float
    timestamp: str
    model_tag: str


def evaluate_compression(clip: TestClip, compressor_fn,
                          vmaf_threshold: Optional[int] = None,
                          target_codec: Optional[str] = None,
                          codec_mode: Optional[str] = None,
                          target_bitrate: Optional[float] = None,
                          model_tag: str = "default") -> CompressionResult:
    """
    Evaluate a compression model against a test clip.

    compressor_fn: callable(
        input_path: str,
        vmaf_threshold: float,
        target_codec: str,
        codec_mode: str,
        target_bitrate: float,
        output_path: str
    ) -> None

    Parameters are randomly selected to match the validator's distribution
    if not specified explicitly.
    """
    ts = datetime.now(timezone.utc).isoformat()

    # Randomly select parameters as validator does
    if vmaf_threshold is None:
        vmaf_threshold = random.choice(VMAF_THRESHOLDS)
    if target_codec is None:
        target_codec = random.choice(TARGET_CODECS)
    if codec_mode is None:
        codec_mode = random.choice(CODEC_MODES)
    if target_bitrate is None:
        target_bitrate = random.choice(TARGET_BITRATES)

    input_size = get_file_size_mb(clip.path)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        output_path = f.name

    t0 = time.time()
    try:
        compressor_fn(clip.path, float(vmaf_threshold), target_codec,
                      codec_mode, target_bitrate, output_path)
    except Exception as e:
        print(f"  Compressor failed: {e}")
        return CompressionResult(
            clip_id=clip.clip_id, vmaf_threshold=vmaf_threshold,
            target_codec=target_codec, codec_mode=codec_mode,
            target_bitrate=target_bitrate, vmaf_score=None,
            compression_rate=None, compression_ratio=None, final_score=None,
            frame_count_match=False, input_size_mb=input_size, output_size_mb=None,
            processing_time_s=time.time()-t0, timestamp=ts, model_tag=model_tag,
        )
    processing_time = time.time() - t0

    # Frame count check
    ref_frames  = get_frame_count(clip.path)
    dist_frames = get_frame_count(output_path)
    frame_match = (ref_frames is not None and dist_frames is not None
                   and ref_frames == dist_frames)
    if not frame_match:
        print(f"  WARNING: Frame count mismatch! ref={ref_frames}, output={dist_frames}")

    output_size = get_file_size_mb(output_path) if Path(output_path).exists() else None
    rate = output_size / input_size if output_size else None
    ratio = input_size / output_size if output_size and output_size > 0 else None

    # VMAF
    vmaf = compute_vmaf(clip.path, output_path) if frame_match else None

    # Score
    score = None
    if vmaf is not None and rate is not None:
        score = compression_score(vmaf, rate, float(vmaf_threshold))

    Path(output_path).unlink(missing_ok=True)

    return CompressionResult(
        clip_id=clip.clip_id, vmaf_threshold=vmaf_threshold,
        target_codec=target_codec, codec_mode=codec_mode,
        target_bitrate=target_bitrate, vmaf_score=vmaf,
        compression_rate=rate, compression_ratio=ratio, final_score=score,
        frame_count_match=frame_match, input_size_mb=input_size,
        output_size_mb=output_size,
        processing_time_s=processing_time, timestamp=ts, model_tag=model_tag,
    )


# ── Result persistence ────────────────────────────────────────────────────────
def save_result(result, task_type: str, model_tag: str):
    path = RESULTS_DIR / f"{task_type}_{model_tag}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def load_results(task_type: str, model_tag: str) -> list:
    path = RESULTS_DIR / f"{task_type}_{model_tag}.jsonl"
    if not path.exists():
        return []
    results = []
    for line in path.read_text().split("\n"):
        if line.strip():
            try:
                results.append(json.loads(line))
            except Exception:
                continue
    return results


def print_summary(results: list, task_type: str, model_tag: str):
    if not results:
        print("  No results.")
        return
    print(f"\n  === {task_type} summary: {model_tag} ({len(results)} clips) ===")

    if task_type == "upscaling":
        vmaf_scores = [r["vmaf_score"] for r in results if r.get("vmaf_score") is not None]
        s_f_scores  = [r["s_f"] for r in results if r.get("s_f") is not None]
        frame_ok    = sum(1 for r in results if r.get("frame_count_match"))

        if vmaf_scores:
            print(f"  VMAF: mean={sum(vmaf_scores)/len(vmaf_scores):.1f} "
                  f"min={min(vmaf_scores):.1f} max={max(vmaf_scores):.1f}")
        if s_f_scores:
            print(f"  S_F:  mean={sum(s_f_scores)/len(s_f_scores):.4f} "
                  f"min={min(s_f_scores):.4f} max={max(s_f_scores):.4f}")
        print(f"  Frame match: {frame_ok}/{len(results)}")

    elif task_type == "compression":
        scores   = [r["final_score"] for r in results if r.get("final_score") is not None]
        vmaf_s   = [r["vmaf_score"] for r in results if r.get("vmaf_score") is not None]
        ratios   = [r["compression_ratio"] for r in results if r.get("compression_ratio") is not None]
        frame_ok = sum(1 for r in results if r.get("frame_count_match"))

        if scores:
            print(f"  Score: mean={sum(scores)/len(scores):.4f} "
                  f"min={min(scores):.4f} max={max(scores):.4f}")
        if vmaf_s:
            print(f"  VMAF:  mean={sum(vmaf_s)/len(vmaf_s):.1f}")
        if ratios:
            print(f"  Ratio: mean={sum(ratios)/len(ratios):.1f}x "
                  f"min={min(ratios):.1f}x max={max(ratios):.1f}x")
        print(f"  Frame match: {frame_ok}/{len(results)}")


def compare_models(task_type: str, before_tag: str, after_tag: str):
    before = load_results(task_type, before_tag)
    after  = load_results(task_type, after_tag)
    print(f"\n  === Model comparison: {before_tag} vs {after_tag} ===")
    print_summary(before, task_type, before_tag)
    print_summary(after,  task_type, after_tag)

    score_key = "s_f" if task_type == "upscaling" else "final_score"
    b_scores = [r[score_key] for r in before if r.get(score_key) is not None]
    a_scores = [r[score_key] for r in after  if r.get(score_key) is not None]
    if b_scores and a_scores:
        b_mean = sum(b_scores) / len(b_scores)
        a_mean = sum(a_scores) / len(a_scores)
        delta  = a_mean - b_mean
        print(f"\n  Delta {score_key}: {delta:+.4f} ({delta/max(b_mean,1e-6)*100:+.1f}%)")
        if delta > 0:
            print(f"  IMPROVEMENT: {after_tag} scores better. Consider deploying.")
        elif delta < -0.01:
            print(f"  REGRESSION:  {after_tag} scores worse. Do not deploy.")
        else:
            print(f"  NO MEANINGFUL CHANGE. Difference within noise.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SN85 Vidaio local test clip framework")
    parser.add_argument("--download", action="store_true",
                        help="Download new test clips from Pexels")
    parser.add_argument("--n-clips", type=int, default=5,
                        help="Number of clips to download (default: 5)")
    parser.add_argument("--duration", type=int, default=5,
                        help="Clip duration in seconds (5 or 10)")
    parser.add_argument("--pexels-key", default=os.environ.get("PEXELS_API_KEY", ""),
                        help="Pexels API key (or set PEXELS_API_KEY env var)")
    parser.add_argument("--add-local", metavar="PATH",
                        help="Add a local clip to the buffer")
    parser.add_argument("--task-type", choices=["HD24K", "SD24K", "SD2HD"],
                        default="HD24K", help="Task type for local clip")
    parser.add_argument("--list", action="store_true",
                        help="List all clips in buffer")
    parser.add_argument("--compare", action="store_true",
                        help="Compare two model runs")
    parser.add_argument("--before-tag", default="before",
                        help="Model tag for 'before' comparison")
    parser.add_argument("--after-tag", default="after",
                        help="Model tag for 'after' comparison")
    parser.add_argument("--eval-type", choices=["upscaling", "compression"],
                        default="upscaling")
    args = parser.parse_args()

    if args.list:
        clips = load_clip_registry()
        print(f"Buffer: {len(clips)} clips in {CLIPS_DIR}")
        for c in clips:
            exists = "OK" if Path(c.path).exists() else "MISSING"
            print(f"  [{exists}] {c.clip_id} {c.task_type} {c.duration_seconds}s {c.source}")
        return

    if args.download:
        print(f"Downloading {args.n_clips} test clips from Pexels...")
        # Distribute across task types matching validator weights
        task_counts = {}
        for _ in range(args.n_clips):
            r = random.random()
            cumulative = 0.0
            chosen = "HD24K"
            for task, weight in TASK_WEIGHTS_NORMALISED.items():
                cumulative += weight
                if r <= cumulative:
                    chosen = task
                    break
            task_counts[chosen] = task_counts.get(chosen, 0) + 1

        print(f"  Distribution: {task_counts}")
        for task_type, count in task_counts.items():
            for i in range(count):
                print(f"  [{i+1}/{count}] Downloading {task_type} clip...")
                clip = download_pexels_clip(args.pexels_key, task_type, args.duration)
                if clip:
                    print(f"  Saved: {clip.path}")
                else:
                    print(f"  Failed. Try setting PEXELS_API_KEY env var.")
        return

    if args.add_local:
        clip = add_local_clip(args.add_local, args.task_type,
                               float(args.duration))
        print(f"Added: {clip.clip_id}")
        return

    if args.compare:
        compare_models(args.eval_type, args.before_tag, args.after_tag)
        return

    # Default: show usage
    clips = load_clip_registry()
    print(f"SN85 Vidaio Local Test Framework")
    print(f"Clips in buffer: {len(clips)}")
    print()
    print("Usage:")
    print("  # Download test clips (requires Pexels API key)")
    print("  export PEXELS_API_KEY=your_key")
    print("  python replay_buffer/local_test_framework.py --download --n-clips 10")
    print()
    print("  # Add a local clip")
    print("  python replay_buffer/local_test_framework.py --add-local /path/to/clip.mp4 --task-type HD24K")
    print()
    print("  # List clips")
    print("  python replay_buffer/local_test_framework.py --list")
    print()
    print("  # In your evaluation script:")
    print("  from replay_buffer.local_test_framework import (")
    print("      load_clip_registry, evaluate_upscaling, evaluate_compression,")
    print("      save_result, compare_models")
    print("  )")
    print()
    print("  # Compare two model versions:")
    print("  python replay_buffer/local_test_framework.py --compare \\")
    print("    --before-tag yolov8n --after-tag real_esrgan \\")
    print("    --eval-type upscaling")


if __name__ == "__main__":
    main()
