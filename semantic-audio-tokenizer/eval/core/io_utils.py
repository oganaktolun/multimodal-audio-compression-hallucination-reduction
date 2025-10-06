# semantic-audio-tokenizer/eval/core/io_utils.py
import os, json, subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
import torchaudio
import torch

try:
    import decord
    decord.bridge.set_bridge('torch')
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False

try:
    from torchvision.io import read_video
except Exception:
    read_video = None

from registry import MODEL_REGISTRY

def load_json_any(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w == h: return img.resize((size, size), Image.BICUBIC)
    if w < h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_w, new_h = int(w * size / h), size
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left, top = (new_w - size)//2, (new_h - size)//2
    return img.crop((left, top, left + size, top + size))

def _to_tensor(frames: List[Image.Image], dtype: torch.dtype) -> torch.Tensor:
    arr = np.stack([np.asarray(f).astype(np.float32) / 255.0 for f in frames], axis=0)
    arr = np.transpose(arr, (0,3,1,2))
    t = torch.from_numpy(arr).to(dtype=dtype)
    return t.unsqueeze(0)  # [1,T,C,H,W]

def load_video_frames(video_path: str, num_frames: int, size: int) -> List[Image.Image]:
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    frames: List[Image.Image] = []
    if HAS_DECORD:
        vr = decord.VideoReader(video_path)
        total = len(vr); assert total > 0, f"No frames in {video_path}"
        idx = np.linspace(0, total - 1, num_frames).astype(int)
        batch = vr.get_batch(idx)
        for i in range(batch.shape[0]):
            frames.append(_center_crop(Image.fromarray(batch[i].cpu().numpy()).convert("RGB"), size))
        return frames
    if read_video is None:
        raise RuntimeError("Neither decord nor torchvision.io.read_video available.")
    vid, _, _ = read_video(video_path, pts_unit='sec')
    total = vid.shape[0]; assert total > 0, f"No frames in {video_path}"
    idx = np.linspace(0, total - 1, num_frames).astype(int)
    for i in idx:
        frames.append(_center_crop(Image.fromarray(vid[i].numpy()).convert("RGB"), size))
    return frames

@lru_cache(maxsize=128)
def cached_video_tensor(video_path: str, num_frames: int, size: int, dtype_name: str):
    frames = load_video_frames(video_path, num_frames=num_frames, size=size)
    dtype = getattr(torch, dtype_name)
    return _to_tensor(frames, dtype=dtype)

def ffmpeg_available(bin_name: str) -> bool:
    try:
        subprocess.run([bin_name, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def ffmpeg_extract_audio(video_path: str, out_wav: str, sr: int, ffmpeg_bin: str = "ffmpeg") -> bool:
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_bin, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", out_wav]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def load_audio_waveform(audio_path: str, target_sr: int):
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr); sr = target_sr
    return wav.squeeze(0), sr

@lru_cache(maxsize=256)
def cached_audio_waveform(audio_path: str, target_sr: int):
    return load_audio_waveform(audio_path, target_sr)

def group_qa_by_type(qa_data) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = {}
    def pick_type(item: dict) -> str:
        t = item.get("task") or item.get("task_type") or ""
        if t: return t
        q = (item.get("text") or item.get("question") or "").lower()
        if "visible in the video" in q: return "Audio-driven Video Hallucination"
        if "making sound in the audio" in q: return "Video-driven Audio Hallucination"
        if "matching" in q: return "Audio-Visual Matching"
        return "Unknown"
    if isinstance(qa_data, dict):
        if any(k in qa_data for k in ["Audio-driven Video Hallucination","Video-driven Audio Hallucination","Audio-Visual Matching"]):
            for k, v in qa_data.items(): buckets[k] = v if isinstance(v, list) else [v]
        elif "tasks" in qa_data and isinstance(qa_data["tasks"], list):
            for it in qa_data["tasks"]: buckets.setdefault(pick_type(it), []).append(it)
        else:
            if isinstance(qa_data.get("data"), list):
                for it in qa_data["data"]: buckets.setdefault(pick_type(it), []).append(it)
            else:
                lst = qa_data if isinstance(qa_data, list) else [qa_data]
                for it in lst: buckets.setdefault(pick_type(it), []).append(it)
    elif isinstance(qa_data, list):
        for it in qa_data: buckets.setdefault(pick_type(it), []).append(it)
    else:
        buckets["Unknown"] = []
    return buckets
