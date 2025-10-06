# semantic-audio-tokenizer/eval/core/registry.py
from typing import Any, Dict
import torch

MODEL_REGISTRY: Dict[str, Any] = {
    "qwengpt_baseline": None,
    "video_preprocess": None,
    "audio_preprocess": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    "frame_size": 224,
    "num_frames": 8,
    "audio_sr": 16000,
    "use_ffmpeg_audio_extract": False,
    "ffmpeg_bin": "ffmpeg",
}

def set_model_registry(**kwargs):
    MODEL_REGISTRY.update(kwargs)
