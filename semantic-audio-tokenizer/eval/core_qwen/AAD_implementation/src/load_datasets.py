# ======================================================================
# 3) Data loading (Clotho-AQA) + utilities
# ======================================================================

from datasets import load_dataset, concatenate_datasets, Audio as HF_Audio
import soundfile as sf
from huggingface_hub import snapshot_download
import ffmpeg
import torch
import torchaudio
import torch.nn.functional as F
from typing import Dict, List
import io
import os
from pathlib import Path

def _mono16(waveform: torch.Tensor, sr: int, duration: float | None = None) -> torch.Tensor:
    # waveform: [C,T] or [T] -> [1,T]@16k, crop/pad to duration
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if sr != 16_000:
        waveform = torchaudio.transforms.Resample(sr, 16_000)(waveform)
    if duration is not None:
        T = int(16_000 * duration)
        if waveform.shape[1] > T:
            waveform = waveform[:, :T]
        elif waveform.shape[1] < T:
            waveform = F.pad(waveform, (0, T - waveform.shape[1]))
    return waveform

def _open_hf_audio(audio_dict: Dict[str, object]) -> tuple[torch.Tensor, int]:
    # Prefer bytes, then absolute path, then snapshot-resolve
    p = audio_dict.get("path")
    b = audio_dict.get("bytes")
    if b is not None:
        data, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
        return wav, int(sr)
    if isinstance(p, str) and os.path.isabs(p) and os.path.exists(p):
        wav, sr = torchaudio.load(p)
        return wav, sr
    if isinstance(p, str):
        repo_root = snapshot_download("lmms-lab/ClothoAQA", repo_type="dataset")
        candidates = [
            Path(repo_root) / "audio_files" / p,
            Path(repo_root) / "clotho_aqa" / "audio_files" / p,
            Path(repo_root) / p,
        ]
        for cp in candidates:
            if cp.exists():
                wav, sr = torchaudio.load(str(cp))
                return wav, sr
    raise FileNotFoundError(f"Could not resolve audio: path={p!r} bytes={'present' if b is not None else 'absent'}")

def load_clotho_aqa_samples(max_items: int = 48, duration: float = 10.0) -> List[Dict]:
    ds = load_dataset("lmms-lab/ClothoAQA", "clotho_aqa")
    split = concatenate_datasets([ds["clotho_aqa_val_filtered"], ds["clotho_aqa_test_filtered"]])
    split = split.cast_column("audio", HF_Audio(decode=False))
    samples: List[Dict] = []
    for ex in split.select(range(min(max_items, len(split)))):
        try:
            wav, sr = _open_hf_audio(ex["audio"])
        except Exception as e:
            print(f"[skip] {ex['audio'].get('path')}: {e}")
            continue
        wav = _mono16(wav, sr, duration)
        samples.append({
            "waveform": wav,
            "path": ex["audio"].get("path", f"clotho:{len(samples)}"),
            "duration": duration,
            "qa": [{
                "question": ex["question"],
                "expected": ex["answer"],  # 'yes'|'no'
                "type": "aqa"
            }]
        })
    return samples

audio_samples = load_clotho_aqa_samples(max_items=48, duration=10.0)
print(f"Loaded {len(audio_samples)} Clotho-AQA samples")
