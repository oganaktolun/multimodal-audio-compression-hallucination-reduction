import os
from typing import Optional, Dict, Any
import torch
from registry import MODEL_REGISTRY
from io_utils import (
    cached_video_tensor, cached_audio_waveform,
    ffmpeg_available, ffmpeg_extract_audio
)

@torch.inference_mode()
def multimodal_inference_baseline(
    audio_path: Optional[str],
    video_path: Optional[str],
    prompt: str,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    device = MODEL_REGISTRY["device"]
    dtype  = MODEL_REGISTRY["dtype"]
    size   = int(MODEL_REGISTRY["frame_size"])
    num_fr = int(MODEL_REGISTRY["num_frames"])
    sr     = int(MODEL_REGISTRY["audio_sr"])
    use_ff = bool(MODEL_REGISTRY["use_ffmpeg_audio_extract"])
    ffbin  = MODEL_REGISTRY["ffmpeg_bin"]

    gen_args = dict(max_new_tokens=64, temperature=0.0, top_p=1.0, do_sample=False)
    if generation_kwargs: gen_args.update(generation_kwargs)

    vid_feats = None
    if video_path:
        vtensor = cached_video_tensor(video_path, num_fr, size, dtype_name=str(dtype).split('.')[-1]).to(device)
        vp = MODEL_REGISTRY["video_preprocess"]
        vid_feats = vp(vtensor) if callable(vp) else vtensor

    audio_feats = None
    if audio_path is None and video_path and use_ff and ffmpeg_available(ffbin):
        tmp_wav = os.path.join(".avhbench_tmp_audio", os.path.basename(video_path) + ".wav")
        if ffmpeg_extract_audio(video_path, tmp_wav, sr, ffbin):
            audio_path = tmp_wav
    if audio_path and os.path.exists(audio_path):
        wav, _ = cached_audio_waveform(audio_path, sr)
        wav = wav.to(device=device, dtype=torch.float32)
        ap = MODEL_REGISTRY["audio_preprocess"]
        audio_feats = ap(wav, sr) if callable(ap) else wav.unsqueeze(0)
        if isinstance(audio_feats, torch.Tensor): audio_feats = audio_feats.to(device)

    out_text = MODEL_REGISTRY["qwen_baseline"].generate(
        prompt=prompt, video=vid_feats, audio=audio_feats, **gen_args
    )
    return str(out_text).strip()
