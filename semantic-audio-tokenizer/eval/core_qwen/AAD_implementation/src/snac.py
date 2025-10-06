# ## 4. SNAC Codec Configuration

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np

from typing import List, Tuple

from datatypes import CodecPreset

def _infer_codes_nq_T(code: torch.Tensor) -> tuple[int, int]:
    """
    Infer (#codebooks, T_frames) from a single SNAC code tensor with batch dim.
    Handles shapes like [B, Q, T] or [B, T, Q] or [B, T] (Q=1).
    """
    assert code.dim() >= 2, f"Unexpected code dim: {code.shape}"
    dims = list(code.shape)
    B = dims.pop(0)
    if len(dims) == 1:
        # [B, T]
        return 1, int(dims[0])
    elif len(dims) == 2:
        d1, d2 = dims
        # Heuristic: T_frames is the larger among d1,d2
        if d1 >= d2:
            return int(d2), int(d1)  # [B, T, Q] -> (Q,T)
        else:
            return int(d1), int(d2)  # [B, Q, T] -> (Q,T)
    else:
        # Fallback: treat last as T, multiply others as Q
        T = int(dims[-1])
        Q = int(np.prod(dims[:-1]))
        return Q, T

def _bits_per_symbol_from_codes(code: torch.Tensor) -> int:
    # Estimate bits/symbol from max index (ceil log2(max+1)), avoid zero -> 1
    maxv = int(code.max().item()) if code.numel() > 0 else 0
    return max(1, int(np.ceil(np.log2(maxv + 1))))

def _estimate_bitrate_snac(codes_list: List[torch.Tensor], keep_indices: set, dur_sec: float) -> float:
    """
    Estimate kbps for the kept scales only.
    kbps = sum_scales (frames/sec * n_q * bits_per_symbol) / 1000
    """
    if dur_sec <= 0:
        return 0.0
    kbps = 0.0
    for i, code in enumerate(codes_list):
        if i not in keep_indices:
            continue
        c = code.detach().cpu()
        n_q, T_frames = _infer_codes_nq_T(c)
        fps = T_frames / dur_sec
        bpsym = _bits_per_symbol_from_codes(c)
        kbps += (fps * n_q * bpsym) / 1000.0
    return float(kbps)

def apply_codec_preset_with_bitrate(snac_model, device, audio: torch.Tensor, preset: CodecPreset) -> Tuple[torch.Tensor, float]:
    """
    Apply SNAC preset and also estimate bitrate from codes.
    Returns: (reconstruction [1,T]@16k, measured_kbps)
    """
    assert preset.codec_type == "snac", "Only SNAC supported in this notebook"
    with torch.no_grad():
        x16 = audio.detach().float().cpu()
        if x16.dim() == 1:
            x16 = x16.unsqueeze(0)
        elif x16.shape[0] > 1:
            x16 = x16.mean(dim=0, keepdim=True)

        snac_sr = 24000
        if snac_sr != 16000:
            xsn_1xT = torchaudio.functional.resample(x16.squeeze(0), 16000, snac_sr).unsqueeze(0)
        else:
            xsn_1xT = x16

        xsn = xsn_1xT.unsqueeze(1).to(device)  # [1,1,Tsn]
        codes = snac_model.encode(xsn)         # list of code tensors per scale

        rate_to_index = {14: 0, 29: 1, 57: 2, 115: 3}
        keep_indices = {rate_to_index[r] for r in preset.scales if r in rate_to_index}

        modified_codes = []
        for i, code in enumerate(codes):
            if i in keep_indices:
                modified_codes.append(code)
            else:
                modified_codes.append(torch.zeros_like(code))

        ysn = snac_model.decode(modified_codes).detach().cpu()  # [1,1,Tsn]
        ysn_1xT = ysn.squeeze(1)

        if snac_sr != 16000:
            y16 = torchaudio.functional.resample(ysn_1xT.squeeze(0), snac_sr, 16000).unsqueeze(0)
        else:
            y16 = ysn_1xT

        # strict length match
        T = x16.shape[1]
        if y16.shape[1] > T:
            y16 = y16[:, :T]
        elif y16.shape[1] < T:
            y16 = F.pad(y16, (0, T - y16.shape[1]))

        # bitrate estimation for kept scales only
        dur_sec = T / 16000.0
        measured_kbps = _estimate_bitrate_snac(codes, keep_indices, dur_sec)

    return y16, measured_kbps

def apply_codec_preset(snac_model, device, audio: torch.Tensor, preset: CodecPreset) -> torch.Tensor:
    """
    Original helper retained for compatibility: returns only reconstructed audio [1,T]@16k.
    """
    y16, _ = apply_codec_preset_with_bitrate(snac_model, device, audio, preset)
    return y16
