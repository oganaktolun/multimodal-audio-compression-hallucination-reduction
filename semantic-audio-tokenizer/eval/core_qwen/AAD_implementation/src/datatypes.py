from dataclasses import dataclass
from typing import List

@dataclass
class CodecPreset:
    name: str
    codec_type: str
    scales: List[int]
    estimated_kbps: float  # kept for display; we compute actuals per sample
