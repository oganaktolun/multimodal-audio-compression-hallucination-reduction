from typing import List, Optional, Tuple

class _EncoderBase:
    def encode(self, audio_path: str, tps: int = 50, stream: str = "semantic") -> List[int]:
        raise NotImplementedError

class SemCodecEncoderPlaceholder(_EncoderBase):
    def encode(self, audio_path: str, tps: int = 50, stream: str = "semantic") -> List[int]:
        raise NotImplementedError("SemantiCodec encoder not wired yet. Provide your encode() function here.")

def build_semcodec(*args, **kwargs) -> Tuple[_EncoderBase, None, None, None, str]:
    """
    Returns (encoder, token_embedder, token_aggregator, audio_projector, feed_mode)
    Kept for future Option-B. Currently just raises when used.
    """
    return SemCodecEncoderPlaceholder(), None, None, None, "ids"
