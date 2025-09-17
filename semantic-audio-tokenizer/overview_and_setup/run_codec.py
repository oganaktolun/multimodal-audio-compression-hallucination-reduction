import os
import tempfile
import argparse

from pathlib import Path

import ffmpeg
import soundfile as sf
from semanticodec import SemantiCodec
from attention_token_allocation import ModifiedSemantiCodec


ORIGINAL_DIR = Path("data/original")
REPLACED_DIR = Path("data/replaced")

# We can use different token rates and vocab sizes see SemantiCodec-inference/test/test_all_settings.py
TOKEN_RATE = 100
VOCAB_SIZE = 16384


def process_audio(input_path: Path,
                  output_path: Path,
                  token_rate: int = TOKEN_RATE,
                  vocab_size: int = VOCAB_SIZE,
                  model: SemantiCodec | ModifiedSemantiCodec):
    """
    Process audio with SemantiCodec
    """
    REPLACED_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        extracted_audio = tmpdir / "extracted.wav"
        processed_audio = tmpdir / "processed.wav"

        # 1) Extract audio for SemantiCodec
        (
            ffmpeg
            .input(str(input_path))
            .output(str(extracted_audio), ac=1, ar=16000)  # mono, 16kHz
            .overwrite_output()
            .run(quiet=True)
        )

        # 2) Run SemantiCodec (encode -> decode)
        semanticodec = model(token_rate=token_rate, semantic_vocab_size=vocab_size)
        tokens = semanticodec.encode(str(extracted_audio))
        waveform = semanticodec.decode(tokens)
        sf.write(str(processed_audio), waveform[0, 0], 16000)
        del semanticodec

        (
            ffmpeg
            .input(str(processed_audio))
            .output(str(output_path), acodec="pcm_s16le")
            .overwrite_output()
            .run(quiet=True)
        )

    print(f"Processed: {input_path.name} -> {output_path}")


def process_video(input_path: Path,
                  output_path: Path,
                  token_rate: int = TOKEN_RATE,
                  vocab_size: int = VOCAB_SIZE,
                  model: SemantiCodec | ModifiedSemantiCodec):
    """
    Extract audio from input_path, process with SemantiCodec, and mux back into the video.
    """
    REPLACED_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        extracted_audio = tmpdir / "extracted.wav"
        processed_audio = tmpdir / "processed.wav"

        # 1) Extract audio for SemantiCodec
        (
            ffmpeg
            .input(str(input_path))
            .output(str(extracted_audio), ac=1, ar=16000)  # mono, 16kHz
            .overwrite_output()
            .run(quiet=True)
        )

        # 2) Run SemantiCodec (encode -> decode)
        semanticodec = model(token_rate=token_rate, semantic_vocab_size=vocab_size)
        tokens = semanticodec.encode(str(extracted_audio))
        waveform = semanticodec.decode(tokens)
        sf.write(str(processed_audio), waveform[0, 0], 16000)
        del semanticodec

        # 3) Mux: original video stream + new audio stream
        video_stream = ffmpeg.input(str(input_path))
        new_audio_stream = ffmpeg.input(str(processed_audio))

        (
            ffmpeg
            # map explicit streams: video from original, audio from processed wav
            .output(
                video_stream.video,
                new_audio_stream.audio,
                str(output_path),
                vcodec="copy",
                acodec="aac",
                shortest=None
            )
            .overwrite_output()
            .run(quiet=True)
        )

    print(f"Processed: {input_path.name} -> {output_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semanticodec", type = str, default = "basic")

    return ap.parse_args()


def main():
    args = parse_args()
    model = SemantiCodec

    if args.semanticodec == "modified":
        model = ModifiedSemantiCodec

    if not ORIGINAL_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {ORIGINAL_DIR.resolve()}")

    REPLACED_DIR.mkdir(parents=True, exist_ok=True)

    mp4s, wavs = [], []
    for p in ORIGINAL_DIR.iterdir():
        match p.suffix.lower():
            case ".mp4":
                mp4s += [p]

            case ".wav":
                wavs += [p]

            case _:
                print(f"not wav/mp4: {p}")

    mp4s = sorted(mp4s)
    wavs = sorted(wavs)

    if not mp4s and not wavs:
        print(f"No .mp4/wav files found in {ORIGINAL_DIR.resolve()}")

        return

    for src in mp4s:
        dst = REPLACED_DIR / src.name
        process_video(src, dst, model)

    for src in wavs:
        dst = REPLACED_DIR / src.name
        process_audio(src, dst, model)

if __name__ == "__main__":
    main()
