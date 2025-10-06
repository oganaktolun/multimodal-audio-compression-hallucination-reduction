import os, sys
from typing import Any, Tuple

def _die(msg: str):
    raise RuntimeError(
        msg + "\n\n"
        "Setup:\n"
        "  git clone https://github.com/dandelin/PandaGPT.git\n"
        "  cd PandaGPT && pip install -r requirements.txt && pip install -e .\n"
        "  export PANDAGPT_HOME=/abs/path/to/PandaGPT\n"
    )

def load_pandagpt_baseline(device: str = "cuda") -> Tuple[Any, Any, Any]:
    home = os.getenv("PANDAGPT_HOME", "").strip()
    if not home or not os.path.isdir(home):
        _die("PANDAGPT_HOME env var not set or path not found.")
    if home not in sys.path: sys.path.insert(0, home)

    PandaGPT = None; build_video_preprocessor = build_audio_preprocessor = None; last_err = None
    for mod_model_path, mod_proc_path in [
        ("pandagpt.models.pandagpt", "pandagpt.processors.imagebind_processors"),
        ("PandaGPT.models.pandagpt", "PandaGPT.processors.imagebind_processors"),
        ("pandagpt.model", "pandagpt.processors"),
    ]:
        try:
            mod_model = __import__(mod_model_path, fromlist=["PandaGPT"])
            mod_proc  = __import__(mod_proc_path,  fromlist=["build_video_preprocessor","build_audio_preprocessor"])
            PandaGPT = getattr(mod_model, "PandaGPT")
            build_video_preprocessor = getattr(mod_proc, "build_video_preprocessor", None)
            build_audio_preprocessor = getattr(mod_proc, "build_audio_preprocessor", None)
            break
        except Exception as e:
            last_err = e
            continue
    if PandaGPT is None:
        _die(f"Could not import PandaGPT: {last_err}")

    try:
        model = PandaGPT.from_pretrained().to(device).eval()
    except Exception:
        try:
            model = PandaGPT().to(device).eval()
        except Exception as e:
            _die(f"Failed to construct PandaGPT: {e}")

    if callable(build_video_preprocessor) and callable(build_audio_preprocessor):
        try:
            video_pre = build_video_preprocessor(device=device)
            audio_pre = build_audio_preprocessor(device=device)
        except Exception:
            def video_pre(v): return v
            def audio_pre(wav, sr): return wav.unsqueeze(0)
    else:
        def video_pre(v): return v
        def audio_pre(wav, sr): return wav.unsqueeze(0)

    class PandaGPTBaselineAdapter:
        def __init__(self, model, video_key="video", audio_key="audio"):
            self.model = model; self.vk = video_key; self.ak = audio_key
        def generate(self, prompt, video=None, audio=None, **gen_args) -> str:
            import torch
            with torch.inference_mode():
                if hasattr(self.model, "generate"):
                    return self.model.generate(prompt=prompt, **{self.vk: video, self.ak: audio}, **gen_args)
                if hasattr(self.model, "chat"):
                    return self.model.chat(prompt=prompt, **{self.vk: video, self.ak: audio}, **gen_args)
                return self.model(prompt=prompt, **{self.vk: video, self.ak: audio}, **gen_args)

    return PandaGPTBaselineAdapter(model), video_pre, audio_pre
