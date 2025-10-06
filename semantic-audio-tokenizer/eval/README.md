# AVHBench Evaluation (PandaGPT Baseline)

This folder contains a **baseline-only** AVHBench evaluation using the public **PandaGPT**.  

## Setup

1. **Install PandaGPT (public)**
   ```bash
   git clone https://github.com/dandelin/PandaGPT.git
   cd PandaGPT
   pip install -r requirements.txt
   pip install -e .

Then export its path:
export PANDAGPT_HOME=/abs/path/to/PandaGPT


2. Install eval dependencies (from your project root)
pip install -r semantic-audio-tokenizer/requirements_avhbench.txt


3. Prepare AVHBench JSONs:
    avhbench_QA.json – list/dict of yes/no items with fields like:
        {"video": "path/to/video.mp4", "text": "Is the dog visible in the video?", "label": "No"}
    avhbench_captions.json – list/dict of caption items with fields like:
        {"video": "path/to/video.mp4", "caption": "A person walks inside while distant traffic noise is heard."}


4. If dataset only has embedded audio in mp4
Run with --use_ffmpeg_audio_extract to auto-extract mono WAV per video using ffmpeg.

EXAMPLE Run:
    export OPENAI_API_KEY=sk-your-key-here


(from the parent directory, workspace folder is the parent of eval/)

    python -m eval.eval_avhbench_pandagpt \
    --qa_json /path/to/avhbench_QA.json \
    --cap_json /path/to/avhbench_captions.json \
    --out_dir semantic-audio-tokenizer/results/avhbench \
    --use_ffmpeg_audio_extract \
    --gavie_model gpt-4o-mini \
    --gavie_batch_size 12 --gavie_concurrency 12 \
    --price_in 0.0005 --price_out 0.0015

        CLI flags (core):
        --qa_json / --cap_json : paths to your JSON files
        --out_dir : where to save CSV/JSON results (default shown above)
        --num_frames / --frame_size : video sampling config (default 8×224)
        --audio_sr : audio resample rate (default 16k)
        --use_ffmpeg_audio_extract : auto-extract WAV from MP4 via ffmpeg
        --ffmpeg_bin : ffmpeg binary name/path (default ffmpeg)

5. Output:
    The script writes four CSVs and one JSON bundle into --out_dir:
        binary_metrics_<ts>.csv — macro QA metrics + per-task (Audio-driven / Video-driven / AV-Matching)
        binary_detailed_<ts>.csv — per-item rows: qid, task, question, gt_yes, pred_yes
        caption_metrics_<ts>.csv — corpus METEOR/CIDEr (and GAVIE if enabled)
        caption_detailed_<ts>.csv — per-item rows: qid, ref, hyp, per-item METEOR, per-item CIDEr
        eval_results_<ts>.json — one JSON with all of the above, including detailed rows

6. GAVIE (OpenAI Judge) — Accuracy Scoring for Captions:

    This project supports GAVIE-A (accuracy-only) caption scoring using the OpenAI Responses API with structured JSON outputs.

    The judge compares each generated caption to a ground-truth caption and returns a 0–10 accuracy score:
        10 = fully faithful (no hallucinated objects/sounds/relations; no contradictions; core content preserved)
        0 = unrelated or invented
        The judge penalizes hallucinated entities and major omissions, while not penalizing harmless paraphrasing.

    Enable GAVIE:

    a. Install OpenAI client (already in requirements_avhbench.txt).
    b. Set your API key (recommended via env var):
        export OPENAI_API_KEY=sk-...  # your key
    c. Run with the judge enabled:
        python semantic-audio-tokenizer/eval/eval_avhbench_pandagpt.py \
        --qa_json /path/to/avhbench_QA.json \
        --cap_json /path/to/avhbench_captions.json \
        --out_dir semantic-audio-tokenizer/results/avhbench \
        --use_ffmpeg_audio_extract \
        --gavie_model gpt-4o-mini
        Useful flags:
        --gavie_model : OpenAI model to use (e.g., gpt-4o-mini); omit to skip GAVIE
        --gavie_batch_size : items per wave (default 8)
        --gavie_concurrency : parallel requests per wave (default 8)
        --price_in / --price_out : USD per 1K tokens (input/output) to report an estimated cost (leave 0 to skip)
    Example (with cost & throughput):
        python ... \
        --gavie_model gpt-4o-mini \
        --gavie_batch_size 12 \
        --gavie_concurrency 12 \
        --price_in 0.0005 \
        --price_out 0.0015

    What gets saved (extra fields in caption_metrics_<ts>.csv and JSON):
        gavie_a_mean — mean accuracy (0–10)
        gavie_a_items — number of scored captions
        gavie_tokens_in / gavie_tokens_out — token usage (sum)
        gavie_cost_usd — estimated cost (if prices provided)
        gavie_items_per_sec — judge throughput
    By default we judge against the first reference per item. If you want best-of-refs (max score over multiple references) or aggregate scoring, open an issue and we’ll add a switch.

    Notes & tips:
        If you hit rate limits (429), lower --gavie_batch_size and/or --gavie_concurrency.
        The judge only sees text (reference + hypothesis); it does not see your media.
        Low GAVIE with decent METEOR/CIDEr often means fluent but ungrounded captions.


7. Wiring Notes:
    The loader in adapters/pandagpt_adapter.py imports PandaGPT from PANDAGPT_HOME.
    If public fork uses different module paths or .generate(...) kwargs, adjust the adapter only, not the eval script.

    When ready to test SemantiCodec tokens:
        Implement encode(audio_path, tps, stream) in adapters/semcodec_adapter.py
        Add a PandaGPTSemanticAdapter (mirroring the baseline one) and a second model wiring

    Reuse the existing evaluation loops.

    To enable GAVIE (GPT judge) later:
        Replace compute_gavie_scores_placeholder in adapters/gavie_judge.py with OpenAI call


Tips:

    If you hit ImportError for PandaGPT, confirm:
        PANDAGPT_HOME points to the repo root
        pip install -e $PANDAGPT_HOME succeeded

    If you lack ffmpeg, install it (macOS: brew install ffmpeg, Ubuntu: apt-get install ffmpeg).

    METEOR is None — make sure nltk is installed and you downloaded wordnet and omw-1.4.
    
    CIDEr is None — ensure pycocoevalcap and pycocotools are installed.
    
    OpenAI errors — check OPENAI_API_KEY, reduce concurrency/batch size on 429s.`

