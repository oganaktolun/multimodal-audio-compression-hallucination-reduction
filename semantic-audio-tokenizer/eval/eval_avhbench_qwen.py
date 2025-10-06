#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Features
--------
- Baseline PandaGPT (public repo) audio+video → LLM
- Binary QA metrics: Accuracy / Precision / Recall / F1 (Yes is positive)
- Caption metrics: METEOR (NLTK), CIDEr (pycocoevalcap), per-item & corpus means
- GAVIE judge (OpenAI Responses API, structured JSON), batched + cost/throughput
- Optional pairwise judge (A vs B) for future semantic codec path
- tqdm progress bars, logging, dataclasses for detailed rows
- Writes 4 CSVs + 1 JSON bundle
- ffmpeg audio extraction from MP4s when separate WAV not provided

Prereqs
-------
1) Install PandaGPT (public) and set env var:
   git clone https://github.com/dandelin/PandaGPT.git
   cd PandaGPT && pip install -r requirements.txt && pip install -e .
   export PANDAGPT_HOME=/abs/path/to/PandaGPT

2) Project deps:
   pip install torch torchvision torchaudio numpy pillow tqdm pandas decord opencv-python nltk pycocotools
   pip install git+https://github.com/salaniz/pycocoevalcap.git
   # (for METEOR)
   python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
   # (for GAVIE; optional)
   pip install openai

3) AVHBench JSONs:
   --qa_json : list/dict of yes/no items with fields like {"video": "...mp4", "text": "...?", "label": "Yes/No"}
   --cap_json: list/dict with {"video": "...mp4", "caption": "reference"} or {"references":[...]} etc.

Run
---
python semantic-audio-tokenizer/eval/eval_avhbench_pandagpt.py \
  --qa_json /path/to/avhbench_QA.json \
  --cap_json /path/to/avhbench_captions.json \
  --out_dir semantic-audio-tokenizer/results/avhbench \
  --use_ffmpeg_audio_extract \
  --gavie_model gpt-4o-mini \
  --gavie_batch_size 12 --gavie_concurrency 12 \
  --price_in 0.0005 --price_out 0.0015

Set OPENAI_API_KEY in your environment for GAVIE scoring.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, argparse
from pathlib import Path
from typing import Any, List, Dict
from tqdm import tqdm
import pandas as pd
from dataclasses import asdict
from datetime import datetime
import logging

# ----- path bootstrapping (no package install required)
BASE = Path(__file__).resolve().parent
CORE = BASE / "core_qwen"
ADAP = BASE / "adapters"
for p in (CORE, ADAP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from .core_qwen.registry import MODEL_REGISTRY, set_model_registry
from .core_qwen.datatypes import QARecord, CapRecord
from .core_qwen.qwen_loader import load_qwen_baseline
from .core_qwen.io_utils import group_qa_by_type, load_json_any
from .core_qwen.inference import multimodal_inference_baseline
from .core_qwen.metrics import compute_classification_metrics, compute_caption_overlap_metrics, normalize_yesno
from .core_qwen.persistence import save_eval_outputs
from .adapters.gavie_judge import compute_gavie_scores_openai_batched, compute_gavie_pairwise_openai_batched

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("avhbench.eval")

def parse_args():
    ap = argparse.ArgumentParser(description="AVHBench eval — PandaGPT baseline (modular)")
    ap.add_argument("--qa_json", type=str, required=True)
    ap.add_argument("--cap_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="semantic-audio-tokenizer/results/avhbench")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--frame_size", type=int, default=224)
    ap.add_argument("--audio_sr", type=int, default=16000)
    ap.add_argument("--use_ffmpeg_audio_extract", action="store_true")
    ap.add_argument("--ffmpeg_bin", type=str, default="ffmpeg")
    # GAVIE
    ap.add_argument("--gavie_model", type=str, default="", help="OpenAI model for GAVIE-A (e.g., gpt-4o-mini).")
    ap.add_argument("--gavie_pairwise", action="store_true", help="A/B judge (requires two hypothesis sets).")
    ap.add_argument("--gavie_batch_size", type=int, default=8)
    ap.add_argument("--gavie_concurrency", type=int, default=8)
    ap.add_argument("--price_in", type=float, default=0.0)
    ap.add_argument("--price_out", type=float, default=0.0)
    return ap.parse_args()


def evaluate_qa(tasks: List[dict]) -> tuple[list[bool], list[bool], list[QARecord]]:
    gt, preds, rows = [], [], []
    for idx, item in enumerate(tqdm(tasks, desc="QA inference (baseline)")):
        video_path = item.get("video") or item.get("video_path")
        audio_path = item.get("audio") or item.get("audio_path")
        q = item.get("text") or item.get("question") or ""
        qid = str(item.get("id", f"qa_{idx}"))
        gt_ans = item.get("label") or item.get("answer")
        if isinstance(gt_ans, bool): gt_label = gt_ans
        elif isinstance(gt_ans, str): gt_label = gt_ans.strip().lower() == "yes"
        else: gt_label = bool(gt_ans)
        model = MODEL_REGISTRY["qwen_baseline"]
        out = model.generate(q, video=video_path, audio=audio_path)
        yn = normalize_yesno(out) or (not gt_label)
        gt.append(gt_label); preds.append(yn)
        rows.append(QARecord(
            qid=qid,
            task=item.get("task") or item.get("task_type") or "",
            question=q, gt_yes=gt_label, pred_yes=yn
        ))
    return gt, preds, rows

def evaluate_captions(caps: List[dict]) -> tuple[list[list[str]], list[str], list[str], list[CapRecord]]:
    ref_lists, hyps, qids, rows = [], [], [], []
    for i, item in enumerate(tqdm(caps, desc="Caption generation (baseline)")):
        video_path = item.get("video") or item.get("video_path")
        audio_path = item.get("audio") or item.get("audio_path")
        refs: list[str] = []
        if "caption" in item and isinstance(item["caption"], str):
            refs = [item["caption"]]
        else:
            for k in ("ground_truth", "reference", "references"):
                v = item.get(k)
                if v:
                    refs = v if isinstance(v, list) else [str(v)]
                    break
        if not refs: 
            continue

        model = MODEL_REGISTRY["qwen_baseline"]
        out = model.generate("Describe what you see and hear.", video=video_path, audio=audio_path)
        ref_lists.append([str(r) for r in refs])
        hyp = str(out) if out is not None else ""
        hyps.append(hyp)
        qid = str(item.get("id", f"cap_{i}"))
        qids.append(qid)
        rows.append(CapRecord(qid=qid, ref=refs[0], hyp=hyp))
    return ref_lists, hyps, qids, rows

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    # wire PandaGPT from public repo
    baseline_model, video_pre, audio_pre = load_qwen_baseline(device=MODEL_REGISTRY["device"])
    
    set_model_registry(
        qwen_baseline=baseline_model,
        video_preprocess=video_pre,
        audio_preprocess=audio_pre,
        frame_size=args.frame_size,
        num_frames=args.num_frames,
        audio_sr=args.audio_sr,
        use_ffmpeg_audio_extract=args.use_ffmpeg_audio_extract,
        ffmpeg_bin=args.ffmpeg_bin,
    )
    logger.info("Qwen baseline wired.")

    # load JSONs
    qa_raw  = load_json_any(args.qa_json)
    caps_raw = load_json_any(args.cap_json)
    qa_buckets = group_qa_by_type(qa_raw)

    # QA per task
    task_order = ["Audio-driven Video Hallucination", "Video-driven Audio Hallucination", "Audio-Visual Matching"]
    qa_rows_all: list[QARecord] = []
    qa_metrics_all: Dict[str, Any] = {}
    logger.info("Running QA tasks...")
    for key in task_order + [k for k in qa_buckets if k not in task_order]:
        tasks = qa_buckets.get(key, [])
        if not tasks: continue
        logger.info(f"[Task] {key} (n={len(tasks)})")
        gt, pr, rows = evaluate_qa(tasks)
        acc, prec, rec, f1 = compute_classification_metrics(gt, pr)
        qa_metrics_all[f"{key}/acc"] = acc; qa_metrics_all[f"{key}/prec"] = prec
        qa_metrics_all[f"{key}/rec"] = rec; qa_metrics_all[f"{key}/f1"] = f1
        qa_rows_all.extend(rows)
        logger.info(f"  Acc {acc*100:.1f} | P {prec*100:.1f} | R {rec*100:.1f} | F1 {f1*100:.1f}")

    # captions
    if isinstance(caps_raw, dict) and "captions" in caps_raw: caps_list = caps_raw["captions"]
    elif isinstance(caps_raw, dict) and not isinstance(next(iter(caps_raw.values())), dict): caps_list = list(caps_raw.values())
    elif isinstance(caps_raw, dict): caps_list = [v for v in caps_raw.values()]
    elif isinstance(caps_raw, list): caps_list = caps_raw
    else: caps_list = [caps_raw]

    logger.info("Generating captions...")
    ref_lists, hyps, qids, cap_rows = evaluate_captions(caps_list)

    logger.info("Scoring METEOR/CIDEr...")
    meteor_mean, cider_mean, meteor_item, cider_item = compute_caption_overlap_metrics(ref_lists, hyps)
    qid_to_ix = {qids[i]: i for i in range(len(qids))}
    for r in cap_rows:
        ix = qid_to_ix[r.qid]
        r.meteor = meteor_item[ix]
        r.cider  = cider_item[ix]
    cap_metrics = {"meteor_mean": meteor_mean, "cider_mean": cider_mean, "num_captions": len(cap_rows)}

    # QA macro
    accs  = [v for k,v in qa_metrics_all.items() if k.endswith("/acc")]
    precs = [v for k,v in qa_metrics_all.items() if k.endswith("/prec")]
    recs  = [v for k,v in qa_metrics_all.items() if k.endswith("/rec")]
    f1s   = [v for k,v in qa_metrics_all.items() if k.endswith("/f1")]
    qa_overall = {
        "qa_macro_acc":  sum(accs)/len(accs) if accs else None,
        "qa_macro_prec": sum(precs)/len(precs) if precs else None,
        "qa_macro_rec":  sum(recs)/len(recs) if recs else None,
        "qa_macro_f1":   sum(f1s)/len(f1s) if f1s else None,
        "qa_items": len(qa_rows_all),
        **qa_metrics_all
    }

    # GAVIE (optional)
    gavie_model = args.gavie_model.strip() or None
    if gavie_model and cap_rows:
        refs_for_judge = [r.ref for r in cap_rows]
        hyps_for_judge = [r.hyp for r in cap_rows]
        logger.info("[GAVIE-A] judging captions...")
        gav_avg, gav_list, gav_stats = compute_gavie_scores_openai_batched(
            refs_for_judge, hyps_for_judge,
            model=gavie_model,
            batch_size=args.gavie_batch_size,
            concurrency=args.gavie_concurrency,
            price_in_per_1k=args.price_in,
            price_out_per_1k=args.price_out,
        )
        cap_metrics.update({
            "gavie_a_mean": gav_avg,
            "gavie_a_items": len(gav_list) if gav_list else 0,
            "gavie_tokens_in": gav_stats.get("total_input_tokens", 0.0),
            "gavie_tokens_out": gav_stats.get("total_output_tokens", 0.0),
            "gavie_cost_usd": gav_stats.get("est_cost_usd", 0.0),
            "gavie_items_per_sec": gav_stats.get("items_per_sec", 0.0),
        })

    # persist
    save_eval_outputs(Path(args.out_dir), qa_overall, qa_rows_all, cap_metrics, cap_rows)
    logger.info("Done.")

if __name__ == "__main__":
    main()
