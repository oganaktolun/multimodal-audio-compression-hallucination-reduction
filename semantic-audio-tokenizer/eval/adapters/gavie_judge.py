from typing import List, Optional, Tuple, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

# ---------- System prompts & schemas ----------
def _make_gavie_accuracy_system_prompt() -> str:
    return (
        "You are an expert judge for Audio-Visual caption accuracy (GAVIE-A). "
        "Given a ground-truth audio+visual caption (canonical) and a candidate caption, "
        "score ONLY factual ACCURACY on a 0–10 scale:\n"
        "10 = fully faithful: all salient objects/actions/sounds/relations in ground-truth are preserved; "
        "no invented entities/sounds/relations; no contradictions; minor paraphrases allowed.\n"
        "8  = mostly accurate: no clear hallucinations; small omissions of non-core details.\n"
        "5  = mixed: some correct, but ≥1 significant inaccuracy (hallucination, wrong relation) or multiple important omissions.\n"
        "2  = largely inaccurate: several hallucinations/contradictions or core scene wrong.\n"
        "0  = unrelated or invented.\n\n"
        "Definitions (strict):\n"
        "- Hallucination: candidate asserts an object/sound/event/attribute/temporal relation NOT entailed by the ground-truth.\n"
        "- Omission: candidate misses a CORE element present in ground-truth (object/sound/action defining the scene). Penalize.\n"
        "- Paraphrase: wording differs but meaning matches. Do NOT penalize.\n"
        "- Uncertainty words: ignore style; judge by factual content.\n"
        "Judge ONLY by entailment from ground-truth; do not use outside knowledge."
    )

def _gavie_accuracy_json_schema():
    return {
        "name": "gavie_score",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "accuracy": {"type": "number", "minimum": 0, "maximum": 10},
                "hallucinated": {"type": "array", "items": {"type": "string"}},
                "omitted_core": {"type": "array", "items": {"type": "string"}},
                "explanation": {"type": "string"}
            },
            "required": ["accuracy"]
        },
    }

def _make_pairwise_system_prompt() -> str:
    return (
        "You are a strict AV caption accuracy judge. "
        "Given a ground-truth caption and TWO candidate captions (A and B), "
        "select which candidate is MORE FACTUALLY ACCURATE relative to the ground-truth. "
        "Consider hallucinated items, contradictions, and omissions of core content. "
        "Output JSON: winner (\"A\"|\"B\"|\"tie\"), confidence (0–1), explanation."
    )

def _pairwise_json_schema():
    return {
        "name": "pairwise_judgment",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "winner": {"type": "string", "enum": ["A", "B", "tie"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "explanation": {"type": "string"}
            },
            "required": ["winner"]
        },
    }

# ---------- OpenAI client ----------
def _client(api_key: Optional[str]):
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. `pip install openai`") from e
    return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

# ---------- Single-call helpers ----------
def _call_openai_gavie(client, model, sys_msg, schema, ref, hyp, timeout=60):
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content":
                "Instruction: Describe what you see and hear.\n\n"
                "Ground-truth audio-visual caption (canonical):\n"
                f"\"{ref}\"\n\n"
                "Candidate caption to score for factual ACCURACY only:\n"
                f"\"{hyp}\"\n\n"
                "Return JSON with fields: accuracy (0–10), hallucinated (list), omitted_core (list), explanation."
            },
        ],
        response_format={"type": "json_schema", "json_schema": schema},
        timeout=timeout,
    )
    data = json.loads(resp.output_text)
    acc = float(data.get("accuracy", 0.0))
    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
    out_tok = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
    return max(0.0, min(10.0, acc)), in_tok, out_tok

def _call_openai_pairwise(client, model, sys_msg, schema, ref, A, B, timeout=60):
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content":
                "Ground-truth caption (canonical):\n"
                f"\"{ref}\"\n\n"
                "Candidate A:\n"
                f"\"{A}\"\n\n"
                "Candidate B:\n"
                f"\"{B}\"\n\n"
                "Return JSON: {\"winner\":\"A\"|\"B\"|\"tie\", \"confidence\":0-1, \"explanation\":\"...\"}"
            },
        ],
        response_format={"type": "json_schema", "json_schema": schema},
        timeout=timeout,
    )
    data = json.loads(resp.output_text)
    winner = data.get("winner", "tie")
    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
    out_tok = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
    return winner, in_tok, out_tok

# ---------- Public APIs (batched) ----------
def compute_gavie_scores_openai_batched(
    references: List[str],
    hypotheses: List[str],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    timeout: int = 60,
    batch_size: int = 8,
    concurrency: int = 8,
    price_in_per_1k: float = 0.0,
    price_out_per_1k: float = 0.0,
) -> Tuple[Optional[float], Optional[List[float]], Dict[str, float]]:
    """
    Returns (avg_gavie_a, per_item_scores, stats_dict)
    stats_dict: total_items, total_input_tokens, total_output_tokens, est_cost_usd, wall_time_sec, items_per_sec
    """
    if not references or not hypotheses:
        return None, None, {}
    if len(references) != len(hypotheses):
        raise ValueError("refs and hyps must have same length")

    client = _client(api_key)
    sys_msg = _make_gavie_accuracy_system_prompt()
    schema  = _gavie_accuracy_json_schema()

    n = len(references)
    scores: List[Optional[float]] = [None] * n
    total_in = total_out = 0
    t0 = time.time()

    def worker(i: int):
        try:
            s, ti, to = _call_openai_gavie(client, model, sys_msg, schema, references[i], hypotheses[i], timeout)
            return i, s, ti, to, None
        except Exception as e:
            return i, 0.0, 0, 0, e

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for fut in as_completed([ex.submit(worker, i) for i in range(start, end)]):
                i, s, ti, to, err = fut.result()
                scores[i] = s
                total_in += ti
                total_out += to

    wall = max(1e-6, time.time() - t0)
    done = sum(1 for s in scores if s is not None)
    avg  = (sum(float(s) for s in scores if s is not None) / done) if done else None
    est_cost = (total_in / 1000.0) * price_in_per_1k + (total_out / 1000.0) * price_out_per_1k

    stats = {
        "total_items": float(done),
        "total_input_tokens": float(total_in),
        "total_output_tokens": float(total_out),
        "est_cost_usd": float(est_cost),
        "wall_time_sec": float(wall),
        "items_per_sec": float(done / wall) if wall > 0 else 0.0,
    }
    return avg, [float(s) for s in scores], stats

def compute_gavie_pairwise_openai_batched(
    references: List[str],
    hyps_a: List[str],
    hyps_b: List[str],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    timeout: int = 60,
    batch_size: int = 8,
    concurrency: int = 8,
    price_in_per_1k: float = 0.0,
    price_out_per_1k: float = 0.0,
) -> Tuple[int, int, int, Dict[str, float]]:
    """
    Returns (wins_A, wins_B, ties, stats_dict)
    """
    if not (len(references) == len(hyps_a) == len(hyps_b)):
        raise ValueError("refs/A/B must have same length")

    client = _client(api_key)
    sys_msg = _make_pairwise_system_prompt()
    schema  = _pairwise_json_schema()

    n = len(references)
    wins_A = wins_B = ties = 0
    total_in = total_out = 0
    t0 = time.time()

    def worker(i: int):
        try:
            w, ti, to = _call_openai_pairwise(client, model, sys_msg, schema, references[i], hyps_a[i], hyps_b[i], timeout)
            return i, w, ti, to, None
        except Exception as e:
            return i, "tie", 0, 0, e

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for fut in as_completed([ex.submit(worker, i) for i in range(start, end)]):
                i, w, ti, to, err = fut.result()
                if w == "A": wins_A += 1
                elif w == "B": wins_B += 1
                else: ties += 1
                total_in += ti
                total_out += to

    wall = max(1e-6, time.time() - t0)
    est_cost = (total_in / 1000.0) * price_in_per_1k + (total_out / 1000.0) * price_out_per_1k

    stats = {
        "total_items": float(n),
        "total_input_tokens": float(total_in),
        "total_output_tokens": float(total_out),
        "est_cost_usd": float(est_cost),
        "wall_time_sec": float(wall),
        "items_per_sec": float(n / wall) if wall > 0 else 0.0,
    }
    return wins_A, wins_B, ties, stats
