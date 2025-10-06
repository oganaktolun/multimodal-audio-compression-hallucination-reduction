from typing import List, Optional, Tuple

try:
    from nltk.translate.meteor_score import meteor_score
except Exception:
    meteor_score = None

def compute_classification_metrics(gt: List[bool], pr: List[bool]) -> Tuple[float, float, float, float]:
    assert len(gt) == len(pr)
    total = len(gt)
    correct = sum(int(g == p) for g, p in zip(gt, pr))
    acc = correct / total if total else 0.0
    TP = sum(1 for g, p in zip(gt, pr) if p and g)
    FP = sum(1 for g, p in zip(gt, pr) if p and not g)
    FN = sum(1 for g, p in zip(gt, pr) if (not p) and g)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec  = TP / (TP + FN) if (TP + FN) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return acc, prec, rec, f1

def compute_caption_overlap_metrics(
    ref_lists: List[List[str]],  # refs per item
    hypotheses: List[str],
) -> Tuple[Optional[float], Optional[float], List[Optional[float]], List[Optional[float]]]:
    n = len(hypotheses)
    # METEOR per item
    meteor_item: List[Optional[float]] = [None] * n
    if meteor_score is not None and n > 0:
        for i, (refs, hyp) in enumerate(zip(ref_lists, hypotheses)):
            try: meteor_item[i] = float(meteor_score(refs, hyp))
            except Exception: meteor_item[i] = None
        meteor_mean = (sum(x for x in meteor_item if x is not None) /
                       max(1, sum(x is not None for x in meteor_item))) if meteor_item else None
    else:
        meteor_mean = None

    # CIDEr per item
    cider_item: List[Optional[float]] = [None] * n
    try:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()
        refs = {i: ref_lists[i] for i in range(n)}
        hyps = {i: [hypotheses[i]] for i in range(n)}
        cider_mean, scores = scorer.compute_score(refs, hyps)
        for i, s in enumerate(scores): cider_item[i] = float(s)
    except Exception:
        cider_mean = None
    return meteor_mean, cider_mean, meteor_item, cider_item

def normalize_yesno(text: str) -> Optional[bool]:
    if text is None: return None
    t = text.strip().lower()
    if t.startswith("yes"): return True
    if t.startswith("no"):  return False
    return None
