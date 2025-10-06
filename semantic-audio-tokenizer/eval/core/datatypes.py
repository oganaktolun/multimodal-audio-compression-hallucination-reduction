# semantic-audio-tokenizer/eval/core/datatypes.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class QARecord:
    qid: str
    task: str
    question: str
    gt_yes: bool
    pred_yes: bool

@dataclass
class CapRecord:
    qid: str
    ref: str
    hyp: str
    meteor: Optional[float] = None
    cider: Optional[float] = None
