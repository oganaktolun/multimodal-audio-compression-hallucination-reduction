from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict
from datetime import datetime
import json
import pandas as pd
from datatypes import QARecord, CapRecord

def save_eval_outputs(
    out_dir: Path,
    qa_metrics: Dict[str, Any],
    qa_rows: List[QARecord],
    cap_metrics: Dict[str, Any],
    cap_rows: List[CapRecord],
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    pd.DataFrame([qa_metrics]).to_csv(out_dir / f"binary_metrics_{ts}.csv", index=False)
    if qa_rows:
        pd.DataFrame([asdict(r) for r in qa_rows]).to_csv(out_dir / f"binary_detailed_{ts}.csv", index=False)

    pd.DataFrame([cap_metrics]).to_csv(out_dir / f"caption_metrics_{ts}.csv", index=False)
    if cap_rows:
        pd.DataFrame([asdict(r) for r in cap_rows]).to_csv(out_dir / f"caption_detailed_{ts}.csv", index=False)

    bundle = {
        "timestamp": datetime.now().isoformat(),
        "binary_evaluation": qa_metrics,
        "caption_evaluation": cap_metrics,
        "binary": [asdict(r) for r in qa_rows],
        "captions": [asdict(r) for r in cap_rows],
    }
    with open(out_dir / f"eval_results_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
