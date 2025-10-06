import torch
import torch.nn.functional as F

from tqdm import tqdm

from typing import List, Dict, Tuple
from datatypes import CodecPreset
from AAD import AudioAwareDecoder
from snac import apply_codec_preset

# Helper functions
def _label_to_int(lbl: str) -> int:
    # yes -> 1, no -> 0
    return 1 if str(lbl).strip().lower().startswith("y") else 0

def _pred_from_delta(delta: float, threshold: float = 0.0) -> int:
    # AAD "yes" if margin > 0
    return 1 if delta > threshold else 0

def _f1_no_positive(preds: List[int], labels: List[int]) -> float:
    # F1 where "no" (0) is the positive class, per AAD paper convention
    import numpy as np
    preds = np.array(preds); labels = np.array(labels)
    tp = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 0) & (labels == 1))
    fn = np.sum((preds == 1) & (labels == 0))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    return float(2 * prec * rec / (prec + rec + 1e-9))

def _compute_clf_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    import numpy as np
    preds = np.array(preds); labels = np.array(labels)
    acc = float(np.mean(preds == labels)) if len(labels) else 0.0
    yes_rate = float(np.mean(preds)) if len(preds) else 0.0
    f1_no = _f1_no_positive(preds, labels)
    return {"accuracy": acc, "yes_rate": yes_rate, "f1_no_positive": f1_no}

# ---- evidence for the correct answer (label: yes=1, no=0) ----
def _delta_correct(delta_val: float, label_int: int) -> float:
    # if label is "yes", evidence = Δ; if label is "no", evidence = -Δ
    return delta_val if label_int == 1 else -delta_val

# ---- tune the Δ->yes/no decision threshold on RAW audio ----
def _tune_threshold(deltas_raw: List[float], labels: List[int]) -> float:
    import numpy as np
    # try a modest grid; widen if needed
    grid = np.linspace(-6.0, 6.0, 97)
    best_tau, best_f1 = 0.0, -1.0
    for t in grid:
        preds = [1 if d > t else 0 for d in deltas_raw]
        f1 = _f1_no_positive(preds, labels)
        if f1 > best_f1:
            best_f1, best_tau = f1, t
    return float(best_tau)

# 6. Event Detection

# %%
def detect_eventful_windows(audio: torch.Tensor,
                          window_size: int = 16000*2,  # 2 seconds
                          num_windows: int = 2) -> List[Tuple[int, int]]:
    """Detect eventful windows using energy-based detection"""
    audio = audio.squeeze()
    energy = audio.pow(2)

    # Compute windowed energy
    num_frames = max(1, len(energy) // window_size)
    windowed_energy = []

    for i in range(num_frames):
        start = i * window_size
        end = min(start + window_size, len(energy))
        window_energy = energy[start:end].mean().item()
        windowed_energy.append((window_energy, start, end))

    # Sort by energy and take top windows
    windowed_energy.sort(reverse=True)
    windows = [(start, end) for _, start, end in windowed_energy[:num_windows]]

    return sorted(windows)

AAD_DECISION_THRESHOLD = 0.0  # filled after baseline

def _delta_correct(delta_val: float, label_int: int) -> float:
    # Evidence for the correct answer: yes->Δ, no->-Δ
    return delta_val if label_int == 1 else -delta_val

def _tune_threshold(deltas_raw: List[float], labels: List[int]) -> float:
    import numpy as np
    grid = np.linspace(-6.0, 6.0, 97)
    best_tau, best_f1 = 0.0, -1.0
    for t in grid:
        preds = [1 if d > t else 0 for d in deltas_raw]
        f1 = _f1_no_positive(preds, labels)
        if f1 > best_f1:
            best_f1, best_tau = f1, t
    return float(best_tau)

def evaluate_aad_baseline(audio_samples: List[Dict],
                          aad_decoder: AudioAwareDecoder) -> Dict:
    """Baseline: AAD only, no compression (tunes decision threshold on raw)."""
    results = {
        'preset_name': 'aad_only',
        'bitrate': 0.0,
        'delta_scores': [],          # raw Δ
        'delta_retention': [],       # 0 by definition
        'reconstruction_quality': [0.0],
        'per_question_type': {'present': [], 'random': [], 'adversarial': []},
        'preds': [],
        'labels': [],
        'decision_threshold': 0.0,
    }

    raw_deltas_all, labels_all = [], []

    for sample in tqdm(audio_samples, desc="Evaluating aad_only"):
        audio = sample['waveform']
        silence = torch.zeros_like(audio)
        windows = detect_eventful_windows(audio)
        if not windows:
            continue
        start, end = windows[0]
        a_raw = audio[:, start:end]
        s_win = silence[:, start:end]

        # Use per-sample QA if available; else fall back to toy questions
        questions = sample.get('qa', generate_test_questions())
        for q in questions:
            label = _label_to_int(q['expected'])
            d_raw = aad_decoder.compute_delta(a_raw, s_win, q['question'])

            raw_deltas_all.append(d_raw)
            labels_all.append(label)

            results['delta_scores'].append(d_raw)
            results['per_question_type'].setdefault(q.get('type', 'na'), []).append(d_raw)
            results['delta_retention'].append(0.0)

    # Tune τ* on raw
    tau_star = _tune_threshold(raw_deltas_all, labels_all)
    results['decision_threshold'] = tau_star
    results['preds'] = [1 if d > tau_star else 0 for d in raw_deltas_all]
    results['labels'] = labels_all

    import numpy as np
    results['mean_delta'] = float(np.mean(results['delta_scores'])) if results['delta_scores'] else 0.0
    results['std_delta']  = float(np.std(results['delta_scores']))  if results['delta_scores'] else 0.0
    results['mean_delta_retention'] = 0.0
    results['mean_mse'] = 0.0
    results.update(_compute_clf_metrics(results['preds'], results['labels']))
    return results

# 7. Preset Evaluation

def generate_test_questions() -> List[Dict]:
    """Generate hallucination test questions"""
    return [
        {'question': "Is there speech in this audio?", 'type': 'present', 'expected': 'yes'},
        {'question': "Is there music in this audio?", 'type': 'random', 'expected': 'no'},
        {'question': "Is there a dog barking in this audio?", 'type': 'adversarial', 'expected': 'no'},
    ]

def evaluate_preset(snac_model,
                    device,
                    preset: CodecPreset,
                    audio_samples: List[Dict],
                    aad_decoder: AudioAwareDecoder) -> Dict:
    """Evaluate a codec preset on validation samples."""
    results = {
        'preset_name': preset.name,
        'bitrate': preset.estimated_kbps,
        'delta_scores': [],              # Δ(compressed)
        'delta_retention': [],           # evidence loss: E_raw - E_cmp
        'reconstruction_quality': [],
        'per_question_type': {'present': [], 'random': [], 'adversarial': []},
        'preds': [],
        'labels': []
    }

    tau = AAD_DECISION_THRESHOLD  # set after baseline
    for sample in tqdm(audio_samples, desc=f"Evaluating {preset.name}"):
        audio = sample['waveform']
        compressed = apply_codec_preset(snac_model, device, audio, preset)
        silence = torch.zeros_like(audio)

        windows = detect_eventful_windows(audio)  # windows chosen on raw
        if not windows:
            continue
        start, end = windows[0]
        a_raw = audio[:, start:end]
        a_cmp = compressed[:, start:end]
        s_win = silence[:, start:end]

        questions = sample.get('qa', generate_test_questions())
        for q in questions:
            try:
                label = _label_to_int(q['expected'])
                d_raw = aad_decoder.compute_delta(a_raw, s_win, q['question'])
                d_cmp = aad_decoder.compute_delta(a_cmp, s_win, q['question'])

                # evidence for the correct answer
                e_raw = _delta_correct(d_raw, label)
                e_cmp = _delta_correct(d_cmp, label)

                results['delta_scores'].append(d_cmp)
                results['delta_retention'].append(e_raw - e_cmp)  # evidence loss (lower is better)
                results['per_question_type'].setdefault(q.get('type', 'na'), []).append(d_cmp)

                pred = 1 if d_cmp > tau else 0
                results['preds'].append(pred)
                results['labels'].append(label)
            except Exception as e:
                print(f"Error computing delta: {e}")
                continue

        # MSE sanity
        results['reconstruction_quality'].append(F.mse_loss(compressed, audio).item())

    import numpy as np
    results['mean_delta'] = float(np.mean(results['delta_scores'])) if results['delta_scores'] else 0.0
    results['std_delta']  = float(np.std(results['delta_scores']))  if results['delta_scores'] else 0.0
    results['mean_delta_retention'] = float(np.mean(results['delta_retention'])) if results['delta_retention'] else 0.0
    results['mean_mse']   = float(np.mean(results['reconstruction_quality'])) if results['reconstruction_quality'] else 0.0
    results.update(_compute_clf_metrics(results['preds'], results['labels']))
    return results
