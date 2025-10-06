import torch
import pandas as pd
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

from src.load_datasets import load_clotho_aqa_samples
from src.AAD import AudioAwareDecoder
from src.evaluate import evaluate_aad_baseline, evaluate_preset
from src.datatypes import CodecPreset
from src.pareto import find_pareto_optimal

def main():
    # Force GPU usage
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available! Enable GPU in Runtime > Change runtime type")

    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Load Qwen2-Audio with Proper Quantization

    # %%
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    # Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16
    )

    print("Loading Qwen2-Audio-7B-Instruct with 8-bit quantization...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()
    print("Model loaded successfully!")

    from transformers import GenerationConfig
    gc = model.generation_config
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    model.generation_config = gc

    audio_samples = load_clotho_aqa_samples(max_items=48, duration=10.0)
    print(f"Loaded {len(audio_samples)} Clotho-AQA samples")

    aad_decoder = AudioAwareDecoder(model, processor, device, alpha=1.0)

    PRESET_GRID = [
        CodecPreset("snac_coarse", "snac", [14], 0.5),
        CodecPreset("snac_medium", "snac", [14, 29], 1.0),
        CodecPreset("snac_fine", "snac", [14, 29, 57], 1.5),
    ]

    import snac
    snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
    snac_model.eval()

    # 8. Run Grid Search

    evaluation_results = []

    # 1) Baseline (tunes τ*)
    baseline = evaluate_aad_baseline(audio_samples, aad_decoder)
    evaluation_results.append(baseline)
    AAD_DECISION_THRESHOLD = baseline.get('decision_threshold', 0.0)
    print(f"Using tuned decision threshold τ* = {AAD_DECISION_THRESHOLD:.3f}")

    # 2) Presets
    for preset in PRESET_GRID:
        print(f"\nEvaluating preset: {preset.name}")
        res = evaluate_preset(snac_model, device, preset, audio_samples, aad_decoder)
        evaluation_results.append(res)
        print(f"  Evidence loss (Δ-retention): {res['mean_delta_retention']:.3f} | "
              f"F1(no+): {res['f1_no_positive']:.3f} | Acc: {res['accuracy']:.3f} | "
              f"Yes-rate: {res['yes_rate']:.3f} | MSE: {res['mean_mse']:.4f}")

    pareto_presets = find_pareto_optimal(evaluation_results)
    print(f"\nFound {len(pareto_presets)} Pareto-optimal presets:")
    for preset in pareto_presets:
        print(f"  - {preset['preset_name']}: {preset['bitrate']} kbps, Δ={preset['mean_delta']:.3f}")

    # ## 10. Visualization

    # %%
    df_results = pd.DataFrame(evaluation_results)

    # Figure A: Rate–Evidence (Δ-retention) and Quality
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Rate vs Δ-retention (lower is better)
    axes[0].scatter(df_results['bitrate'], df_results['mean_delta_retention'], s=100, alpha=0.8)
    for idx, row in df_results.iterrows():
        axes[0].annotate(row['preset_name'],
                         (row['bitrate'], row['mean_delta_retention']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0].set_xlabel('Bitrate (kbps)')
    axes[0].set_ylabel('Δ-retention (Δ(raw) − Δ(compressed))')
    axes[0].set_title('Rate–Evidence (lower is better)')
    axes[0].grid(True, alpha=0.3)

    # Quality vs Bitrate (MSE)
    axes[1].scatter(df_results['bitrate'], df_results['mean_mse'], s=100, alpha=0.8)
    for idx, row in df_results.iterrows():
        axes[1].annotate(row['preset_name'],
                         (row['bitrate'], row['mean_mse']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1].set_xlabel('Bitrate (kbps)')
    axes[1].set_ylabel('Reconstruction Error (MSE)')
    axes[1].set_title('Quality vs Bitrate')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Figure B: Hallucination metrics
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for metric in ['f1_no_positive', 'accuracy', 'yes_rate']:
        ax.plot(df_results['bitrate'], df_results[metric], marker='o', label=metric)
    for idx, row in df_results.iterrows():
        ax.annotate(row['preset_name'], (row['bitrate'], row['accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Bitrate (kbps)')
    ax.set_ylabel('Score')
    ax.set_title('Hallucination Metrics vs Bitrate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

    # ## 11. Save Results

    # %%
    discovered_presets = {
        'pareto_optimal': pareto_presets,
        'all_evaluated': evaluation_results,
        'config': {
            'num_samples': len(audio_samples),
            'alpha': aad_decoder.alpha,
            'model': "Qwen2-Audio-7B-Instruct"
        }
    }

    with open('discovered_presets.json', 'w') as f:
        json.dump(discovered_presets, f, indent=2, default=str)

    print("\nPreset discovery complete!")
    print(f"Results saved to discovered_presets.json")

if __name__ == "__main__":
    main()
