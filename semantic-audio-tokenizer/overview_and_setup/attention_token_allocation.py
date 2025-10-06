import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
from scipy import signal
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")

from pesq import pesq
from pystoi import stoi

#pip install pesq, pystoi

class ComplexityAnalyzer:
#analyze audio complexity
    def __init__(self, sr=16000):
        self.sr = sr
        
    def analyze_audio(self, audio_path, target_frames):
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)        
        hop_length = max(64, min(512, int(len(audio) / target_frames)))        
        complexity_scores = []
        
        for i in range(target_frames):
            start_sample = i * hop_length
            end_sample = min(start_sample + hop_length * 2, len(audio))
            frame = audio[start_sample:end_sample]
            if len(frame) == 0:
                complexity_scores.append(0.0)
                continue
            spectral_complexity = self.spectral_complexity(frame)
            temporal_complexity = self.temporal_complexity(frame)
            entropy_complexity = self.spectral_entropy(frame)
            
            # Weighted combination - weights need empirical validation
            combined_complexity = (
                0.4 * spectral_complexity +    
                0.3 * temporal_complexity +       
                0.3 * entropy_complexity        
            )
            
            complexity_scores.append(combined_complexity)
        
        return np.array(complexity_scores)
    
    def spectral_complexity(self, frame):
        if len(frame) < 64:
            return 0.0
            
        try:
            centroids = librosa.feature.spectral_centroid(
                y=frame, sr=self.sr, hop_length=64
            )[0]
            if len(centroids) > 1:
                complexity = np.std(centroids) / (np.mean(centroids) + 1e-8)
                return min(complexity, 1.0)
            else:
                return 0.0
        except:
            return 0.0
    
    def temporal_complexity(self, frame):
        if len(frame) < 128:
            return 0.0
        try:
            window_size = len(frame) // 4
            zcr_values = []
            for i in range(0, len(frame) - window_size, window_size // 2):
                window = frame[i:i + window_size]
                zcr = librosa.feature.zero_crossing_rate(window)[0][0]
                zcr_values.append(zcr)
            if len(zcr_values) > 1:
                complexity = np.std(zcr_values) / (np.mean(zcr_values) + 1e-8)
                return min(complexity, 1.0)
            else:
                return 0.0
        except:
            return 0.0
    
    def spectral_entropy(self, frame):
        if len(frame) < 64:
            return 0.0
        try:
            freqs, psd = signal.welch(frame, fs=self.sr, nperseg=min(256, len(frame)))            
            psd_norm = psd / (np.sum(psd) + 1e-8)
            entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-8))
            max_entropy = np.log(len(psd_norm))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            return normalized_entropy
        except:
            return 0.0


class QualityMetrics:    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def evaluate_quality(self, reference_audio, degraded_audio):      
        min_len = min(len(reference_audio), len(degraded_audio))
        ref = reference_audio[:min_len]
        deg = degraded_audio[:min_len]
        
        metrics = {}
        
        # PESQ (Perceptual Evaluation of Speech Quality) range 1.0-4.5
        pesq_score = pesq(self.sr, ref, deg, 'wb')
        metrics['pesq'] = {
            'score': pesq_score,
            'interpretation': self.pesq(pesq_score)
        }

        # STOI (Short-Time Objective Intelligibility) range 0.0-1.0
        stoi_score = stoi(ref, deg, self.sr)
        metrics['stoi'] = {
            'score': stoi_score,
            'interpretation': self.stoi(stoi_score)
        }
        
        return metrics
    
    def pesq(self, score):
        if score >= 4.0:
            return "Excellent quality"
        elif score >= 3.0:
            return "Good quality"
        elif score >= 2.0:
            return "Fair quality"
        else:
            return "Poor quality"
    
    def stoi(self, score):
        if score >= 0.9:
            return "High intelligibility"
        elif score >= 0.7:
            return "Good intelligibility"
        elif score >= 0.5:
            return "Fair intelligibility"
        else:
            return "Poor intelligibility"


class AttentionProcessor:
#process tokens using entropy
    
    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.quality_metrics = QualityMetrics()
    
    def process_tokens(self, audio_path, tokens, codec):        
        B, T, C = tokens.shape
        
        complexity_scores = self.complexity_analyzer.analyze_audio(audio_path, T)
        
        #baseline quality
        original_audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        baseline_reconstruction = codec.decode(tokens)
        baseline_audio = self._extract_audio_array(baseline_reconstruction)
        
        min_len = min(len(original_audio), len(baseline_audio))
        original_audio = original_audio[:min_len]
        baseline_audio = baseline_audio[:min_len]
        
        baseline_quality = self.quality_metrics.evaluate_quality(original_audio, baseline_audio)
        baseline_pesq = baseline_quality.get('pesq', {}).get('score', 0)
        
        print(f"Baseline PESQ: {baseline_pesq:.3f}")
        
        #entropy-based token selection
        processed_tokens = self._entropy_based_selection(tokens, complexity_scores)        
        processed_reconstruction = codec.decode(processed_tokens)
        processed_audio = self._extract_audio_array(processed_reconstruction)[:min_len]
        processed_quality = self.quality_metrics.evaluate_quality(original_audio, processed_audio)
        processed_pesq = processed_quality.get('pesq', {}).get('score', 0)
        
        improvement = processed_pesq - baseline_pesq
        print(f"Processed PESQ: {processed_pesq:.3f} (change: {improvement:+.3f})")
        
        analysis_data = {
            'complexity_scores': complexity_scores,
            'baseline_quality': baseline_quality,
            'processed_quality': processed_quality,
            'pesq_improvement': improvement
        }
        
        return processed_tokens, analysis_data
    
    def _entropy_based_selection(self, tokens, complexity_scores):
        processed_tokens = tokens.clone()
        
        entropies = []
        window_size = 5
        
        for t in range(len(complexity_scores)):
            start = max(0, t - window_size // 2)
            end = min(len(complexity_scores), t + window_size // 2 + 1)
            
            #calculate entropy of token values in window
            window_tokens = tokens[0, start:end, :].cpu().flatten().numpy()
            unique_tokens, counts = np.unique(window_tokens, return_counts=True)
            probabilities = counts / len(window_tokens)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            entropies.append(entropy)
        
        entropies = np.array(entropies)
        
        #low entropy AND low complexity
        low_entropy_threshold = np.percentile(entropies, 30)
        low_complexity_threshold = np.percentile(complexity_scores, 30)
        
        modifications = 0
        for t in range(len(complexity_scores)):
            if (entropies[t] < low_entropy_threshold and 
                complexity_scores[t] < low_complexity_threshold):
                
                for c in range(tokens.shape[2]):
                    current_token = tokens[0, t, c].item()
                    #quantization (2-bit reduction)
                    quantized = ((current_token + 2) // 4) * 4
                    processed_tokens[0, t, c] = min(16383, quantized)
                    if quantized != current_token:
                        modifications += 1

        print(f"Entropy-based modifications: {modifications} tokens")
        return processed_tokens
    
    def _extract_audio_array(self, audio_tensor):
        if isinstance(audio_tensor, torch.Tensor):
            audio_array = audio_tensor.cpu().numpy()
        else:
            audio_array = audio_tensor
        
        return audio_array.flatten()


class ModifiedSemantiCodec(nn.Module):
#Semanticodec modified with dynamic token allocation    
    def __init__(self, token_rate=100, semantic_vocab_size=16384, force_cpu=True):
        super().__init__()
        
        if force_cpu:
            self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
        
        from semanticodec import SemantiCodec as OriginalSemantiCodec
        self.base_codec = OriginalSemantiCodec(
            token_rate=token_rate,
            semantic_vocab_size=semantic_vocab_size
        )
        
        self.entropy_processor = AttentionProcessor()
    
    def encode(self, filepath):
        base_tokens = self.base_codec.encode(filepath)
        try:
            processed_tokens, analysis_data = self.entropy_processor.process_tokens(
                filepath, base_tokens, self.base_codec
            )
            return {
                'tokens': processed_tokens,
                'base_tokens': base_tokens,
                'analysis': analysis_data
            }
            
        except Exception as e:
            print(f"processing failed: {e}")
            import traceback
            traceback.print_exc()
            return base_tokens
    
    def decode(self, encoded_data):
        if isinstance(encoded_data, dict):
            tokens = encoded_data.get('tokens', encoded_data.get('base_tokens'))
        else:
            tokens = encoded_data
        
        target_device = next(self.base_codec.encoder.parameters()).device
        if tokens.device != target_device:
            tokens = tokens.to(target_device)
        
        return self.base_codec.decode(tokens)


def test_attention():
    audio_path = "overview_and_setup/data/original/avhbench_example.wav"
    
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        return False
    
    model = ModifiedSemantiCodec(force_cpu=True)
    
    result = model.encode(audio_path)
    
    if isinstance(result, dict) and 'analysis' in result:
        analysis = result['analysis']
        
        print("\nResults:")
        print(f"  Baseline PESQ: {analysis['baseline_quality']['pesq']['score']:.3f}")
        print(f"  Modified PESQ: {analysis['processed_quality']['pesq']['score']:.3f}")
        print(f"  Improvement: {analysis['pesq_improvement']:+.3f}")
        
        print(f"\n  Baseline STOI: {analysis['baseline_quality']['stoi']['score']:.3f}")
        print(f"  Processed STOI: {analysis['processed_quality']['stoi']['score']:.3f}")
        
        if analysis['pesq_improvement'] >= -0.05:  # Within 0.05 PESQ units
            print("\n SUCCESS: Quality preserved within acceptable range")
            return True
        else:
            print(f"\n Quality degradation: {analysis['pesq_improvement']:.3f}")
            return False
    else:
        print("processing failed")
        return False

class AADProcessor:
    """Process tokens using AAD-guided importance from LLM"""
    
    def __init__(self, model, processor, alpha=1.0):
        self.model = model
        self.processor = processor
        self.alpha = alpha
        self.quality_metrics = QualityMetrics()
        
    def compute_importance_map(self, audio_path, question, target_frames):
        """
        Compute per-frame importance using AAD + attention attribution.
        Returns importance scores for each audio frame.
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
        
        # Create silence
        silence_tensor = torch.zeros_like(audio_tensor)
        
        # Prepare inputs for model
        audio_np = audio_tensor.squeeze().numpy().astype(np.float32)
        silence_np = silence_tensor.squeeze().numpy().astype(np.float32)
        
        # Get logits and attention WITH audio
        inputs_with = self._prepare_inputs(audio_np, question)
        outputs_with = self.model.generate(
            **inputs_with,
            max_new_tokens=10,  # Generate a few tokens
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        
        # Get logits and attention WITHOUT audio (silence)
        inputs_without = self._prepare_inputs(silence_np, question)
        outputs_without = self.model.generate(
            **inputs_without,
            max_new_tokens=10,
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        
        # Compute importance via attention-weighted AAD
        importance_scores = self._attribute_to_audio_frames(
            outputs_with, outputs_without, target_frames, len(audio)
        )
        
        return importance_scores
    
    def _prepare_inputs(self, audio_np, question):
        """Prepare model inputs"""
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": question},
            ],
        }]
        
        try:
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(
                text=prompt, audio=audio_np, sampling_rate=16000, return_tensors="pt"
            )
        except Exception:
            audio_tok = getattr(self.processor, "audio_token", "<|AUDIO|>")
            prompt = f"{audio_tok}\n{question}"
            inputs = self.processor(
                text=prompt, audio=audio_np, sampling_rate=16000, return_tensors="pt"
            )
        
        # Move to device
        device = next(self.model.parameters()).device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        
        return inputs
    
    def _attribute_to_audio_frames(self, outputs_with, outputs_without, 
                                    target_frames, audio_len):
        """
        Map AAD signals back to audio frames using attention weights.
        """
        # Extract generated token count
        n_generated = len(outputs_with.scores)
        
        # Initialize importance accumulator for audio frames
        importance = np.zeros(target_frames)
        
        try:
            # For each generated text token
            for t in range(n_generated):
                # Compute AAD signal for this token
                logits_with = outputs_with.scores[t][0].float().cpu()  # [vocab]
                logits_without = outputs_without.scores[t][0].float().cpu()
                
                # AAD formula: (1+α) * with - α * without
                aad_logits = (1 + self.alpha) * logits_with - self.alpha * logits_without
                
                # Importance = magnitude of change
                # Use max absolute difference as importance signal
                token_importance = torch.max(torch.abs(aad_logits)).item()
                
                # Get attention weights for this token
                # attentions: tuple of layers, each [batch, heads, seq_len, seq_len]
                if hasattr(outputs_with, 'attentions') and outputs_with.attentions:
                    # Average across all layers and heads for this generation step
                    attn = outputs_with.attentions[t]  # Last layer for this token
                    
                    # attn is typically: (batch, num_heads, seq_len, seq_len)
                    # We want attention TO audio tokens FROM current text token
                    attn_avg = attn.mean(dim=1)[0, -1, :]  # [seq_len]
                    attn_avg = attn_avg.cpu().numpy()
                    
                    # Map attention weights to audio frames
                    # Assumption: first N positions are audio tokens
                    # This is model-specific - you may need to adjust
                    audio_token_start = 1  # After BOS token
                    audio_token_end = audio_token_start + target_frames
                    
                    if audio_token_end <= len(attn_avg):
                        audio_attention = attn_avg[audio_token_start:audio_token_end]
                        
                        # Normalize attention
                        audio_attention = audio_attention / (audio_attention.sum() + 1e-8)
                        
                        # Accumulate: importance[frame] += aad_signal * attention[frame]
                        importance += token_importance * audio_attention[:target_frames]
        
        except Exception as e:
            print(f"Warning: Attention extraction failed: {e}")
            # Fallback: uniform importance
            importance = np.ones(target_frames)
        
        # Normalize to [0, 1]
        if importance.max() > 0:
            importance = importance / importance.max()
        else:
            importance = np.ones(target_frames)
        
        return importance
    
    def process_tokens(self, audio_path, tokens, codec, question):
        """Process tokens based on AAD-derived importance"""
        B, T, C = tokens.shape
        
        # Compute AAD-based importance map
        print("Computing AAD importance map...")
        importance_scores = self.compute_importance_map(audio_path, question, T)
        
        # Baseline quality
        original_audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        baseline_reconstruction = codec.decode(tokens)
        baseline_audio = self._extract_audio_array(baseline_reconstruction)
        
        min_len = min(len(original_audio), len(baseline_audio))
        original_audio = original_audio[:min_len]
        baseline_audio = baseline_audio[:min_len]
        
        baseline_quality = self.quality_metrics.evaluate_quality(original_audio, baseline_audio)
        baseline_pesq = baseline_quality.get('pesq', {}).get('score', 0)
        
        print(f"Baseline PESQ: {baseline_pesq:.3f}")
        
        # AAD-guided token modification
        processed_tokens = self._aad_guided_selection(tokens, importance_scores)
        
        # Evaluate processed quality
        processed_reconstruction = codec.decode(processed_tokens)
        processed_audio = self._extract_audio_array(processed_reconstruction)[:min_len]
        processed_quality = self.quality_metrics.evaluate_quality(original_audio, processed_audio)
        processed_pesq = processed_quality.get('pesq', {}).get('score', 0)
        
        improvement = processed_pesq - baseline_pesq
        print(f"Processed PESQ: {processed_pesq:.3f} (change: {improvement:+.3f})")
        
        analysis_data = {
            'importance_scores': importance_scores,
            'baseline_quality': baseline_quality,
            'processed_quality': processed_quality,
            'pesq_improvement': improvement
        }
        
        return processed_tokens, analysis_data
    
    def _aad_guided_selection(self, tokens, importance_scores):
        """
        Modify tokens based on AAD importance:
        - High importance regions: preserve tokens (no quantization)
        - Low importance regions: quantize aggressively
        """
        processed_tokens = tokens.clone()
        
        # Determine thresholds
        high_importance_threshold = np.percentile(importance_scores, 70)
        low_importance_threshold = np.percentile(importance_scores, 30)
        
        modifications = 0
        
        for t in range(len(importance_scores)):
            if importance_scores[t] < low_importance_threshold:
                # Low importance: aggressive quantization (4-bit reduction)
                for c in range(tokens.shape[2]):
                    current_token = tokens[0, t, c].item()
                    quantized = ((current_token + 8) // 16) * 16
                    processed_tokens[0, t, c] = min(16383, quantized)
                    if quantized != current_token:
                        modifications += 1
            
            elif importance_scores[t] < high_importance_threshold:
                # Medium importance: moderate quantization (2-bit reduction)
                for c in range(tokens.shape[2]):
                    current_token = tokens[0, t, c].item()
                    quantized = ((current_token + 2) // 4) * 4
                    processed_tokens[0, t, c] = min(16383, quantized)
                    if quantized != current_token:
                        modifications += 1
            
            # else: high importance, no modification
        
        print(f"AAD-guided modifications: {modifications} tokens")
        return processed_tokens
    
    def _extract_audio_array(self, audio_tensor):
        if isinstance(audio_tensor, torch.Tensor):
            audio_array = audio_tensor.cpu().numpy()
        else:
            audio_array = audio_tensor
        return audio_array.flatten()


class AADSemanticCodec(nn.Module):
    """SemantiCodec with AAD-guided dynamic token allocation"""
    
    def __init__(self, aad_model, aad_processor, token_rate=100, 
                 semantic_vocab_size=16384, alpha=1.0, force_cpu=False):
        super().__init__()
        
        self.device = torch.device("cpu") if force_cpu else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        from semanticodec import SemantiCodec as OriginalSemantiCodec
        self.base_codec = OriginalSemantiCodec(
            token_rate=token_rate,
            semantic_vocab_size=semantic_vocab_size
        )
        
        self.aad_processor = AADProcessor(aad_model, aad_processor, alpha)
    
    def encode(self, filepath, question):
        """
        Encode audio with AAD-guided allocation.
        
        Args:
            filepath: Path to audio file
            question: Question being asked about the audio
        """
        # Standard encoding
        base_tokens = self.base_codec.encode(filepath)
        
        try:
            # AAD-guided processing
            processed_tokens, analysis_data = self.aad_processor.process_tokens(
                filepath, base_tokens, self.base_codec, question
            )
            
            return {
                'tokens': processed_tokens,
                'base_tokens': base_tokens,
                'analysis': analysis_data,
                'question': question
            }
            
        except Exception as e:
            print(f"AAD processing failed: {e}")
            import traceback
            traceback.print_exc()
            return base_tokens
    
    def decode(self, encoded_data):
        if isinstance(encoded_data, dict):
            tokens = encoded_data.get('tokens', encoded_data.get('base_tokens'))
        else:
            tokens = encoded_data
        
        target_device = next(self.base_codec.encoder.parameters()).device
        if tokens.device != target_device:
            tokens = tokens.to(target_device)
        
        return self.base_codec.decode(tokens)


def test_aad_codec(model, processor, audio_path, question):
    """Test AAD-guided codec"""
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        return False
    
    codec = AADSemanticCodec(
        aad_model=model,
        aad_processor=processor,
        alpha=1.0,
        force_cpu=False  # Use GPU for LLM
    )
    
    result = codec.encode(audio_path, question)
    
    if isinstance(result, dict) and 'analysis' in result:
        analysis = result['analysis']
        
        print("\n=== AAD-Guided Results ===")
        print(f"Question: {question}")
        print(f"Baseline PESQ: {analysis['baseline_quality']['pesq']['score']:.3f}")
        print(f"Processed PESQ: {analysis['processed_quality']['pesq']['score']:.3f}")
        print(f"Improvement: {analysis['pesq_improvement']:+.3f}")
        print(f"Mean importance: {np.mean(analysis['importance_scores']):.3f}")
        print(f"Importance variance: {np.var(analysis['importance_scores']):.3f}")
        
        return analysis['pesq_improvement'] >= -0.05
    else:
        print("AAD processing failed")
        return False


class DASTAudioProcessor:
    """DAST-style dynamic allocation using AAD (local) + Attention (global)"""
    
    def __init__(self, model, processor, alpha=0.5):
        self.model = model
        self.processor = processor
        self.alpha = alpha
        self.quality_metrics = QualityMetrics()
        
    def compute_local_importance(self, audio_path, question, target_frames):
        """
        Local importance via AAD signal per audio segment.
        Analogous to perplexity in DAST.
        """
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
        silence_tensor = torch.zeros_like(audio_tensor)
        
        # Divide audio into segments
        segment_length = len(audio) // target_frames
        local_scores = []
        
        for i in range(target_frames):
            start = i * segment_length
            end = min((i + 1) * segment_length, len(audio))
            
            audio_seg = audio_tensor[:, start:end]
            silence_seg = silence_tensor[:, start:end]
            
            # Compute AAD for this segment
            aad_signal = self._compute_segment_aad(audio_seg, silence_seg, question)
            local_scores.append(aad_signal)
        
        return np.array(local_scores)
    
    def _compute_segment_aad(self, audio_seg, silence_seg, question):
        """Compute AAD signal strength for a segment"""
        audio_np = audio_seg.squeeze().numpy().astype(np.float32)
        silence_np = silence_seg.squeeze().numpy().astype(np.float32)
        
        inputs_with = self._prepare_inputs(audio_np, question)
        inputs_without = self._prepare_inputs(silence_np, question)
        
        # Generate one token to get AAD signal
        with torch.no_grad():
            out_with = self.model.generate(
                **inputs_with,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
            out_without = self.model.generate(
                **inputs_without,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        # AAD signal = magnitude of difference
        logits_with = out_with.scores[0][0].float().cpu()
        logits_without = out_without.scores[0][0].float().cpu()
        
        aad_diff = torch.abs(logits_with - logits_without)
        return float(aad_diff.max().item())
    
    def compute_global_importance(self, audio_path, question, target_frames):
        """
        Global importance via attention weights.
        Analogous to attention in DAST.
        """
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
        audio_np = audio_tensor.squeeze().numpy().astype(np.float32)
        
        inputs = self._prepare_inputs(audio_np, question)
        
        # Generate with attention tracking
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,  # Generate a few tokens
                output_attentions=True,
                return_dict_in_generate=True,
            )
        
        # Extract attention to audio frames
        global_scores = np.zeros(target_frames)
        
        try:
            # Average attention across all generated tokens and layers
            for t in range(len(outputs.attentions)):
                attn = outputs.attentions[t][-1]  # Last layer
                attn_avg = attn.mean(dim=1)[0, -1, :].cpu().numpy()  # [seq_len]
                
                # Map to audio frames (model-specific, may need adjustment)
                audio_start = 1
                audio_end = audio_start + target_frames
                
                if audio_end <= len(attn_avg):
                    frame_attn = attn_avg[audio_start:audio_end]
                    global_scores += frame_attn[:target_frames]
            
            # Normalize
            if global_scores.max() > 0:
                global_scores = global_scores / global_scores.max()
                
        except Exception as e:
            print(f"Warning: Attention extraction failed: {e}")
            global_scores = np.ones(target_frames)
        
        return global_scores
    
    def compute_combined_scores(self, local_scores, global_scores):
        """
        DAST formula: Si = Ai * α - (Pi / sum(Pk)) * (1 - α)
        Adapted: Si = Gi * α + (Li / sum(Lk)) * (1 - α)
        
        Note: We use + instead of - because:
        - In DAST: lower perplexity = more important (inverse relationship)
        - Here: higher AAD = more important (direct relationship)
        """
        # Normalize local scores
        local_norm = local_scores / (local_scores.sum() + 1e-8)
        
        # Combine: global * α + local * (1 - α)
        combined = global_scores * self.alpha + local_norm * (1 - self.alpha)
        
        # Softmax normalization
        combined_exp = np.exp(combined - combined.max())
        scores = combined_exp / combined_exp.sum()
        
        return scores
    
    def allocate_tokens(self, scores, total_tokens, target_frames):
        """
        Allocate tokens based on combined scores.
        di = M × Si
        """
        allocation = (scores * total_tokens).astype(int)
        
        # Ensure at least 1 token per frame
        allocation = np.maximum(allocation, 1)
        
        # Adjust to match total budget
        current_total = allocation.sum()
        diff = total_tokens - current_total
        
        if diff > 0:
            # Add remaining tokens to highest-scoring frames
            top_indices = np.argsort(scores)[-diff:]
            allocation[top_indices] += 1
        elif diff < 0:
            # Remove excess tokens from lowest-scoring frames
            bottom_indices = np.argsort(scores)[:abs(diff)]
            allocation[bottom_indices] = np.maximum(allocation[bottom_indices] - 1, 1)
        
        return allocation
    
    def process_tokens(self, audio_path, tokens, codec, question):
        """Apply DAST-style dynamic allocation"""
        B, T, C = tokens.shape
        
        print("Computing local importance (AAD)...")
        local_scores = self.compute_local_importance(audio_path, question, T)
        
        print("Computing global importance (Attention)...")
        global_scores = self.compute_global_importance(audio_path, question, T)
        
        print("Combining scores...")
        combined_scores = self.compute_combined_scores(local_scores, global_scores)
        
        # For visualization
        total_tokens = T * C  # Total token budget
        allocation = self.allocate_tokens(combined_scores, total_tokens, T)
        
        print(f"Token allocation - Min: {allocation.min()}, Max: {allocation.max()}, Mean: {allocation.mean():.1f}")
        
        # Apply token modification based on allocation
        processed_tokens = self._apply_allocation(tokens, allocation, combined_scores)
        
        # Evaluate quality
        original_audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        baseline_reconstruction = codec.decode(tokens)
        baseline_audio = self._extract_audio_array(baseline_reconstruction)
        
        min_len = min(len(original_audio), len(baseline_audio))
        original_audio = original_audio[:min_len]
        baseline_audio = baseline_audio[:min_len]
        
        baseline_quality = self.quality_metrics.evaluate_quality(original_audio, baseline_audio)
        baseline_pesq = baseline_quality.get('pesq', {}).get('score', 0)
        
        print(f"Baseline PESQ: {baseline_pesq:.3f}")
        
        processed_reconstruction = codec.decode(processed_tokens)
        processed_audio = self._extract_audio_array(processed_reconstruction)[:min_len]
        processed_quality = self.quality_metrics.evaluate_quality(original_audio, processed_audio)
        processed_pesq = processed_quality.get('pesq', {}).get('score', 0)
        
        improvement = processed_pesq - baseline_pesq
        print(f"Processed PESQ: {processed_pesq:.3f} (change: {improvement:+.3f})")
        
        analysis_data = {
            'local_scores': local_scores,
            'global_scores': global_scores,
            'combined_scores': combined_scores,
            'allocation': allocation,
            'baseline_quality': baseline_quality,
            'processed_quality': processed_quality,
            'pesq_improvement': improvement
        }
        
        return processed_tokens, analysis_data
    
    def _apply_allocation(self, tokens, allocation, scores):
        """
        Modify tokens based on allocation:
        - High allocation (high score): preserve tokens
        - Low allocation (low score): quantize aggressively
        """
        processed_tokens = tokens.clone()
        
        # Determine quantization level per frame based on score percentile
        high_threshold = np.percentile(scores, 66)
        low_threshold = np.percentile(scores, 33)
        
        modifications = 0
        
        for t in range(len(scores)):
            if scores[t] >= high_threshold:
                # High importance: no modification
                continue
            elif scores[t] >= low_threshold:
                # Medium importance: light quantization
                for c in range(tokens.shape[2]):
                    current = tokens[0, t, c].item()
                    quantized = ((current + 2) // 4) * 4
                    processed_tokens[0, t, c] = min(16383, quantized)
                    if quantized != current:
                        modifications += 1
            else:
                # Low importance: aggressive quantization
                for c in range(tokens.shape[2]):
                    current = tokens[0, t, c].item()
                    quantized = ((current + 8) // 16) * 16
                    processed_tokens[0, t, c] = min(16383, quantized)
                    if quantized != current:
                        modifications += 1
        
        print(f"DAST-style modifications: {modifications} tokens")
        return processed_tokens
    
    def _prepare_inputs(self, audio_np, question):
        """Prepare model inputs"""
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": question},
            ],
        }]
        
        try:
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(
                text=prompt, audio=audio_np, sampling_rate=16000, return_tensors="pt"
            )
        except Exception:
            audio_tok = getattr(self.processor, "audio_token", "<|AUDIO|>")
            prompt = f"{audio_tok}\n{question}"
            inputs = self.processor(
                text=prompt, audio=audio_np, sampling_rate=16000, return_tensors="pt"
            )
        
        device = next(self.model.parameters()).device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        
        return inputs
    
    def _extract_audio_array(self, audio_tensor):
        if isinstance(audio_tensor, torch.Tensor):
            audio_array = audio_tensor.cpu().numpy()
        else:
            audio_array = audio_tensor
        return audio_array.flatten()


class DASTSemanticCodec(nn.Module):
    """SemantiCodec with DAST-style dynamic allocation"""
    
    def __init__(self, aad_model, aad_processor, token_rate=100, 
                 semantic_vocab_size=16384, alpha=0.5, force_cpu=False):
        super().__init__()
        
        self.device = torch.device("cpu") if force_cpu else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        from semanticodec import SemantiCodec as OriginalSemantiCodec
        self.base_codec = OriginalSemantiCodec(
            token_rate=token_rate,
            semantic_vocab_size=semantic_vocab_size
        )
        
        self.dast_processor = DASTAudioProcessor(aad_model, aad_processor, alpha)
    
    def encode(self, filepath, question):
        """Encode with DAST-style dynamic allocation"""
        base_tokens = self.base_codec.encode(filepath)
        
        try:
            processed_tokens, analysis_data = self.dast_processor.process_tokens(
                filepath, base_tokens, self.base_codec, question
            )
            
            return {
                'tokens': processed_tokens,
                'base_tokens': base_tokens,
                'analysis': analysis_data,
                'question': question
            }
            
        except Exception as e:
            print(f"DAST processing failed: {e}")
            import traceback
            traceback.print_exc()
            return base_tokens
    
    def decode(self, encoded_data):
        if isinstance(encoded_data, dict):
            tokens = encoded_data.get('tokens', encoded_data.get('base_tokens'))
        else:
            tokens = encoded_data
        
        target_device = next(self.base_codec.encoder.parameters()).device
        if tokens.device != target_device:
            tokens = tokens.to(target_device)
        
        return self.base_codec.decode(tokens)


if __name__ == "__main__":        
    success = test_attention()
