# ## 5. AAD Implementation with Actual Logits
import torch
import torch.nn.functional as F

import numpy as np
from typing import Dict

# %%
class AudioAwareDecoder:
    """Implements Audio-Aware Decoding with actual logit extraction"""

    def __init__(self, model, processor, device, alpha: float = 1.0):
        self.model = model
        self.processor = processor
        self.alpha = alpha
        self.device = device

    def prepare_inputs(self, audio: torch.Tensor, question: str):
        """Prepare inputs for Qwen2-Audio (text + audio with an audio token in text)."""
        # [1, T] -> numpy float32 1-D
        audio_np = audio.squeeze().detach().cpu().numpy().astype(np.float32)

        # 1) Try chat template with an explicit audio item to auto-insert the token
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": f"Answer with only 'yes' or 'no': {question}"},
            ],
        }]

        try:
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            # Preferred path: text has the token, and we pass one audio
            inputs = self.processor(
                text=prompt,
                audio=audio_np,           # singular; matches 1 token in text
                sampling_rate=16000,
                return_tensors="pt",
            )
        except Exception:
            # 2) Fallback: manually inject the audio token the processor expects
            audio_tok = getattr(self.processor, "audio_token", "<|AUDIO|>")
            prompt = (
                f"{audio_tok}\n"
                f"Answer with only 'yes' or 'no': {question}"
            )
            inputs = self.processor(
                text=prompt,
                audio=audio_np,
                sampling_rate=16000,
                return_tensors="pt",
            )

        # Move tensors to device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)
        return inputs

    def get_answer_token_logits(self, audio: torch.Tensor, question: str) -> Dict[str, float]:
        """Get raw logits for yes/no tokens

        CHANGES:
        - Use model.generate(..., output_scores=True, max_new_tokens=1) to get the
          next-token distribution directly and robustly.
        - Use leading-space tokens (' yes'/' no') which typically align with tokenization.
        """
        inputs = self.prepare_inputs(audio, question)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
            # First generated step's logits
            logits = out.scores[0][0]  # [vocab]

            tok = self.processor.tokenizer
            # Prefer leading-space variants; fallback to bare if needed
            yes_ids = tok(" yes", add_special_tokens=False).input_ids
            no_ids  = tok(" no",  add_special_tokens=False).input_ids
            if not yes_ids:  # fallback
                yes_ids = tok("yes", add_special_tokens=False).input_ids
            if not no_ids:
                no_ids = tok("no", add_special_tokens=False).input_ids

            yes_id = yes_ids[0]
            no_id  = no_ids[0]

            yes_logit = logits[yes_id].item()
            no_logit  = logits[no_id].item()

        return {"yes": yes_logit, "no": no_logit}

    def compute_delta(self, audio: torch.Tensor, silence: torch.Tensor, question: str) -> float:
        """
        Compute AAD delta: the evidence margin after contrastive decoding
        (kept the same signature)
        """
        # Get logits with audio
        logits_with_audio = self.get_answer_token_logits(audio, question)

        # Get logits with silence (must be same length)
        logits_with_silence = self.get_answer_token_logits(silence, question)

        # Apply AAD formula: z_AAD = (1 + α) * z_with - α * z_without
        yes_aad = (1 + self.alpha) * logits_with_audio["yes"] - self.alpha * logits_with_silence["yes"]
        no_aad = (1 + self.alpha) * logits_with_audio["no"] - self.alpha * logits_with_silence["no"]

        # Compute probabilities using softmax
        logits_aad = torch.tensor([no_aad, yes_aad], dtype=torch.float32)
        probs_aad = F.softmax(logits_aad, dim=0)

        # Delta is log probability ratio (positive means yes is more likely)
        delta = torch.log(probs_aad[1] / probs_aad[0]).item()

        return delta
