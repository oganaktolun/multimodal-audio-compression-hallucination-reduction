from typing import Any, Tuple
import torch

def load_qwen_baseline(device: str = "cuda") -> Tuple[Any, Any, Any]:

    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()

    def video_pre(v):
        return v

    def audio_pre(wav, sr):
        if isinstance(wav, torch.Tensor) and wav.dim() == 1:
            return wav.unsqueeze(0)
        return wav

    class QwenBaselineAdapter:
        def __init__(self, model, processor, video_key="video", audio_key="audio"):
            self.model = model
            self.processor = processor
            self.vk = video_key
            self.ak = audio_key

        def generate(self, prompt, video=None, audio=None, **gen_args) -> str:
            print(f"prompt = {prompt[:50]}...")

            default_gen_args = {
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            default_gen_args.update(gen_args)

            with torch.inference_mode():
                conversation = [{"role": "user", "content": []}]
                if audio is not None:
                    print(f"audio: {audio}")
                    if isinstance(audio, str):
                        conversation[0]["content"].append({
                            "type": "audio",
                            "audio_url": audio
                        })
                    else:
                        conversation[0]["content"].append({
                            "type": "audio",
                            "audio": audio
                        })

                if video is not None:
                    if isinstance(video, str):
                        conversation[0]["content"].append({
                            "type": "video",
                            "video_url": video
                        })
                    else:
                        conversation[0]["content"].append({
                            "type": "video",
                            "video": video
                        })

                conversation[0]["content"].append({
                    "type": "text",
                    "text": prompt
                })
 
                inputs = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True
                )

                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in inputs.items()}

                outputs = self.model.generate(**inputs, **default_gen_args)

                generated_ids = outputs[0][len(inputs["input_ids"][0]):]
                response = self.processor.batch_decode(
                    [generated_ids], 
                    skip_special_tokens=True
                )[0]
                result = response.strip()
                print(f"res: {result[:100]}...")
                return result
 
    adapter = QwenBaselineAdapter(model, processor)
    result = (adapter, video_pre, audio_pre)

    return result
