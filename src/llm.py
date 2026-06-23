"""Local causal LLM served via Hugging Face Transformers.

Defaults to Qwen2.5-3B-Instruct in 4-bit (NF4) so it fits in a few GB of VRAM,
leaving headroom for the embedding model on the same 8 GB card. Falls back to
fp16 automatically if bitsandbytes is unavailable.
"""

from __future__ import annotations

from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import config


class LocalLLM:
    def __init__(self, model_name: str | None = None, load_in_4bit: bool | None = None):
        self.model_name = model_name or config.llm_model
        load_in_4bit = config.load_in_4bit if load_in_4bit is None else load_in_4bit

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs: Dict = {"torch_dtype": torch.float16, "device_map": "auto"}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401  (presence check)
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as e:  # bitsandbytes missing/broken -> fp16 fallback
                print(f"[LocalLLM] 4-bit unavailable ({e}); loading in fp16 instead.")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

    @torch.inference_mode()
    def chat(self, messages: List[Dict[str, str]],
             max_new_tokens: int | None = None,
             temperature: float | None = None) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        temperature = config.temperature if temperature is None else temperature
        do_sample = temperature > 0
        gen_kwargs: Dict = {
            "max_new_tokens": max_new_tokens or config.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        out = self.model.generate(**inputs, **gen_kwargs)
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def complete(self, system: str, user: str, **kwargs) -> str:
        return self.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            **kwargs,
        )
