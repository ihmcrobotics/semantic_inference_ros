"""Local Hugging Face Qwen client for structured text generation."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from semantic_inference_python.config import Config


@dataclass
class QwenClientConfig(Config):
    """Configuration for a local Qwen text-generation model."""

    model_name: str = "Qwen/Qwen3-1.7B"
    model_path: str = ""
    dtype: str = "bfloat16"
    use_cuda: bool = True
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.8
    repetition_penalty: float = 1.0


class QwenClient:
    """Generate text locally with a Qwen causal language model."""

    def __init__(
        self,
        config: QwenClientConfig,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.config = config
        self.system_prompt = system_prompt or ""
        self.device = torch.device(
            "cuda"
            if config.use_cuda and torch.cuda.is_available()
            else "cpu"
        )
        self.dtype = self._resolve_dtype(config.dtype)
        model_source = config.model_path or config.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        if self.device.type == "cpu":
            return torch.float32
        if dtype == "bfloat16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "Qwen is configured for bfloat16, but the GPU does not "
                    "support BF16. Set dtype to float16."
                )
            return torch.bfloat16
        if dtype == "float16":
            return torch.float16
        if dtype == "float32":
            return torch.float32
        raise ValueError(f"Unsupported Qwen dtype: {dtype}")

    def generate_response(
        self,
        prompt: str,
        log: bool = False,
    ) -> Tuple[str, bool]:
        """Generate one response using the OpenAI-client-compatible API."""
        messages = []
        if self.system_prompt:
            messages.append(
                {"role": "system", "content": self.system_prompt}
            )
        messages.append({"role": "user", "content": prompt})

        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        ).to(self.device)

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.do_sample:
            generation_kwargs.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            generated_tokens = generated[0, inputs.input_ids.shape[1] :]
            response = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()
            if log:
                print(f"[QwenClient] User prompt: {prompt}")
                print(f"[QwenClient] Response: {response}")
            return response, True
        except Exception as exception:
            return str(exception), False
