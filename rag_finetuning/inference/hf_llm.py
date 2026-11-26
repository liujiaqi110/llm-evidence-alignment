"""Optional local generator for the knowledge platform: Qwen2.5-1.5B-Instruct (4-bit) with or without the P6 adapter.

It implements the same small interface as ``rag_finetuning.compat.local_llm.OllamaClient`` (``chat`` / ``model_info``), so it is
used as a drop-in ``llm`` behind the EXISTING safety chain and never instead of it:

    KnowledgeBaseService ─ AclRetriever (ACL at candidate stage)
                         ─ PromptGuardLLM (refuses any unauthorised context block)  ← this class sits behind it
                         ─ SafeRagPipeline citation validation (unknown / unauthorised IDs are blocked)

Nothing here is wired into ``kb_platform`` by default; production keeps ``qwen2.5:7b`` via Ollama. Opt-in example:

    from kb_platform.rag_backend import RagBackend
    backend = RagBackend(index=..., llm_factory=lambda: LocalHFLLM(adapter="workspace/runs/<run>/adapter"))

Offline only: loads from the local Hugging Face cache (``local_files_only=True``); never downloads.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_finetuning.compat.local_llm import GenerationError, GenResult

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


class LocalHFLLM:
    def __init__(self, adapter: Optional[str] = None, max_new_tokens_cap: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise GenerationError("unavailable", "CUDA GPU required for the local 4-bit model")
        self.torch, self.adapter, self.cap = torch, adapter, max_new_tokens_cap
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, quantization_config=q, device_map={"": 0},
                                                 dtype=torch.bfloat16, local_files_only=True)
        if adapter:
            from peft import PeftModel
            if not (Path(adapter) / "adapter_config.json").exists():
                raise GenerationError("model_missing", f"adapter not found: {adapter}")
            m = PeftModel.from_pretrained(m, adapter)
        self.model = m.eval()
        self.model_name = "qwen2.5-1.5b-instruct-nf4" + (f"+{Path(adapter).parent.name}" if adapter else "")
        self._lock = threading.Lock()

    def model_info(self) -> Dict[str, Any]:
        return {"model": self.model_name, "hf_id": MODEL_ID, "revision": REVISION, "adapter": self.adapter,
                "quantization": "nf4", "digest": REVISION[:12]}

    def chat(self, messages: List[Dict[str, str]], options: Dict[str, Any], fmt: Optional[str] = None) -> GenResult:
        if fmt:
            raise GenerationError("bad_response", "structured output (format=) is not supported by LocalHFLLM")
        torch = self.torch
        n = min(int(options.get("num_predict", 512)), self.cap)
        with self._lock:
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False,
                                          pad_token_id=self.tok.eos_token_id)
            wall = (time.perf_counter() - t0) * 1000
        new = out[0, ids.shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        if not text:
            raise GenerationError("empty_response", "empty generation")
        return GenResult(text=text, model=self.model_name, prompt_tokens=int(ids.shape[1]), output_tokens=int(new.shape[0]),
                         total_ms=round(wall, 2), done_reason="length" if new.shape[0] >= n else "stop")
