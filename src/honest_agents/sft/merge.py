"""Merge an SFT LoRA adapter into its base model, so GRPO (trainer and vLLM server) can start from the SFT policy.

The GRPO trainer builds a fresh LoRA on `model.name`; it has no option to start from an existing adapter. Merging the SFT
adapter into the bf16 base and pointing `model.name` at the merged directory gives GRPO the SFT policy as its starting
point, with the same architecture (vLLM loads it like the original checkpoint) and a fresh RL adapter on top.

    python -m honest_agents.sft.merge --base Qwen/Qwen3.5-4B --adapter outputs/sft/teacher-qwen35-s0/final \
        --out outputs/merged/qwen35-teacher-sft

The base is loaded with the class named in its config (`architectures[0]`), the class the adapter was trained on, so
the adapter's module names match; a merge that loads no LoRA weights, or leaves the weights unchanged, is refused.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def merge(base: str, adapter: str, out: str, dtype: str = "bfloat16") -> dict:
    import torch
    import transformers
    from peft import PeftModel

    config = transformers.AutoConfig.from_pretrained(base)
    arch = (config.architectures or ["AutoModelForCausalLM"])[0]
    cls = getattr(transformers, arch, None) or transformers.AutoModelForCausalLM
    model = cls.from_pretrained(base, dtype=getattr(torch, dtype))
    peft_model = PeftModel.from_pretrained(model, adapter)
    lora_a = [n for n, _ in peft_model.named_parameters() if "lora_A" in n]
    if not lora_a:
        raise SystemExit(f"no LoRA weights loaded from {adapter} onto {arch}: module names do not match")
    # One adapted layer, to check the merge really moved it: `...<module>.lora_A.default.weight` adapts
    # `...<module>.base_layer.weight`, which becomes `<module>.weight` after merge_and_unload.
    module = lora_a[0].split(".lora_A.")[0]
    params = dict(peft_model.named_parameters())
    before = params[f"{module}.base_layer.weight"].detach().clone()
    merged = peft_model.merge_and_unload()
    probe_name = module.removeprefix("base_model.model.") + ".weight"
    moved = not torch.equal(before, dict(merged.named_parameters())[probe_name].detach())
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    # Tokenizer, chat template and (for Qwen3.5) the processor files travel with the weights, so vLLM and the trainer
    # render prompts exactly as they would for the original checkpoint.
    try:
        transformers.AutoProcessor.from_pretrained(base).save_pretrained(out_dir)
    except Exception:  # noqa: BLE001 - text-only models have no processor
        transformers.AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)
    for extra in ("generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        src = Path(transformers.utils.cached_file(base, extra, _raise_exceptions_for_missing_entries=False) or "")
        if src.is_file() and not (out_dir / extra).exists():
            shutil.copy(src, out_dir / extra)
    info = {"base": base, "adapter": adapter, "architecture": arch, "lora_modules": len(lora_a),
            "probe_param": probe_name, "probe_changed": bool(moved)}  # fmt: skip
    (out_dir / "merge_info.json").write_text(json.dumps(info, indent=2))
    if not moved:
        raise SystemExit(f"merge left {probe_name} unchanged: the adapter did not apply ({info})")
    return info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m honest_agents.sft.merge", description=__doc__.split("\n\n")[0])
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args(argv)
    print(json.dumps(merge(args.base, args.adapter, args.out, args.dtype), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
