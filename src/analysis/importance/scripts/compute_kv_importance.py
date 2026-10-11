
import argparse
import os

import torch
from tqdm import tqdm

from analysis.importance.code.kv_importance import KVImportanceAnalyzer
from common.adapters import get_adapter


def _get_sample_id(filename):
    assert filename.endswith(".pt")
    return filename[:-3]


def _gather_kv_heads(tensor, device):
    world_size = torch.distributed.get_world_size()
    local = tensor.to(device).contiguous()
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local)
    return torch.cat([t.cpu() for t in gathered], dim=1)


def _existing_ids(output_dir):
    if not os.path.exists(output_dir):
        return set()
    return {_get_sample_id(f) for f in os.listdir(output_dir) if f.endswith(".pt")}


def main():
    parser = argparse.ArgumentParser(description="Compute KV importance")
    parser.add_argument("--adapter", type=str, required=True,
                        choices=["voxtral", "qwen2_5_omni"])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--preprocess-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--tp-plan", type=str, default=None,
                        help="Tensor parallel plan, e.g. 'auto'. Requires torchrun launch.")
    args = parser.parse_args()

    adapter = get_adapter(args.adapter)

    print("Loading model...")
    if args.tp_plan:
        if args.adapter == "qwen2_5_omni":
            from transformers import Qwen2_5OmniForConditionalGeneration
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16, tp_plan=args.tp_plan,
            )
        elif args.adapter == "voxtral":
            from transformers import AutoModelForSpeechSeq2Seq
            from transformers.models.mistral.configuration_mistral import MistralConfig
            prefix = "language_model.model."
            voxtral_plan = {prefix + k: v for k, v in MistralConfig.base_model_tp_plan.items()}
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16, tp_plan=voxtral_plan,
                trust_remote_code=True,
            )
        else:
            raise ValueError(
                f"--tp-plan is only supported for qwen2_5_omni and voxtral, not {args.adapter}"
            )
    else:
        model, _processor = adapter.load(args.model_path, args.device_map, degqa=False)
    model.eval()

    analyzer = KVImportanceAnalyzer(model, adapter)
    audio_token_id = adapter.get_audio_token_id(model)

    all_files = sorted(f for f in os.listdir(args.preprocess_dir) if f.endswith(".pt"))
    existing = _existing_ids(args.output_dir)
    pending = [f for f in all_files if _get_sample_id(f) not in existing]
    print(f"Total: {len(all_files)}, existing: {len(existing)}, pending: {len(pending)}")

    is_main = True
    if args.tp_plan and torch.distributed.is_initialized():
        is_main = torch.distributed.get_rank() == 0

    os.makedirs(args.output_dir, exist_ok=True)
    for filename in tqdm(pending, desc="KV importance", disable=not is_main):
        sample_id = _get_sample_id(filename)
        data = torch.load(os.path.join(args.preprocess_dir, filename), weights_only=True)

        audio_positions = (data["input_ids"][0] == audio_token_id).nonzero(as_tuple=True)[0]

        result = analyzer.compute(
            inputs_embeds=data["inputs_embeds"],
            labels=data["labels"],
            audio_positions=audio_positions,
            input_ids=data["input_ids"] if adapter.requires_input_ids else None,
        )

        if args.tp_plan and torch.distributed.is_initialized() \
                and torch.distributed.get_world_size() > 1:
            gather_device = next(model.parameters()).device
            result["audio_sensitivity_k"] = _gather_kv_heads(
                result["audio_sensitivity_k"], gather_device)
            result["audio_sensitivity_v"] = _gather_kv_heads(
                result["audio_sensitivity_v"], gather_device)

        if is_main:
            torch.save(result, os.path.join(args.output_dir, f"{sample_id}.pt"))

        del data, result
        torch.cuda.empty_cache()

    if is_main:
        print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
