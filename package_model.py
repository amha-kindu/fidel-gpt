import os
import re
import glob
import json
import shutil
import inspect
import torch
import argparse

from config import DEVICE, Config, ModelConfig, ModelWithLoRAConfig, LOGGER


def find_latest_checkpoint(base_path: str) -> str | None:
    stem = base_path[:-3] if base_path.endswith('.pt') else base_path
    step_pattern = re.compile(r'-(\d+(?:\.\d+)?)K\.pt$')
    versioned = []
    for path in glob.glob(f"{stem}-*.pt"):
        if step_pattern.search(path):
            versioned.append((float(step_pattern.search(path).group(1)), path))
    return max(versioned, key=lambda x: x[0])[1] if versioned else None


def resolve_checkpoint(path: str, label: str) -> str:
    latest = find_latest_checkpoint(path)
    if latest:
        LOGGER.info(f"[{label}] Using latest versioned checkpoint: '{latest}'")
        return latest
    if os.path.isfile(path):
        LOGGER.info(f"[{label}] Using checkpoint: '{path}'")
        return path
    raise FileNotFoundError(f"No checkpoint found at or matching '{path}'")


def build_config_py(is_lora: bool) -> str:
    sections = [
        "import torch",
        "",
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "MIXED_PRECISION_ENABLED = ("
        "\n    torch.amp.autocast_mode.is_autocast_available(DEVICE.type)"
        "\n    if torch.cuda.is_available() else False"
        "\n)",
        "",
        "",
        inspect.getsource(Config),
        "",
        "",
        inspect.getsource(ModelConfig),
    ]
    if is_lora:
        sections += ["", "", inspect.getsource(ModelWithLoRAConfig)]
    return "\n".join(sections)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Package a trained checkpoint for deployment')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Base pretrained checkpoint path')
    parser.add_argument('--lora-checkpoint', type=str, default='',
                        help='Base path for LoRA adapter checkpoints.')
    parser.add_argument('--finetuned-checkpoint', type=str, default='',
                        help='Base path for selective-param finetuned checkpoints.')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Output directory name under --output-dir')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Root output directory (default: models/)')

    args = parser.parse_args()
    
    out_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(out_dir, exist_ok=True)

    # --- Base checkpoint ---
    base_path = resolve_checkpoint(args.checkpoint, 'base')
    base_ckpt = torch.load(base_path, map_location=DEVICE, weights_only=False)
    base_weights: dict = dict(base_ckpt['weights'])
    base_config: ModelConfig = base_ckpt['model_config']

    # Merge finetuned delta into base weights if provided
    if bool(args.finetuned_checkpoint):
        ft_path = resolve_checkpoint(args.finetuned_checkpoint, 'finetuned')
        ft_ckpt = torch.load(ft_path, map_location='cpu', weights_only=False)
        base_weights.update(ft_ckpt['weights'])
        LOGGER.info("Merged finetuned weights into base.")

    ckpt_out = os.path.join(out_dir, 'checkpoint.pt')
    LOGGER.info(f"Saving checkpoint → '{ckpt_out}'...")
    torch.save(base_weights, ckpt_out)

    meta_out = os.path.join(out_dir, 'metadata.json')
    with open(meta_out, 'w', encoding='utf-8') as f:
        json.dump(base_config.to_dict(), f, indent=2)
    LOGGER.info(f"Saved metadata → '{meta_out}'.")

    is_lora = bool(args.lora_checkpoint)
    if is_lora:
        lora_path = resolve_checkpoint(args.lora_checkpoint, 'lora')
        lora_ckpt = torch.load(lora_path, map_location='cpu', weights_only=False)
        lora_config: ModelWithLoRAConfig = lora_ckpt['model_config']

        lora_ckpt_out = os.path.join(out_dir, 'checkpoint-lora.pt')
        LOGGER.info(f"Saving LoRA checkpoint → '{lora_ckpt_out}'...")
        torch.save(lora_ckpt['weights'], lora_ckpt_out)

        lora_meta_out = os.path.join(out_dir, 'metadata-lora.json')
        with open(lora_meta_out, 'w', encoding='utf-8') as f:
            json.dump(lora_config.to_dict(), f, indent=2)
        LOGGER.info(f"Saved LoRA metadata → '{lora_meta_out}'.")

    # --- Copy Python files ---
    shutil.copy2(os.path.join(PROJECT_ROOT, 'model.py'), os.path.join(out_dir, 'model.py'))
    LOGGER.info("Copied 'model.py'.")

    if is_lora:
        shutil.copy2(os.path.join(PROJECT_ROOT, 'lora.py'), os.path.join(out_dir, 'lora.py'))
        LOGGER.info("Copied 'lora.py'.")

    config_out = os.path.join(out_dir, 'config.py')
    with open(config_out, 'w', encoding='utf-8') as f:
        f.write(build_config_py(is_lora))
    LOGGER.info(f"Wrote trimmed config → '{config_out}'.")

    LOGGER.info("Done.")
