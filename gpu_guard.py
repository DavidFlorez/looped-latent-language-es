"""GPU guard verifying NVIDIA Blackwell architecture with >= 32 GB VRAM."""

from __future__ import annotations

import torch


def verify_blackwell_gpu(min_vram_gb: float = 32.0) -> tuple[str, float]:
    """Print GPU name and memory; raise RuntimeError if not Blackwell with >= min_vram_gb."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Must run on a Google Colab G4 (Blackwell) GPU.")

    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    vram_gb = props.total_memory / (1024 ** 3)
    cc = f"{props.major}.{props.minor}"
    print(f"[GPU] Name: {gpu_name} | VRAM: {vram_gb:.2f} GB | Compute Capability: sm_{props.major}{props.minor}")

    is_blackwell = ("blackwell" in gpu_name.lower()) or ("rtx pro 6000" in gpu_name.lower()) or ("rtx 5090" in gpu_name.lower()) or (props.major >= 10) or (props.major == 12)
    if not is_blackwell:
        raise RuntimeError(
            f"GPU '{gpu_name}' (sm_{props.major}{props.minor}, {vram_gb:.2f} GB) is not Blackwell. Stopping."
        )
    if vram_gb < min_vram_gb:
        raise RuntimeError(
            f"GPU '{gpu_name}' has {vram_gb:.2f} GB VRAM (< {min_vram_gb:.1f} GB required). Stopping."
        )
    return gpu_name, vram_gb


if __name__ == "__main__":
    verify_blackwell_gpu()
