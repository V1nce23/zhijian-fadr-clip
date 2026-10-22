"""Export the supplied best checkpoint without changing any model tensor.

Run with the inference Python environment. The source release is left untouched.
"""

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SHA256 = "b0ba99d06f0399e544bd45dbff4cdaa48f771d2324e3bacc78007fbff0597a09"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="FADR_CLIP release directory")
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_path = args.source / "weights/fadr_clip_best_main_model.pth"
    if sha256(checkpoint_path) != EXPECTED_SHA256:
        raise ValueError("最佳主模型权重校验不匹配，未导出")
    # Only the known NumPy scalar metadata types in this checked release are allowed.
    with torch.serialization.safe_globals([
        np._core.multiarray.scalar, np.dtype,
        np.dtypes.Float64DType, np.dtypes.Float32DType,
    ]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["model_state_dict"]
    target = args.models / "fadr_clip"
    target.mkdir(parents=True, exist_ok=True)
    weights = target / "model.safetensors"
    save_file(state, str(weights), metadata={"source_checkpoint_sha256": EXPECTED_SHA256})
    exported = load_file(str(weights))
    if set(exported) != set(state) or not all(torch.equal(exported[k], v) for k, v in state.items()):
        raise ValueError("导出权重与原始权重不一致")
    # Discard training paths/optimizer settings; keep every architecture/inference option.
    excluded = {
        "log_dir", "dataset_json_folder", "dataset_root_path", "resolved_run_name",
        "optimizer", "lr_scheduler", "all_dataset", "label_dict", "train_dataset",
        "test_dataset", "generalization_monitor_datasets", "data_aug",
    }
    config = {k: v for k, v in checkpoint["config"].items() if k not in excluded}
    config.update(local_files_only=True, clip_model_name_or_path="local-complete-checkpoint")
    config_path = target / "model_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "model": "FADR-CLIP", "version": "best-b0ba99d0",
        "source_checkpoint_sha256": EXPECTED_SHA256,
        "selection_metric": checkpoint["metric_scoring"],
        "selection_dataset": "FaceForensics++",
        "selection_value": float(checkpoint["best_avg_metric"]),
        "epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"],
        "tensor_count": len(state), "tensor_values_unchanged": True,
        "files": {p.name: {"sha256": sha256(p), "size": p.stat().st_size} for p in (weights, config_path)},
    }
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
