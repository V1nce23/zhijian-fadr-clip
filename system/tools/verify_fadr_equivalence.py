"""Compare the original release's forward pass with the deployed offline adapter."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    import numpy as np
    import torch
    import importlib
    from types import ModuleType
    from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig
    from app.detectors.catalog import detector_specs, FADR_CHECKPOINT_SHA256
    from app.detectors.fadr_clip.adapter import FADRClipDetector

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    deployed = FADRClipDetector(detector_specs("image")[0], args.models, device="cpu")
    deployed.load()
    faces, _, _ = deployed.faces.extract(args.image, "image")
    image = deployed._input_tensor(faces[0].image)
    batch = torch.stack([image, image.flip(-1)])
    with torch.inference_mode():
        actual = deployed.model({"image": batch}, inference=True)
    sys.path.insert(0, str(args.source))
    # Load the original files without executing training-only package initializers
    # (trainer/__init__ imports TensorBoard, which inference does not require).
    for name, relative in (("model", "model"), ("trainer", "trainer"),
                           ("trainer.metrics", "trainer/metrics"), ("utils", "utils")):
        package = ModuleType(name)
        package.__path__ = [str(args.source / relative)]
        sys.modules[name] = package
    original = importlib.import_module("model.clip_lora_classifier")

    # No text weights are used by the supplied classifier. A tiny text tower keeps
    # reference construction inexpensive; its vision architecture is the original L/14.
    def local_backbone(*unused, **kwargs):
        config_path = Path(__file__).resolve().parents[1] / "services/api/app/detectors/fadr_clip/clip_vision_config.json"
        vision = CLIPVisionConfig.from_json_file(str(config_path))
        text = CLIPTextConfig(vocab_size=32, hidden_size=64, intermediate_size=128,
                              num_hidden_layers=1, num_attention_heads=4)
        return CLIPModel(CLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), projection_dim=768))

    with torch.serialization.safe_globals([
        np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType,
    ]):
        checkpoint = torch.load(args.source / "weights/fadr_clip_best_main_model.pth",
                                map_location="cpu", weights_only=True, mmap=True)
    with patch.object(original.CLIPModel, "from_pretrained", side_effect=local_backbone):
        reference = original.CLIPLoRAClassifier(checkpoint["config"])
    reference.load_state_dict(checkpoint["model_state_dict"], strict=True, assign=True)
    reference.eval()
    with torch.inference_mode():
        expected = reference({"image": batch}, inference=True)
    errors = {}
    for key in ("cls", "prob", "feat", "attn_weights"):
        errors[key] = float((actual[key] - expected[key]).abs().max())
        torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-6)
    report = {"source_checkpoint_sha256": FADR_CHECKPOINT_SHA256,
              "state_keys": len(reference.state_dict()), "device": "cpu", "precision": "float32",
              "inputs": "detected/aligned real face and its horizontal flip",
              "reference": "unmodified supplied forward; original best state strictly loaded; local CLIP architecture",
              "max_absolute_errors": errors, "passed": True}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
