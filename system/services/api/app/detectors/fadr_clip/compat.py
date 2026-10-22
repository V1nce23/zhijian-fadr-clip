from copy import deepcopy

import torch

DEFAULT_ALL_DATASETS = [
    "FaceForensics++",
    "FF-F2F",
    "FF-DF",
    "FF-FS",
    "FF-NT",
    "FaceShifter",
    "DeepFakeDetection",
    "Celeb-DF-v1",
    "Celeb-DF-v2",
    "DFDCP",
    "DFDC",
    "DeeperForensics-1.0",
    "UADFV",
]

DEFAULT_LABEL_DICT = {
    "DFD_fake": 1,
    "DFD_real": 0,
    "FF-SH": 1,
    "FF-F2F": 1,
    "FF-DF": 1,
    "FF-FS": 1,
    "FF-NT": 1,
    "FF-FH": 1,
    "FF-real": 0,
    "CelebDFv1_real": 0,
    "CelebDFv1_fake": 1,
    "CelebDFv2_real": 0,
    "CelebDFv2_fake": 1,
    "DFDCP_Real": 0,
    "DFDCP_FakeA": 1,
    "DFDCP_FakeB": 1,
    "DFDC_Fake": 1,
    "DFDC_Real": 0,
    "DF_fake": 1,
    "DF_real": 0,
    "UADFV_Fake": 1,
    "UADFV_Real": 0,
    "roop_Real": 0,
    "roop_Fake": 1,
}


def build_experiment_tag(config):
    """Build a readable tag for the current model variant and feature setup."""
    fusion_type = str(
        config.get("feature_fusion_type", "single_layer_cls")
    ).lower()
    layer_indices = [
        int(layer_index) + 1
        for layer_index in config.get("vision_layer_indices", [])
    ]
    layer_tag = "-".join(f"L{layer_index}" for layer_index in layer_indices)

    if fusion_type == "multi_layer_cls_concat":
        if layer_indices:
            return f"multi_cls_concat_{layer_tag}"
        return "multi_cls_concat"

    if fusion_type == "patch_router_fusion":
        if layer_indices:
            return f"patch_router_{layer_tag}"
        return "patch_router"

    if fusion_type == "frequency_router_fusion":
        if layer_indices:
            return f"frequency_router_{layer_tag}"
        return "frequency_router"

    if fusion_type == "single_layer_cls":
        return "single_cls_last"

    return fusion_type


def build_run_name(config):
    """Compose the per-run folder name placed under logs/."""
    explicit_run_name = str(config.get("run_name", "")).strip()
    if explicit_run_name:
        return explicit_run_name

    base_name = str(config.get("model_name", "clip_lora_baseline")).strip()
    experiment_tag = str(config.get("experiment_tag", "")).strip()
    if not experiment_tag:
        experiment_tag = build_experiment_tag(config)
    if not experiment_tag or base_name.endswith(experiment_tag):
        return base_name
    return f"{base_name}_{experiment_tag}"


def apply_default_config(config):
    """Fill missing config keys with DeepfakeBench-compatible defaults."""
    merged = deepcopy(config)
    merged.setdefault("model_variant", "clip_lora_baseline")
    merged.setdefault("model_name", "clip_lora_baseline")
    merged.setdefault(
        "clip_model_name_or_path",
        "openai/clip-vit-large-patch14",
    )
    merged.setdefault("local_files_only", False)
    merged.setdefault("vision_layer_indices", [5, 11, 17, 23])
    merged.setdefault("num_router_layers", len(merged["vision_layer_indices"]))
    merged.setdefault("feature_fusion_type", "frequency_router_fusion")
    merged.setdefault("router_ablation_mode", "full")
    merged.setdefault("enable_lora", True)
    merged.setdefault("lora_layer_policy", "all")
    merged.setdefault("lora_start_layer", None)
    merged.setdefault("lora_end_layer", None)
    merged.setdefault("lora_layer_indices", None)
    merged.setdefault("lora_target_modules", ["mlp.fc1", "mlp.fc2"])
    merged.setdefault("lora_rank", 16)
    merged.setdefault("lora_alpha", 8)
    merged.setdefault("lora_dropout", 0.0)
    merged.setdefault("descriptor_dim", 768)
    merged.setdefault(
        "descriptor_hidden_dim",
        merged["descriptor_dim"],
    )
    merged.setdefault("descriptor_dropout", 0.1)
    merged.setdefault("patch_attention_hidden_dim", 256)
    merged.setdefault("router_hidden_dim", merged["descriptor_dim"])
    merged.setdefault("router_dropout", 0.1)
    merged.setdefault("router_floor", 0.05)
    merged.setdefault("frequency_extractor_type", "rgb_haar_9d")
    merged.setdefault("frequency_hidden_dim", 128)
    merged.setdefault("frequency_dropout", 0.1)
    merged.setdefault("enable_route_supervision", False)
    merged.setdefault("route_supervision_tau", 0.7)
    merged.setdefault("route_aux_loss_weight", 0.2)
    merged.setdefault("route_kl_loss_weight", 0.1)
    merged.setdefault("ortho_loss_weight", 0.0)
    merged.setdefault("return_attn_weights", False)
    merged.setdefault("enable_dgso", False)
    merged.setdefault(
        "dgso_shared_modules",
        ["vision_lora"],
    )
    merged.setdefault("all_dataset", list(DEFAULT_ALL_DATASETS))
    merged.setdefault("label_dict", deepcopy(DEFAULT_LABEL_DICT))
    merged.setdefault("frame_num", {"train": 32, "test": 32})
    merged.setdefault("resolution", 256)
    merged.setdefault("clip_input_resolution", 224)
    merged.setdefault("prefetch_factor", 2)
    merged.setdefault("persistent_workers", True)
    merged.setdefault("mean", [0.48145466, 0.4578275, 0.40821073])
    merged.setdefault("std", [0.26862954, 0.26130258, 0.27577711])
    merged.setdefault("save_ckpt", True)
    merged.setdefault("save_latest_ckpt", False)
    merged.setdefault("save_feat", False)
    merged.setdefault("manualSeed", 42)
    merged.setdefault("metric_scoring", "auc")
    merged.setdefault("use_data_augmentation", True)
    merged.setdefault("enable_generalization_monitor", True)
    merged.setdefault(
        "generalization_monitor_datasets",
        ["Celeb-DF-v1", "Celeb-DF-v2", "DFDC", "DFDCP"],
    )
    merged.setdefault("cuda", True)
    merged.setdefault("cudnn", True)
    merged.setdefault("allow_tf32", True)
    merged.setdefault("float32_matmul_precision", "high")
    return merged


def binarize_labels(labels):
    """Map any non-zero label to 1 to match DeepfakeBench binary metrics."""
    return torch.where(labels != 0, 1, 0)
