import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace

from transformers import CLIPVisionConfig, CLIPVisionModel

from .vision_lora import VisionLoRA
from .metrics import calculate_metrics_for_train
from .compat import apply_default_config, binarize_labels


class AttentionPooling(nn.Module):
    """Attention pooling over patch tokens to focus on suspicious regions."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("Attention pooling hidden_dim must be positive.")

        self.scorer = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, patch_tokens):
        attention_scores = self.scorer(patch_tokens).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        pooled_feature = torch.sum(
            patch_tokens * attention_weights.unsqueeze(-1),
            dim=1,
        )
        return pooled_feature, attention_weights


class PatchAwareLayerDescriptor(nn.Module):
    """Build per-layer descriptors from mean and attention pooled patches."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        dropout=0.1,
        attention_hidden_dim=None,
    ):
        super().__init__()
        attention_hidden_dim = (
            max(int(input_dim) // 4, 1)
            if attention_hidden_dim is None
            else int(attention_hidden_dim)
        )
        if int(hidden_dim) <= 0:
            raise ValueError("Descriptor hidden_dim must be positive.")
        if int(output_dim) <= 0:
            raise ValueError("Descriptor output_dim must be positive.")

        self.attention_pool = AttentionPooling(
            input_dim=input_dim,
            hidden_dim=attention_hidden_dim,
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(int(input_dim) * 2),
            nn.Linear(int(input_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, patch_tokens):
        if patch_tokens.ndim != 4:
            raise ValueError(
                "PatchAwareLayerDescriptor expects patch tokens shaped "
                "(B, L, P, D)."
            )

        batch_size, num_layers, num_patches, hidden_dim = patch_tokens.shape
        flattened_tokens = patch_tokens.reshape(
            batch_size * num_layers,
            num_patches,
            hidden_dim,
        )
        global_feature = flattened_tokens.mean(dim=1)
        attentive_feature, patch_attention_weights = self.attention_pool(
            flattened_tokens
        )
        combined_feature = torch.cat(
            [global_feature, attentive_feature],
            dim=-1,
        )
        layer_descriptors = self.projector(combined_feature).reshape(
            batch_size,
            num_layers,
            -1,
        )
        patch_attention_weights = patch_attention_weights.reshape(
            batch_size,
            num_layers,
            num_patches,
        )
        return layer_descriptors, patch_attention_weights


class DepthRouter(nn.Module):
    """Predict image-adaptive layer weights with a routing floor."""

    def __init__(
        self,
        num_layers,
        descriptor_dim,
        hidden_dim,
        dropout=0.1,
        routing_floor=0.05,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.descriptor_dim = int(descriptor_dim)
        self.routing_floor = float(routing_floor)

        if self.num_layers <= 0:
            raise ValueError("DepthRouter num_layers must be positive.")
        if int(hidden_dim) <= 0:
            raise ValueError("DepthRouter hidden_dim must be positive.")
        if not 0.0 <= self.routing_floor < 1.0:
            raise ValueError("routing_floor must be in [0.0, 1.0).")

        input_dim = self.num_layers * self.descriptor_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_layers),
        )

    def forward(self, layer_descriptors):
        if layer_descriptors.ndim != 3:
            raise ValueError(
                "DepthRouter expects layer descriptors shaped (B, L, D)."
            )

        flattened_descriptors = layer_descriptors.reshape(
            layer_descriptors.shape[0],
            -1,
        )
        routing_logits = self.mlp(flattened_descriptors)
        routing_weights = torch.softmax(routing_logits, dim=-1)
        if self.routing_floor > 0:
            routing_weights = (
                (1.0 - self.routing_floor) * routing_weights
                + (self.routing_floor / self.num_layers)
            )
        return routing_weights, routing_logits


class HaarWaveletEnergyExtractor(nn.Module):
    """Compute a 9D RGB-aware Haar high-frequency energy fingerprint."""

    def __init__(self):
        super().__init__()
        base_kernels = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1) / 2.0
        self.register_buffer(
            "weight",
            base_kernels.repeat(3, 1, 1, 1),
        )
        self.output_dim = 9

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "HaarWaveletEnergyExtractor expects images shaped "
                "(B, 3, H, W)."
            )

        wavelet_response = F.conv2d(
            image,
            self.weight.to(dtype=image.dtype),
            stride=2,
            groups=3,
        )
        batch_size, _, height, width = wavelet_response.shape
        wavelet_response = wavelet_response.reshape(
            batch_size,
            3,
            4,
            height,
            width,
        )
        hl_energy = wavelet_response[:, :, 1].abs().mean(dim=(-2, -1))
        lh_energy = wavelet_response[:, :, 2].abs().mean(dim=(-2, -1))
        hh_energy = wavelet_response[:, :, 3].abs().mean(dim=(-2, -1))
        return torch.cat([hh_energy, hl_energy, lh_energy], dim=1)


class GrayHaarWaveletEnergyExtractor(nn.Module):
    """Compute a 3D grayscale Haar high-frequency energy fingerprint."""

    def __init__(self):
        super().__init__()
        base_kernels = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1) / 2.0
        rgb_to_gray = torch.tensor(
            [0.2989, 0.5870, 0.1140],
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self.register_buffer("rgb_to_gray", rgb_to_gray)
        self.register_buffer("weight", base_kernels)
        self.output_dim = 3

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "GrayHaarWaveletEnergyExtractor expects images shaped "
                "(B, 3, H, W)."
            )

        gray_image = torch.sum(
            image * self.rgb_to_gray.to(dtype=image.dtype),
            dim=1,
            keepdim=True,
        )
        wavelet_response = F.conv2d(
            gray_image,
            self.weight.to(dtype=image.dtype),
            stride=2,
        )
        hl_energy = wavelet_response[:, 1].abs().mean(dim=(-2, -1))
        lh_energy = wavelet_response[:, 2].abs().mean(dim=(-2, -1))
        hh_energy = wavelet_response[:, 3].abs().mean(dim=(-2, -1))
        return torch.stack([hh_energy, hl_energy, lh_energy], dim=1)


class RGBDCTHighFrequencyExtractor(nn.Module):
    """Compute a 9D RGB block-DCT high-frequency energy fingerprint."""

    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = int(block_size)
        if self.block_size <= 1:
            raise ValueError("DCT block_size must be greater than 1.")
        dct_matrix = self._build_dct_matrix(self.block_size)
        row_indices = torch.arange(self.block_size).view(-1, 1)
        col_indices = torch.arange(self.block_size).view(1, -1)
        frequency_order = row_indices + col_indices
        self.register_buffer("dct_matrix", dct_matrix)
        self.register_buffer(
            "mid_mask",
            (frequency_order >= 4) & (frequency_order < 8),
        )
        self.register_buffer(
            "high_mask",
            (frequency_order >= 8) & (frequency_order < 12),
        )
        self.register_buffer("ultra_mask", frequency_order >= 12)
        self.output_dim = 9

    @staticmethod
    def _build_dct_matrix(block_size):
        positions = torch.arange(block_size, dtype=torch.float32)
        frequencies = torch.arange(block_size, dtype=torch.float32).view(-1, 1)
        matrix = torch.cos(
            math.pi
            * (positions + 0.5).view(1, -1)
            * frequencies
            / float(block_size)
        )
        matrix[0] *= math.sqrt(1.0 / float(block_size))
        matrix[1:] *= math.sqrt(2.0 / float(block_size))
        return matrix

    def _band_energy(self, dct_coefficients, mask):
        masked_coefficients = dct_coefficients[..., mask]
        return masked_coefficients.abs().mean(dim=-1).mean(dim=1)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "RGBDCTHighFrequencyExtractor expects images shaped "
                "(B, 3, H, W)."
            )

        block_size = self.block_size
        height = image.shape[-2] - (image.shape[-2] % block_size)
        width = image.shape[-1] - (image.shape[-1] % block_size)
        if height <= 0 or width <= 0:
            raise ValueError(
                "Input image is too small for block-DCT frequency extraction."
            )

        cropped_image = image[..., :height, :width]
        blocks = F.unfold(
            cropped_image,
            kernel_size=block_size,
            stride=block_size,
        )
        batch_size = image.shape[0]
        blocks = blocks.transpose(1, 2).reshape(
            batch_size,
            -1,
            3,
            block_size,
            block_size,
        )
        dct_matrix = self.dct_matrix.to(
            device=image.device,
            dtype=image.dtype,
        )
        dct_coefficients = torch.matmul(
            torch.matmul(dct_matrix, blocks),
            dct_matrix.t(),
        )
        mid_energy = self._band_energy(dct_coefficients, self.mid_mask)
        high_energy = self._band_energy(dct_coefficients, self.high_mask)
        ultra_energy = self._band_energy(dct_coefficients, self.ultra_mask)
        return torch.cat([mid_energy, high_energy, ultra_energy], dim=1)


def build_frequency_extractor(extractor_type):
    extractor_type = str(extractor_type or "rgb_haar_9d").lower()
    aliases = {
        "rgb_haar_9d": HaarWaveletEnergyExtractor,
        "haar_rgb_9d": HaarWaveletEnergyExtractor,
        "dwt_rgb_9d": HaarWaveletEnergyExtractor,
        "rgb_dwt_9d": HaarWaveletEnergyExtractor,
        "gray_haar_3d": GrayHaarWaveletEnergyExtractor,
        "haar_gray_3d": GrayHaarWaveletEnergyExtractor,
        "dwt_gray_3d": GrayHaarWaveletEnergyExtractor,
        "gray_dwt_3d": GrayHaarWaveletEnergyExtractor,
        "rgb_dct_high_9d": RGBDCTHighFrequencyExtractor,
        "dct_rgb_high_9d": RGBDCTHighFrequencyExtractor,
        "rgb_dct_9d": RGBDCTHighFrequencyExtractor,
        "dct_rgb_9d": RGBDCTHighFrequencyExtractor,
    }
    if extractor_type not in aliases:
        raise ValueError(
            "Unsupported frequency_extractor_type: "
            f"{extractor_type}. Supported values: {sorted(aliases)}"
        )
    return aliases[extractor_type]()


class FrequencyRouter(nn.Module):
    """Route CLIP layers using a frequency prior and layer descriptors."""

    def __init__(
        self,
        input_dim,
        num_layers,
        descriptor_dim,
        hidden_dim,
        dropout=0.1,
        routing_floor=0.05,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_layers = int(num_layers)
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.routing_floor = float(routing_floor)

        if self.input_dim <= 0:
            raise ValueError("FrequencyRouter input_dim must be positive.")
        if self.num_layers <= 0:
            raise ValueError("FrequencyRouter num_layers must be positive.")
        if self.descriptor_dim <= 0:
            raise ValueError(
                "FrequencyRouter descriptor_dim must be positive."
            )
        if self.hidden_dim <= 0:
            raise ValueError("FrequencyRouter hidden_dim must be positive.")
        if not 0.0 <= self.routing_floor < 1.0:
            raise ValueError("routing_floor must be in [0.0, 1.0).")

        self.prior_encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.descriptor_dim),
        )
        self.key_encoder = nn.Sequential(
            nn.LayerNorm(self.descriptor_dim),
            nn.Linear(self.descriptor_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.descriptor_dim),
        )
        self.content_scorer = nn.Sequential(
            nn.LayerNorm(self.descriptor_dim),
            nn.Linear(self.descriptor_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, frequency_feature, layer_descriptors):
        if frequency_feature.ndim != 2:
            raise ValueError(
                "FrequencyRouter expects features shaped (B, D)."
            )
        if layer_descriptors.ndim != 3:
            raise ValueError(
                "FrequencyRouter expects layer descriptors shaped (B, L, D)."
            )
        if layer_descriptors.shape[1] != self.num_layers:
            raise ValueError(
                "FrequencyRouter layer_descriptors shape mismatch: "
                f"expected {self.num_layers} layers, got "
                f"{layer_descriptors.shape[1]}."
            )
        if layer_descriptors.shape[2] != self.descriptor_dim:
            raise ValueError(
                "FrequencyRouter layer_descriptors last dim mismatch: "
                f"expected {self.descriptor_dim}, got "
                f"{layer_descriptors.shape[2]}."
            )

        prior_query = self.prior_encoder(frequency_feature)
        layer_keys = self.key_encoder(layer_descriptors)
        prior_scores = torch.einsum(
            "bd,bld->bl",
            prior_query,
            layer_keys,
        ) / math.sqrt(self.descriptor_dim)
        content_scores = self.content_scorer(layer_descriptors).squeeze(-1)
        routing_logits = prior_scores + content_scores
        routing_weights = torch.softmax(routing_logits, dim=-1)
        if self.routing_floor > 0:
            routing_weights = (
                (1.0 - self.routing_floor) * routing_weights
                + (self.routing_floor / self.num_layers)
            )
        return routing_weights, routing_logits


class ContentOnlyRouter(nn.Module):
    """Predict sample-wise layer weights from layer descriptors only."""

    def __init__(
        self,
        num_layers,
        descriptor_dim,
        hidden_dim,
        dropout=0.1,
        routing_floor=0.05,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.routing_floor = float(routing_floor)

        if self.num_layers <= 0:
            raise ValueError("ContentOnlyRouter num_layers must be positive.")
        if self.descriptor_dim <= 0:
            raise ValueError(
                "ContentOnlyRouter descriptor_dim must be positive."
            )
        if self.hidden_dim <= 0:
            raise ValueError("ContentOnlyRouter hidden_dim must be positive.")
        if not 0.0 <= self.routing_floor < 1.0:
            raise ValueError("routing_floor must be in [0.0, 1.0).")

        self.content_scorer = nn.Sequential(
            nn.LayerNorm(self.descriptor_dim),
            nn.Linear(self.descriptor_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, layer_descriptors):
        if layer_descriptors.ndim != 3:
            raise ValueError(
                "ContentOnlyRouter expects layer descriptors shaped "
                "(B, L, D)."
            )
        if layer_descriptors.shape[1] != self.num_layers:
            raise ValueError(
                "ContentOnlyRouter layer_descriptors shape mismatch: "
                f"expected {self.num_layers} layers, got "
                f"{layer_descriptors.shape[1]}."
            )
        if layer_descriptors.shape[2] != self.descriptor_dim:
            raise ValueError(
                "ContentOnlyRouter layer_descriptors last dim mismatch: "
                f"expected {self.descriptor_dim}, got "
                f"{layer_descriptors.shape[2]}."
            )

        routing_logits = self.content_scorer(layer_descriptors).squeeze(-1)
        routing_weights = torch.softmax(routing_logits, dim=-1)
        if self.routing_floor > 0:
            routing_weights = (
                (1.0 - self.routing_floor) * routing_weights
                + (self.routing_floor / self.num_layers)
            )
        return routing_weights, routing_logits


class StaticLearnedLayerRouter(nn.Module):
    """Use one learned layer-weight distribution shared by all samples."""

    def __init__(self, num_layers, routing_floor=0.05):
        super().__init__()
        self.num_layers = int(num_layers)
        self.routing_floor = float(routing_floor)
        if self.num_layers <= 0:
            raise ValueError(
                "StaticLearnedLayerRouter num_layers must be positive."
            )
        if not 0.0 <= self.routing_floor < 1.0:
            raise ValueError("routing_floor must be in [0.0, 1.0).")
        self.static_layer_logits = nn.Parameter(torch.zeros(self.num_layers))

    def forward(self, reference_tensor):
        if not torch.is_tensor(reference_tensor) or reference_tensor.ndim < 1:
            raise ValueError(
                "StaticLearnedLayerRouter expects a tensor with batch "
                "dimension as reference."
            )

        batch_size = int(reference_tensor.shape[0])
        routing_logits = self.static_layer_logits.to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )
        routing_weights = torch.softmax(routing_logits, dim=-1)
        if self.routing_floor > 0:
            routing_weights = (
                (1.0 - self.routing_floor) * routing_weights
                + (self.routing_floor / self.num_layers)
            )
        return (
            routing_weights.unsqueeze(0).expand(batch_size, -1),
            routing_logits.unsqueeze(0).expand(batch_size, -1),
        )


class UniformMeanRouter(nn.Module):
    """Return fixed uniform layer weights without trainable parameters."""

    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError("UniformMeanRouter num_layers must be positive.")

    def forward(self, reference_tensor):
        if not torch.is_tensor(reference_tensor) or reference_tensor.ndim < 1:
            raise ValueError(
                "UniformMeanRouter expects a tensor with batch dimension as "
                "reference."
            )

        batch_size = int(reference_tensor.shape[0])
        routing_weights = reference_tensor.new_full(
            (batch_size, self.num_layers),
            1.0 / self.num_layers,
        )
        routing_logits = reference_tensor.new_zeros(
            (batch_size, self.num_layers)
        )
        return routing_weights, routing_logits


class LayerExpertHead(nn.Module):
    """A lightweight per-layer probe used for route supervision."""

    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), 2),
        )

    def forward(self, inputs):
        return self.classifier(inputs)


class CLIPLoRAClassifier(nn.Module):
    """Vision-only CLIP baseline with a frozen backbone and LoRA tuning."""

    SUPPORTED_FEATURE_FUSION_TYPES = {
        "single_layer_cls",
        "multi_layer_cls_concat",
        "patch_router_fusion",
        "frequency_router_fusion",
    }
    SUPPORTED_ROUTER_ABLATION_MODES = {
        "full",
        "content_only",
        "single_layer",
        "static_learned",
        "uniform_mean",
    }

    def __init__(self, config):
        super().__init__()
        self.config = apply_default_config(config)
        self.model_variant = "clip_lora_baseline"
        self.clip_model_name_or_path = self.config["clip_model_name_or_path"]
        self.ortho_loss_weight = float(
            self.config.get("ortho_loss_weight", 0.0)
        )
        self.return_attn_weights = bool(
            self.config.get("return_attn_weights", False)
        )
        self.feature_fusion_type = self._resolve_feature_fusion_type()
        self.router_ablation_mode = self._resolve_router_ablation_mode()

        clip_model = self.build_backbone(self.config)
        self.loss_func = self.build_loss(self.config)
        self.vision_branch = VisionLoRA(
            vision_model=clip_model.vision_model,
            layer_indices=self.config.get("vision_layer_indices"),
            num_router_layers=self.config.get("num_router_layers", 4),
            lora_layer_indices=self.config.get("lora_layer_indices"),
            lora_layer_policy=self.config.get(
                "lora_layer_policy",
                "all",
            ),
            lora_start_layer=self.config.get("lora_start_layer"),
            lora_end_layer=self.config.get("lora_end_layer"),
            lora_rank=self.config["lora_rank"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            lora_train_bias=self.config.get("lora_train_bias", False),
            lora_target_modules=self.config.get("lora_target_modules"),
            enable_lora=self.config.get("enable_lora", True),
        )
        self.visual_projection = clip_model.visual_projection
        self._freeze_module(self.visual_projection)

        self.num_selected_layers = len(self.vision_branch.layer_indices)
        self.projected_embed_dim = int(self.visual_projection.out_features)
        mean = torch.tensor(
            self.config.get("mean", [0.48145466, 0.4578275, 0.40821073]),
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.config.get("std", [0.26862954, 0.26130258, 0.27577711]),
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)
        descriptor_dim = self.config.get("descriptor_dim")
        self.router_descriptor_dim = int(
            self.projected_embed_dim
            if descriptor_dim is None
            else descriptor_dim
        )
        self.enable_route_supervision = bool(
            self.config.get("enable_route_supervision", False)
        )
        self.route_supervision_tau = float(
            self.config.get("route_supervision_tau", 1.0)
        )
        self.route_aux_loss_weight = float(
            self.config.get("route_aux_loss_weight", 0.2)
        )
        self.route_kl_loss_weight = float(
            self.config.get("route_kl_loss_weight", 0.1)
        )
        if self.route_supervision_tau <= 0:
            raise ValueError("route_supervision_tau must be positive.")
        if self.route_aux_loss_weight < 0:
            raise ValueError("route_aux_loss_weight must be non-negative.")
        if self.route_kl_loss_weight < 0:
            raise ValueError("route_kl_loss_weight must be non-negative.")
        self.layer_descriptor = None
        self.frequency_extractor = None
        self.router = None
        self.layer_expert_heads = None
        if self.feature_fusion_type == "patch_router_fusion":
            self.layer_descriptor = PatchAwareLayerDescriptor(
                input_dim=self.vision_branch.hidden_size,
                output_dim=self.router_descriptor_dim,
                hidden_dim=int(
                    self.config.get(
                        "descriptor_hidden_dim",
                        self.router_descriptor_dim,
                    )
                ),
                dropout=float(self.config.get("descriptor_dropout", 0.1)),
                attention_hidden_dim=self.config.get(
                    "patch_attention_hidden_dim"
                ),
            )
            self.router = DepthRouter(
                num_layers=self.num_selected_layers,
                descriptor_dim=self.router_descriptor_dim,
                hidden_dim=int(
                    self.config.get(
                        "router_hidden_dim",
                        self.router_descriptor_dim,
                    )
                ),
                dropout=float(self.config.get("router_dropout", 0.1)),
                routing_floor=float(self.config.get("router_floor", 0.05)),
            )
        elif self.feature_fusion_type == "frequency_router_fusion":
            router_hidden_dim = int(
                self.config.get(
                    "frequency_hidden_dim",
                    128,
                )
            )
            router_dropout = float(
                self.config.get("frequency_dropout", 0.1)
            )
            routing_floor = float(self.config.get("router_floor", 0.05))
            if self.router_ablation_mode == "full":
                self.layer_descriptor = PatchAwareLayerDescriptor(
                    input_dim=self.vision_branch.hidden_size,
                    output_dim=self.router_descriptor_dim,
                    hidden_dim=int(
                        self.config.get(
                            "descriptor_hidden_dim",
                            self.router_descriptor_dim,
                        )
                    ),
                    dropout=float(self.config.get("descriptor_dropout", 0.1)),
                    attention_hidden_dim=self.config.get(
                        "patch_attention_hidden_dim"
                    ),
                )
                self.frequency_extractor = build_frequency_extractor(
                    self.config.get("frequency_extractor_type", "rgb_haar_9d")
                )
                self.router = FrequencyRouter(
                    input_dim=self.frequency_extractor.output_dim,
                    num_layers=self.num_selected_layers,
                    descriptor_dim=self.router_descriptor_dim,
                    hidden_dim=router_hidden_dim,
                    dropout=router_dropout,
                    routing_floor=routing_floor,
                )
            elif self.router_ablation_mode == "content_only":
                self.layer_descriptor = PatchAwareLayerDescriptor(
                    input_dim=self.vision_branch.hidden_size,
                    output_dim=self.router_descriptor_dim,
                    hidden_dim=int(
                        self.config.get(
                            "descriptor_hidden_dim",
                            self.router_descriptor_dim,
                        )
                    ),
                    dropout=float(self.config.get("descriptor_dropout", 0.1)),
                    attention_hidden_dim=self.config.get(
                        "patch_attention_hidden_dim"
                    ),
                )
                self.router = ContentOnlyRouter(
                    num_layers=self.num_selected_layers,
                    descriptor_dim=self.router_descriptor_dim,
                    hidden_dim=router_hidden_dim,
                    dropout=router_dropout,
                    routing_floor=routing_floor,
                )
            elif self.router_ablation_mode == "single_layer":
                if self.num_selected_layers != 1:
                    raise ValueError(
                        "router_ablation_mode='single_layer' requires exactly "
                        "one vision_layer_indices entry."
                    )
                self.enable_route_supervision = False
                self.route_aux_loss_weight = 0.0
                self.route_kl_loss_weight = 0.0
            elif self.router_ablation_mode == "static_learned":
                self.router = StaticLearnedLayerRouter(
                    num_layers=self.num_selected_layers,
                    routing_floor=routing_floor,
                )
            elif self.router_ablation_mode == "uniform_mean":
                self.router = UniformMeanRouter(
                    num_layers=self.num_selected_layers
                )
                self.route_kl_loss_weight = 0.0
            if (
                self.enable_route_supervision
                and self.router_ablation_mode not in {
                    "single_layer",
                    "uniform_mean",
                }
            ):
                self.layer_expert_heads = nn.ModuleList(
                    LayerExpertHead(self.vision_branch.hidden_size)
                    for _ in range(self.num_selected_layers)
                )
        self.classifier_input_dim = self._resolve_classifier_input_dim()
        self.embed_dim = self.classifier_input_dim
        self.cls_head = self._build_classifier_head()

    def build_backbone(self, config):
        # The best checkpoint contains the complete vision tower and projection.
        # Construct only that architecture locally; strict loading fills every tensor.
        vision_config = CLIPVisionConfig.from_json_file(
            str(Path(__file__).with_name("clip_vision_config.json"))
        )
        return SimpleNamespace(
            vision_model=CLIPVisionModel(vision_config).vision_model,
            visual_projection=nn.Linear(
                vision_config.hidden_size, vision_config.projection_dim, bias=False
            ),
        )

    @staticmethod
    def _freeze_module(module):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    def build_loss(self, config):
        return nn.CrossEntropyLoss()

    def _resolve_feature_fusion_type(self):
        fusion_type = str(
            self.config.get(
                "feature_fusion_type",
                "frequency_router_fusion",
            )
        ).lower()
        if fusion_type not in self.SUPPORTED_FEATURE_FUSION_TYPES:
            raise ValueError(
                "Unsupported feature_fusion_type: "
                f"{self.config.get('feature_fusion_type')}"
            )
        return fusion_type

    def _resolve_router_ablation_mode(self):
        mode = str(self.config.get("router_ablation_mode", "full")).lower()
        if mode not in self.SUPPORTED_ROUTER_ABLATION_MODES:
            raise ValueError(
                "Unsupported router_ablation_mode: "
                f"{self.config.get('router_ablation_mode')}. Supported modes: "
                f"{sorted(self.SUPPORTED_ROUTER_ABLATION_MODES)}"
            )
        if (
            mode != "full"
            and str(
                self.config.get(
                    "feature_fusion_type",
                    "frequency_router_fusion",
                )
            ).lower()
            != "frequency_router_fusion"
        ):
            raise ValueError(
                "router_ablation_mode other than 'full' is only supported "
                "for feature_fusion_type='frequency_router_fusion'."
            )
        return mode

    def _resolve_classifier_input_dim(self):
        if self.feature_fusion_type == "single_layer_cls":
            return self.projected_embed_dim
        if self.feature_fusion_type == "patch_router_fusion":
            return self.router_descriptor_dim
        if self.feature_fusion_type == "frequency_router_fusion":
            return self.vision_branch.hidden_size

        return self.projected_embed_dim * self.num_selected_layers

    def _build_classifier_head(self):
        hidden_dim = int(self.config.get("classifier_hidden_dim", 512))
        dropout = float(self.config.get("classifier_dropout", 0.1))
        return nn.Sequential(
            nn.LayerNorm(self.classifier_input_dim),
            nn.Linear(self.classifier_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _extract_features_and_weights(self, data_dict):
        image = data_dict["image"] if isinstance(data_dict, dict) else data_dict
        if self.feature_fusion_type == "single_layer_cls":
            cls_token = self.vision_branch.forward_cls_token(image)
            return self.visual_projection(cls_token), None, {}

        if self.feature_fusion_type == "multi_layer_cls_concat":
            cls_tokens = self.vision_branch.forward_multi_cls(image)
            projected_tokens = self.visual_projection(cls_tokens)
            return projected_tokens.flatten(start_dim=1), None, {}

        if self.feature_fusion_type == "frequency_router_fusion":
            return self._extract_frequency_routed_features(image)

        layer_patch_tokens = self.vision_branch.forward_multi_patch(image)
        layer_descriptors, _ = self.layer_descriptor(layer_patch_tokens)
        routing_weights, _ = self.router(layer_descriptors)
        fused_feature = torch.sum(
            layer_descriptors * routing_weights.unsqueeze(-1),
            dim=1,
        )
        return fused_feature, routing_weights, {}

    def _denormalize_image(self, image):
        return torch.clamp(
            image * self.image_std.to(dtype=image.dtype)
            + self.image_mean.to(dtype=image.dtype),
            min=0.0,
            max=1.0,
        )

    def _extract_frequency_routed_features(self, image):
        if (
            self.router_ablation_mode == "full"
            and self.frequency_extractor is None
        ):
            raise RuntimeError(
                "Frequency extraction modules are not initialized for "
                "frequency_router_fusion."
            )
        if (
            self.router_ablation_mode in {"full", "content_only"}
            and self.layer_descriptor is None
        ):
            raise RuntimeError(
                "Layer descriptor is not initialized for "
                "frequency_router_fusion."
            )

        layer_patch_tokens = self.vision_branch.forward_multi_patch(image)
        layer_features = layer_patch_tokens.mean(dim=2)
        if self.router_ablation_mode == "full":
            layer_descriptors, _ = self.layer_descriptor(layer_patch_tokens)
            restored_image = self._denormalize_image(image)
            frequency_feature = self.frequency_extractor(restored_image)
            routing_weights, _ = self.router(
                frequency_feature,
                layer_descriptors,
            )
        elif self.router_ablation_mode == "content_only":
            layer_descriptors, _ = self.layer_descriptor(layer_patch_tokens)
            routing_weights, _ = self.router(layer_descriptors)
        elif self.router_ablation_mode == "single_layer":
            return (
                layer_features[:, 0, :],
                None,
                {},
            )
        elif self.router_ablation_mode == "static_learned":
            routing_weights, _ = self.router(layer_features)
        elif self.router_ablation_mode == "uniform_mean":
            routing_weights, _ = self.router(layer_features)
        else:
            raise RuntimeError(
                f"Unsupported router_ablation_mode: {self.router_ablation_mode}"
            )
        fused_feature = torch.sum(
            layer_features * routing_weights.unsqueeze(-1),
            dim=1,
        )
        return (
            fused_feature,
            routing_weights,
            {"layer_features": layer_features},
        )

    def _compute_layer_logits(self, layer_features):
        if self.layer_expert_heads is None:
            return None
        if layer_features.ndim != 3:
            raise ValueError(
                "LayerExpertHead expects features shaped (B, L, D)."
            )
        logits = [
            head(layer_features[:, layer_index, :])
            for layer_index, head in enumerate(self.layer_expert_heads)
        ]
        return torch.stack(logits, dim=1)

    def features(self, data_dict):
        features, _, _ = self._extract_features_and_weights(data_dict)
        return features

    def classifier(self, features):
        return self.cls_head(features)

    def forward(self, data_dict, inference=False):
        pooled_feature, routing_weights, aux_outputs = self._extract_features_and_weights(
            data_dict
        )
        logits = self.classifier(pooled_feature)
        layer_logits = None
        if (
            not inference
            and self.layer_expert_heads is not None
            and "layer_features" in aux_outputs
        ):
            layer_logits = self._compute_layer_logits(
                aux_outputs["layer_features"]
            )
        probability = torch.softmax(logits, dim=1)[:, 1]
        loss_ortho = pooled_feature.new_zeros(())

        return {
            "preds": logits,
            "cls": logits,
            "prob": probability,
            "feat": pooled_feature,
            "loss_ortho": loss_ortho,
            "attn_weights": routing_weights,
            "layer_logits": layer_logits,
        }

    def get_losses(self, data_dict, pred_dict):
        label = binarize_labels(data_dict["label"])
        logits = pred_dict["preds"]
        loss_ce = self.loss_func(logits, label)
        loss_ortho = logits.new_zeros(())
        loss_aux_ce = logits.new_zeros(())
        loss_route = logits.new_zeros(())

        layer_logits = pred_dict.get("layer_logits")
        routing_weights = pred_dict.get("attn_weights")
        if (
            self.layer_expert_heads is not None
            and torch.is_tensor(layer_logits)
            and torch.is_tensor(routing_weights)
        ):
            if layer_logits.ndim != 3:
                raise ValueError(
                    "layer_logits must be shaped (B, L, C) for route "
                    "supervision."
                )
            expanded_labels = label.unsqueeze(1).expand(
                -1,
                layer_logits.shape[1],
            )
            per_layer_loss = F.cross_entropy(
                layer_logits.reshape(-1, layer_logits.shape[-1]),
                expanded_labels.reshape(-1),
                reduction="none",
            ).reshape(layer_logits.shape[0], layer_logits.shape[1])
            if self.route_aux_loss_weight > 0:
                loss_aux_ce = per_layer_loss.mean()
            if self.route_kl_loss_weight > 0:
                teacher_distribution = torch.softmax(
                    -per_layer_loss.detach() / self.route_supervision_tau,
                    dim=1,
                )
                student_distribution = routing_weights.clamp_min(1e-8)
                loss_route = F.kl_div(
                    student_distribution.log(),
                    teacher_distribution,
                    reduction="batchmean",
                )

        overall_loss = (
            loss_ce
            + (self.route_aux_loss_weight * loss_aux_ce)
            + (self.route_kl_loss_weight * loss_route)
        )
        return {
            "overall": overall_loss,
            "loss_ce": loss_ce,
            "loss_aux_ce": loss_aux_ce,
            "loss_route": loss_route,
            "loss_ortho": loss_ortho,
        }

    def get_train_metrics(self, data_dict, pred_dict):
        label = binarize_labels(data_dict["label"])
        pred = pred_dict["preds"]
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(),
            pred.detach(),
        )
        return {
            "acc": acc,
            "auc": auc,
            "eer": eer,
            "ap": ap,
        }

    def get_dgso_shared_parameters(self):
        module_names = self.config.get(
            "dgso_shared_modules",
            ["vision_lora"],
        )
        params = []
        seen = set()

        if "vision_lora" in module_names:
            params.extend(self.vision_branch.lora_parameters())
        if "router" in module_names and self.router is not None:
            params.extend(self.router.parameters())
        if (
            "descriptor" in module_names
            and self.layer_descriptor is not None
        ):
            params.extend(self.layer_descriptor.parameters())
        if "classifier" in module_names:
            params.extend(self.cls_head.parameters())
            if self.layer_expert_heads is not None:
                params.extend(self.layer_expert_heads.parameters())

        unique_params = []
        for parameter in params:
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            unique_params.append(parameter)
        return unique_params
