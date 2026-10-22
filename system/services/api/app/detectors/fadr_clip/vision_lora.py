import numpy as np
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(
        self,
        base_linear,
        rank=16,
        alpha=8,
        dropout=0.0,
        train_bias=False,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.base_layer = base_linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = bool(train_bias)

        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base_linear.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base_linear.out_features, self.rank)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

    def forward(self, inputs):
        base_output = self.base_layer(inputs)
        lora_output = self.dropout(inputs) @ self.lora_A.t()
        lora_output = lora_output @ self.lora_B.t()
        return base_output + lora_output * self.scaling


class VisionLoRA(nn.Module):
    """Frozen CLIP vision tower with LoRA injected into selected linears."""

    def __init__(
        self,
        vision_model,
        layer_indices=None,
        num_router_layers=4,
        lora_layer_indices=None,
        lora_layer_policy="latter_half",
        lora_start_layer=None,
        lora_end_layer=None,
        lora_rank=16,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_train_bias=False,
        lora_target_modules=None,
        enable_lora=True,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.enable_lora = bool(enable_lora)
        self.hidden_size = int(self.vision_model.config.hidden_size)
        self.num_layers = int(self.vision_model.config.num_hidden_layers)
        self.image_size = int(self.vision_model.config.image_size)
        self.patch_size = int(self.vision_model.config.patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.patches_per_layer = self.grid_size * self.grid_size

        self._freeze_vision_tower()
        self.layer_indices = self._resolve_layer_indices(
            layer_indices,
            num_router_layers,
        )
        if self.enable_lora:
            self.lora_layer_indices = self._resolve_lora_layer_indices(
                lora_layer_indices=lora_layer_indices,
                lora_layer_policy=lora_layer_policy,
                lora_start_layer=lora_start_layer,
                lora_end_layer=lora_end_layer,
            )
            self.lora_target_modules = self._resolve_lora_target_modules(
                lora_target_modules
            )
            self._inject_lora(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                train_bias=lora_train_bias,
            )
        else:
            self.lora_layer_indices = []
            self.lora_target_modules = []

    def _freeze_vision_tower(self):
        self.vision_model.eval()
        for parameter in self.vision_model.parameters():
            parameter.requires_grad = False

    def _resolve_layer_indices(self, layer_indices, num_router_layers):
        if layer_indices:
            resolved = [int(index) for index in layer_indices]
        else:
            resolved = np.linspace(
                0,
                self.num_layers - 1,
                num=num_router_layers,
                dtype=int,
            ).tolist()

        for index in resolved:
            if index < 0 or index >= self.num_layers:
                raise ValueError(
                    f"Layer index {index} is out of range for "
                    f"{self.num_layers} CLIP blocks."
                )
        return resolved

    def _resolve_lora_layer_indices(
        self,
        lora_layer_indices,
        lora_layer_policy,
        lora_start_layer,
        lora_end_layer,
    ):
        if lora_layer_indices:
            resolved = sorted({int(index) for index in lora_layer_indices})
        else:
            if lora_start_layer is not None or lora_end_layer is not None:
                start = (
                    0 if lora_start_layer is None else int(lora_start_layer)
                )
                end = (
                    self.num_layers - 1
                    if lora_end_layer is None
                    else int(lora_end_layer)
                )
            else:
                policy = str(lora_layer_policy or "all").lower()
                if policy in {"latter_half", "last_half", "second_half"}:
                    start = self.num_layers // 2
                    end = self.num_layers - 1
                elif policy in {"former_half", "first_half"}:
                    start = 0
                    end = max((self.num_layers // 2) - 1, 0)
                elif policy in {"all", "full"}:
                    start = 0
                    end = self.num_layers - 1
                else:
                    raise ValueError(
                        "Unsupported lora_layer_policy: "
                        f"{lora_layer_policy}"
                    )
            resolved = list(range(start, end + 1))

        if not resolved:
            raise ValueError("LoRA layer indices cannot be empty.")

        for index in resolved:
            if index < 0 or index >= self.num_layers:
                raise ValueError(
                    f"LoRA layer index {index} is out of range for "
                    f"{self.num_layers} CLIP blocks."
                )
        return resolved

    def _resolve_lora_target_modules(self, lora_target_modules):
        if lora_target_modules is None:
            raw_targets = ["ffn"]
        elif isinstance(lora_target_modules, str):
            raw_targets = [
                target.strip()
                for target in lora_target_modules.split(",")
                if target.strip()
            ]
        else:
            raw_targets = list(lora_target_modules)

        aliases = {
            "ffn": ("mlp.fc1", "mlp.fc2"),
            "mlp": ("mlp.fc1", "mlp.fc2"),
            "mlp.fc1": ("mlp.fc1",),
            "mlp.fc2": ("mlp.fc2",),
            "fc1": ("mlp.fc1",),
            "fc2": ("mlp.fc2",),
            "attention_qv": ("self_attn.q_proj", "self_attn.v_proj"),
            "attn_qv": ("self_attn.q_proj", "self_attn.v_proj"),
            "qv": ("self_attn.q_proj", "self_attn.v_proj"),
            "self_attn.q_proj": ("self_attn.q_proj",),
            "self_attn.v_proj": ("self_attn.v_proj",),
            "q_proj": ("self_attn.q_proj",),
            "v_proj": ("self_attn.v_proj",),
        }

        resolved = []
        seen = set()
        for raw_target in raw_targets:
            key = str(raw_target).strip().lower()
            if not key:
                continue
            if key not in aliases:
                raise ValueError(
                    "Unsupported lora_target_modules entry: "
                    f"{raw_target}. Supported aliases include ffn, mlp, "
                    "mlp.fc1, mlp.fc2, qv, q_proj, and v_proj."
                )
            for target in aliases[key]:
                if target not in seen:
                    resolved.append(target)
                    seen.add(target)

        if not resolved:
            raise ValueError("LoRA target modules cannot be empty.")
        return resolved

    def _get_parent_module(self, encoder_layer, target_path):
        parent_path, attr_name = target_path.rsplit(".", 1)
        parent = encoder_layer
        for path_part in parent_path.split("."):
            parent = getattr(parent, path_part)
        return parent, attr_name

    def _inject_lora(self, rank, alpha, dropout, train_bias):
        lora_layer_set = set(self.lora_layer_indices)
        for layer_index, encoder_layer in enumerate(
            self.vision_model.encoder.layers
        ):
            if layer_index not in lora_layer_set:
                continue
            for target_path in self.lora_target_modules:
                parent, attr_name = self._get_parent_module(
                    encoder_layer,
                    target_path,
                )
                base_linear = getattr(parent, attr_name)
                if not isinstance(base_linear, nn.Linear):
                    raise TypeError(
                        "LoRA can only wrap nn.Linear modules, but "
                        f"{target_path} is {type(base_linear).__name__}."
                    )
                setattr(
                    parent,
                    attr_name,
                    LoRALinear(
                        base_linear,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        train_bias=train_bias,
                    ),
                )

    def lora_parameters(self):
        for module in self.modules():
            if isinstance(module, LoRALinear):
                yield module.lora_A
                yield module.lora_B
                if module.base_layer.bias is not None and module.base_layer.bias.requires_grad:
                    yield module.base_layer.bias

    def forward_cls_token(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        if outputs.pooler_output is not None:
            return outputs.pooler_output.contiguous()
        return outputs.last_hidden_state[:, 0, :].contiguous()

    def _forward_hidden_states(self, pixel_values):
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        return outputs.hidden_states

    def forward_multi_cls(self, pixel_values):
        hidden_states = self._forward_hidden_states(pixel_values)
        post_layernorm = getattr(self.vision_model, "post_layernorm", None)

        cls_token_list = []
        for layer_index in self.layer_indices:
            layer_tokens = hidden_states[layer_index + 1]
            cls_token = layer_tokens[:, 0, :].contiguous()
            if post_layernorm is not None:
                cls_token = post_layernorm(cls_token)
            cls_token_list.append(cls_token)

        return torch.stack(cls_token_list, dim=1)

    def forward_multi_patch(self, pixel_values):
        hidden_states = self._forward_hidden_states(pixel_values)

        patch_token_list = []
        for layer_index in self.layer_indices:
            # hidden_states[0] 是 patch embedding，真实的第 i 个 Transformer 输出在 hidden_states[i + 1]
            layer_tokens = hidden_states[layer_index + 1]
            # 去掉 CLS token 后，每层保留 16 x 16 = 256 个 patch tokens
            layer_patch_tokens = layer_tokens[:, 1:, :].contiguous()
            patch_token_list.append(layer_patch_tokens)

        return torch.stack(patch_token_list, dim=1)

    def forward(self, pixel_values):
        patch_tokens = self.forward_multi_patch(pixel_values)
        batch_size, num_layers, num_patches, hidden_dim = patch_tokens.shape
        return patch_tokens.reshape(
            batch_size,
            num_layers * num_patches,
            hidden_dim,
        )
