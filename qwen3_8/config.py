from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class Qwen3_8Config:
    """
    对应 `qwen3.8 27B/config.json` 的模型配置类。
    支持直接从 JSON 配置文件加载所有超参数（含混合注意力、Gated DeltaNet 线性注意力、M-RoPE 及 MTP 配置）。
    """

    # 基础架构参数
    model_type: str = "qwen3_5_text"
    vocab_size: int = 248320
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 64
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    dtype: str = "bfloat16"
    bos_token_id: int = 248044
    eos_token_id: int = 248044
    pad_token_id: Optional[int] = None
    max_position_embeddings: int = 262144

    # 混合注意力层排布 (3 层 linear_attention + 1 层 full_attention 循环)
    full_attention_interval: int = 4
    layer_types: List[str] = field(default_factory=list)

    # Full Attention (GQA + Output Gate + Partial RoPE) 参数
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_output_gate: bool = True
    output_gate_type: str = "swish"
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10000000.0
    mrope_interleaved: bool = True
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])

    # Linear Attention (Gated DeltaNet) 参数
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_value_head_dim: int = 128
    mamba_ssm_dtype: str = "float32"

    # Multi-Token Prediction (MTP) 参数
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False

    # 量化与多模态附加配置 (保留原始字典备查)
    vision_config: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = [
                "full_attention"
                if (i + 1) % self.full_attention_interval == 0
                else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]

    @property
    def rotary_dim(self) -> int:
        """实际参与 RoPE 旋转的维度大小 (256 * 0.25 = 64)。"""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def full_attention_layer_indices(self) -> List[int]:
        """返回所有使用标准 Full Attention 的层号列表。"""
        return [
            i for i, l_type in enumerate(self.layer_types) if l_type == "full_attention"
        ]

    @property
    def linear_attention_layer_indices(self) -> List[int]:
        """返回所有使用 Linear Attention (Gated DeltaNet) 的层号列表。"""
        return [
            i for i, l_type in enumerate(self.layer_types) if l_type == "linear_attention"
        ]

    @classmethod
    def from_json_file(cls, config_path: Union[str, Path]) -> "Qwen3_8Config":
        """
        从 `qwen3.8 27B/config.json` 读取并解析模型配置。
        自动兼容顶层多模态嵌套结构（提取 `text_config`）或纯文本配置结构。
        """
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            raw: Dict[str, Any] = json.load(f)

        text_cfg: Dict[str, Any] = raw.get("text_config", raw)
        rope_params: Dict[str, Any] = text_cfg.get("rope_parameters", {})

        return cls(
            model_type=text_cfg.get("model_type", "qwen3_5_text"),
            vocab_size=text_cfg.get("vocab_size", 248320),
            hidden_size=text_cfg.get("hidden_size", 5120),
            intermediate_size=text_cfg.get("intermediate_size", 17408),
            num_hidden_layers=text_cfg.get("num_hidden_layers", 64),
            hidden_act=text_cfg.get("hidden_act", "silu"),
            rms_norm_eps=text_cfg.get("rms_norm_eps", 1e-6),
            initializer_range=text_cfg.get("initializer_range", 0.02),
            tie_word_embeddings=text_cfg.get(
                "tie_word_embeddings", raw.get("tie_word_embeddings", False)
            ),
            dtype=text_cfg.get("dtype", "bfloat16"),
            bos_token_id=text_cfg.get("bos_token_id", 248044),
            eos_token_id=text_cfg.get("eos_token_id", 248044),
            pad_token_id=text_cfg.get("pad_token_id", None),
            max_position_embeddings=text_cfg.get("max_position_embeddings", 262144),
            full_attention_interval=text_cfg.get("full_attention_interval", 4),
            layer_types=list(text_cfg.get("layer_types", [])),
            num_attention_heads=text_cfg.get("num_attention_heads", 24),
            num_key_value_heads=text_cfg.get("num_key_value_heads", 4),
            head_dim=text_cfg.get("head_dim", 256),
            attention_bias=text_cfg.get("attention_bias", False),
            attention_dropout=text_cfg.get("attention_dropout", 0.0),
            attn_output_gate=text_cfg.get("attn_output_gate", True),
            output_gate_type=text_cfg.get("output_gate_type", "swish"),
            partial_rotary_factor=rope_params.get(
                "partial_rotary_factor", text_cfg.get("partial_rotary_factor", 0.25)
            ),
            rope_theta=float(rope_params.get("rope_theta", 10000000.0)),
            mrope_interleaved=rope_params.get("mrope_interleaved", True),
            mrope_section=list(rope_params.get("mrope_section", [11, 11, 10])),
            linear_conv_kernel_dim=text_cfg.get("linear_conv_kernel_dim", 4),
            linear_key_head_dim=text_cfg.get("linear_key_head_dim", 128),
            linear_num_key_heads=text_cfg.get("linear_num_key_heads", 16),
            linear_num_value_heads=text_cfg.get("linear_num_value_heads", 48),
            linear_value_head_dim=text_cfg.get("linear_value_head_dim", 128),
            mamba_ssm_dtype=text_cfg.get("mamba_ssm_dtype", "float32"),
            mtp_num_hidden_layers=text_cfg.get("mtp_num_hidden_layers", 1),
            mtp_use_dedicated_embeddings=text_cfg.get(
                "mtp_use_dedicated_embeddings", False
            ),
            vision_config=raw.get("vision_config"),
            quantization_config=raw.get("quantization_config"),
        )
