from enum import Enum
from importlib.util import find_spec
from typing import Type

from .models import ModelSpecification
from .models.wan import WanControlModelSpecification, WanModelSpecification


class ModelType(str, Enum):
    COGVIDEOX = "cogvideox"
    COGVIEW4 = "cogview4"
    FLUX = "flux"
    HUNYUAN_VIDEO = "hunyuan_video"
    LTX_VIDEO = "ltx_video"
    WAN = "wan"


class TrainingType(str, Enum):
    # SFT
    LORA = "lora"
    FULL_FINETUNE = "full-finetune"

    # Control
    CONTROL_LORA = "control-lora"
    CONTROL_FULL_FINETUNE = "control-full-finetune"


def _optional_import(module: str, symbol: str):
    if find_spec(module) is None:
        return None
    module_obj = __import__(module, fromlist=[symbol])
    return getattr(module_obj, symbol)


CogVideoXModelSpecification = _optional_import("finetrainers.models.cogvideox", "CogVideoXModelSpecification")
CogView4ModelSpecification = _optional_import("finetrainers.models.cogview4", "CogView4ModelSpecification")
CogView4ControlModelSpecification = _optional_import(
    "finetrainers.models.cogview4", "CogView4ControlModelSpecification"
)
FluxModelSpecification = _optional_import("finetrainers.models.flux", "FluxModelSpecification")
HunyuanVideoModelSpecification = _optional_import("finetrainers.models.hunyuan_video", "HunyuanVideoModelSpecification")
LTXVideoModelSpecification = _optional_import("finetrainers.models.ltx_video", "LTXVideoModelSpecification")


SUPPORTED_MODEL_CONFIGS = {}

if CogVideoXModelSpecification is not None:
    SUPPORTED_MODEL_CONFIGS[ModelType.COGVIDEOX] = {
        TrainingType.LORA: CogVideoXModelSpecification,
        TrainingType.FULL_FINETUNE: CogVideoXModelSpecification,
    }

if CogView4ModelSpecification is not None:
    SUPPORTED_MODEL_CONFIGS[ModelType.COGVIEW4] = {
        TrainingType.LORA: CogView4ModelSpecification,
        TrainingType.FULL_FINETUNE: CogView4ModelSpecification,
    }
    if CogView4ControlModelSpecification is not None:
        SUPPORTED_MODEL_CONFIGS[ModelType.COGVIEW4].update(
            {
                TrainingType.CONTROL_LORA: CogView4ControlModelSpecification,
                TrainingType.CONTROL_FULL_FINETUNE: CogView4ControlModelSpecification,
            }
        )

if FluxModelSpecification is not None:
    SUPPORTED_MODEL_CONFIGS[ModelType.FLUX] = {
        TrainingType.LORA: FluxModelSpecification,
        TrainingType.FULL_FINETUNE: FluxModelSpecification,
    }

if HunyuanVideoModelSpecification is not None:
    SUPPORTED_MODEL_CONFIGS[ModelType.HUNYUAN_VIDEO] = {
        TrainingType.LORA: HunyuanVideoModelSpecification,
        TrainingType.FULL_FINETUNE: HunyuanVideoModelSpecification,
    }

if LTXVideoModelSpecification is not None:
    SUPPORTED_MODEL_CONFIGS[ModelType.LTX_VIDEO] = {
        TrainingType.LORA: LTXVideoModelSpecification,
        TrainingType.FULL_FINETUNE: LTXVideoModelSpecification,
    }

SUPPORTED_MODEL_CONFIGS[ModelType.WAN] = {
    TrainingType.LORA: WanModelSpecification,
    TrainingType.FULL_FINETUNE: WanModelSpecification,
    TrainingType.CONTROL_LORA: WanControlModelSpecification,
    TrainingType.CONTROL_FULL_FINETUNE: WanControlModelSpecification,
}


def _get_model_specifiction_cls(model_name: str, training_type: str) -> Type[ModelSpecification]:
    if model_name not in SUPPORTED_MODEL_CONFIGS:
        raise ValueError(
            f"Model {model_name} not supported. Supported models are: {list(SUPPORTED_MODEL_CONFIGS.keys())}"
        )
    if training_type not in SUPPORTED_MODEL_CONFIGS[model_name]:
        raise ValueError(
            f"Training type {training_type} not supported for model {model_name}. Supported training types are: {list(SUPPORTED_MODEL_CONFIGS[model_name].keys())}"
        )
    return SUPPORTED_MODEL_CONFIGS[model_name][training_type]
