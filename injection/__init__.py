from .schedules import (
    Schedule,
    ConstantSchedule,
    StepSchedule,
    LinearDecaySchedule,
    CosineDecaySchedule,
)
from .strategies import (
    InjectionStrategy,
    NoInjection,
    HardInjection,
    BlendedInjection,
)
from .controller import InjectionController


def __getattr__(name):
    if name == "AdaptiveController":
        from .adaptive import AdaptiveController
        return AdaptiveController
    if name == "register_attention_control":
        from .hooks import register_attention_control
        return register_attention_control
    if name == "register_conv_control":
        from .hooks import register_conv_control
        return register_conv_control
    raise AttributeError(f"module 'injection' has no attribute {name}")


__all__ = [
    "Schedule", "ConstantSchedule", "StepSchedule",
    "LinearDecaySchedule", "CosineDecaySchedule",
    "InjectionStrategy", "NoInjection", "HardInjection", "BlendedInjection",
    "InjectionController", "AdaptiveController",
    "register_attention_control", "register_conv_control",
]
