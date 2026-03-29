from abc import ABC, abstractmethod
from .schedules import Schedule


class InjectionStrategy(ABC):
    """Determines injection strength for each layer type at a given progress."""

    @abstractmethod
    def get_alpha(self, progress, layer_type):
        """
        Args:
            progress: float in [0, 1], fraction of denoising completed.
            layer_type: "attention" or "conv".
        Returns:
            float in [0, 1]. 0 = no injection, 1 = full replacement.
        """
        pass


class NoInjection(InjectionStrategy):
    """No injection at any timestep."""

    def get_alpha(self, progress, layer_type):
        return 0.0


class HardInjection(InjectionStrategy):
    """Reproduces original PnP: full replacement before cutoff, nothing after."""

    def __init__(self, attn_cutoff=0.5, conv_cutoff=0.8):
        self.attn_cutoff = attn_cutoff
        self.conv_cutoff = conv_cutoff

    def get_alpha(self, progress, layer_type):
        if layer_type == "attention":
            return 1.0 if progress < self.attn_cutoff else 0.0
        elif layer_type == "conv":
            return 1.0 if progress < self.conv_cutoff else 0.0
        return 0.0


class BlendedInjection(InjectionStrategy):
    """Uses Schedule objects to determine alpha for each layer type."""

    def __init__(self, attn_schedule, conv_schedule):
        """
        Args:
            attn_schedule: Schedule instance for attention injection.
            conv_schedule: Schedule instance for conv injection.
        """
        self.attn_schedule = attn_schedule
        self.conv_schedule = conv_schedule

    def get_alpha(self, progress, layer_type):
        if layer_type == "attention":
            return self.attn_schedule.get_value(progress)
        elif layer_type == "conv":
            return self.conv_schedule.get_value(progress)
        return 0.0
