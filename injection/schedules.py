import math
from abc import ABC, abstractmethod


class Schedule(ABC):
    """Maps normalized denoising progress [0,1] to injection alpha [0,1]."""

    @abstractmethod
    def get_value(self, progress):
        """
        Args:
            progress: float in [0, 1]. 0 = first denoising step (highest noise),
                      1 = last step (lowest noise).
        Returns:
            float in [0, 1]. Injection strength.
        """
        pass


class ConstantSchedule(Schedule):
    """Returns a fixed alpha at all timesteps."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def get_value(self, progress):
        return self.alpha


class StepSchedule(Schedule):
    """Binary step: alpha before cutoff, 0 after. Reproduces original PnP behavior."""

    def __init__(self, alpha=1.0, cutoff=0.5):
        self.alpha = alpha
        self.cutoff = cutoff

    def get_value(self, progress):
        return self.alpha if progress < self.cutoff else 0.0


class LinearDecaySchedule(Schedule):
    """Linearly decays from start_alpha to end_alpha over [0, cutoff], then 0."""

    def __init__(self, start_alpha=1.0, end_alpha=0.0, cutoff=1.0):
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha
        self.cutoff = cutoff

    def get_value(self, progress):
        if progress >= self.cutoff:
            return 0.0
        t = progress / self.cutoff
        return self.start_alpha + (self.end_alpha - self.start_alpha) * t


class CosineDecaySchedule(Schedule):
    """Cosine decay from start_alpha to 0 over [0, cutoff], then 0."""

    def __init__(self, start_alpha=1.0, cutoff=1.0):
        self.start_alpha = start_alpha
        self.cutoff = cutoff

    def get_value(self, progress):
        if progress >= self.cutoff:
            return 0.0
        t = progress / self.cutoff
        return self.start_alpha * 0.5 * (1.0 + math.cos(math.pi * t))
