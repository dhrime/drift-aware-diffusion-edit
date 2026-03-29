import torch
from .controller import InjectionController


class AdaptiveController(InjectionController):
    """
    Adjusts injection strength based on feature deviation between source and target.

    Uses a proportional-derivative (PD) controller:
        alpha_adjusted = clip(base_alpha + Kp * error + Kd * d_error, 0, 1)

    where error = normalized_deviation - target_deviation.
    normalized_deviation = ||target_features - source_features|| / ||source_features||
    """

    def __init__(self, strategy, total_timesteps, kp=0.5, kd=0.1,
                 target_deviation=0.1):
        """
        Args:
            strategy: base InjectionStrategy for nominal alpha values.
            total_timesteps: scheduler timesteps.
            kp: proportional gain.
            kd: derivative gain.
            target_deviation: desired deviation level.
        """
        super().__init__(strategy, total_timesteps)
        self.kp = kp
        self.kd = kd
        self.target_deviation = target_deviation
        self._prev_error = {"attention": 0.0, "conv": 0.0}
        self._current_adjustment = {"attention": 0.0, "conv": 0.0}
        self._alpha_history = []

    def get_alpha(self, t, layer_type):
        base_alpha = super().get_alpha(t, layer_type)
        adjustment = self._current_adjustment.get(layer_type, 0.0)
        alpha = max(0.0, min(1.0, base_alpha + adjustment))
        return alpha

    def log_features(self, t, layer_type, source_features, target_features):
        """
        Computes deviation and updates PD controller state.
        Called from hooks before blending.
        """
        with torch.no_grad():
            source_norm = source_features.float().norm()
            if source_norm < 1e-8:
                return
            deviation = (target_features.float() - source_features.float()).norm() / source_norm
            error = deviation.item() - self.target_deviation

            d_error = error - self._prev_error[layer_type]
            self._current_adjustment[layer_type] = self.kp * error + self.kd * d_error
            self._prev_error[layer_type] = error

    def step_update(self, t):
        """Record alpha values for analysis."""
        step_index = self._timestep_to_index.get(int(t), 0)
        progress = step_index / self.n_steps
        record = {
            "t": int(t),
            "progress": progress,
        }
        for layer_type in ["attention", "conv"]:
            base = self.strategy.get_alpha(progress, layer_type)
            adj = self._current_adjustment.get(layer_type, 0.0)
            record[f"{layer_type}_base"] = base
            record[f"{layer_type}_adjustment"] = adj
            record[f"{layer_type}_alpha"] = max(0.0, min(1.0, base + adj))
        self._alpha_history.append(record)

    def get_alpha_history(self):
        """Returns recorded alpha trajectories for analysis."""
        return self._alpha_history
