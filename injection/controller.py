class InjectionController:
    """
    Central object that determines injection strength at each timestep.
    Attached to UNet modules via setattr; queried inside monkey-patched forwards.
    """

    def __init__(self, strategy, total_timesteps):
        """
        Args:
            strategy: an InjectionStrategy instance.
            total_timesteps: list/tensor of scheduler timesteps (e.g., 50 values from 981 down to 1).
        """
        self.strategy = strategy
        self.total_timesteps = total_timesteps
        self.n_steps = len(total_timesteps)
        self._timestep_to_index = {int(t): i for i, t in enumerate(total_timesteps)}

    def get_alpha(self, t, layer_type):
        """
        Returns injection strength in [0, 1].

        Args:
            t: current timestep value (int).
            layer_type: "attention" or "conv".
        """
        step_index = self._timestep_to_index.get(int(t), 0)
        progress = step_index / self.n_steps
        return self.strategy.get_alpha(progress, layer_type)

    def log_features(self, t, layer_type, source_features, target_features):
        """Hook for adaptive controllers. No-op in base class."""
        pass

    def step_update(self, t):
        """Called after each denoising step. No-op in base class."""
        pass
