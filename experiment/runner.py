import os
import json
import yaml
import torch

from pnp_utils import seed_everything
from pnp_controlled import PNPControlled
from injection.controller import InjectionController
from injection.adaptive import AdaptiveController
from injection.strategies import NoInjection, HardInjection, BlendedInjection
from injection.schedules import ConstantSchedule, StepSchedule, LinearDecaySchedule, CosineDecaySchedule


SCHEDULE_MAP = {
    "ConstantSchedule": ConstantSchedule,
    "StepSchedule": StepSchedule,
    "LinearDecaySchedule": LinearDecaySchedule,
    "CosineDecaySchedule": CosineDecaySchedule,
}

STRATEGY_MAP = {
    "NoInjection": NoInjection,
    "HardInjection": HardInjection,
}


def build_schedule(sched_config):
    """Build a Schedule from a config dict."""
    cls = SCHEDULE_MAP[sched_config["type"]]
    return cls(**sched_config.get("params", {}))


def build_strategy_from_config(run_config):
    """Build an InjectionStrategy from a run config dict."""
    strategy_name = run_config["strategy"]

    if strategy_name in STRATEGY_MAP:
        return STRATEGY_MAP[strategy_name](**run_config.get("params", {}))

    if strategy_name == "BlendedInjection":
        attn_sched = build_schedule(run_config["attn_schedule"])
        conv_sched = build_schedule(run_config["conv_schedule"])
        return BlendedInjection(attn_sched, conv_sched)

    raise ValueError(f"Unknown strategy: {strategy_name}")


def build_controller_from_config(run_config, timesteps=None):
    """Build an InjectionController (or AdaptiveController) from config."""
    strategy = build_strategy_from_config(run_config)

    if "adaptive_params" in run_config:
        ap = run_config["adaptive_params"]
        return AdaptiveController(
            strategy=strategy,
            total_timesteps=timesteps,
            kp=ap.get("kp", 0.5),
            kd=ap.get("kd", 0.1),
            target_deviation=ap.get("target_deviation", 0.1),
        )

    return InjectionController(strategy=strategy, total_timesteps=timesteps)


def save_run_metadata(output_path, name, run_config, base_config):
    """Save run metadata for reproducibility."""
    metadata = {
        "name": name,
        "run_config": run_config,
        "base_config": base_config,
    }
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)


class ExperimentRunner:
    """
    Runs multiple PNP configurations, saves outputs with metadata,
    and generates comparison grids + metrics.
    """

    def __init__(self, base_config, output_root):
        self.base_config = base_config
        self.output_root = output_root
        self.runs = []
        self.results = {}

    def add_run(self, name, run_config):
        """
        Args:
            name: string identifier for this run.
            run_config: dict with strategy/schedule config (from YAML).
        """
        self.runs.append((name, run_config))

    def execute(self):
        """Execute all queued runs sequentially."""
        os.makedirs(self.output_root, exist_ok=True)

        for name, run_config in self.runs:
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print(f"{'='*60}")

            run_output = os.path.join(self.output_root, name)
            os.makedirs(run_output, exist_ok=True)

            config = self.base_config.copy()
            config["output_path"] = run_output

            seed_everything(config["seed"])

            # Build controller (timesteps set to None, will be initialized in PNPControlled)
            controller = build_controller_from_config(run_config, timesteps=None)
            pnp = PNPControlled(config, controller=controller)
            output_img = pnp.run_pnp()

            self.results[name] = {
                "output_path": run_output,
                "image": output_img,
                "controller": pnp.controller,
            }

            save_run_metadata(run_output, name, run_config, config)

            # Save adaptive alpha history if available
            if hasattr(pnp.controller, 'get_alpha_history'):
                history = pnp.controller.get_alpha_history()
                if history:
                    with open(os.path.join(run_output, "alpha_history.json"), "w") as f:
                        json.dump(history, f, indent=2)

            print(f"Saved: {run_output}")

    def compare(self, reference_name=None):
        """Generate comparison grid and compute metrics."""
        from experiment.metrics import compute_metrics, make_comparison_grid

        if not self.results:
            print("No results to compare.")
            return

        # Generate comparison grid
        images = {name: r["image"] for name, r in self.results.items()}
        grid = make_comparison_grid(images)
        grid_path = os.path.join(self.output_root, "comparison_grid.png")
        grid.save(grid_path)
        print(f"\nComparison grid saved to: {grid_path}")

        # Compute metrics against reference
        if reference_name and reference_name in self.results:
            ref_img = self.results[reference_name]["image"]
            print(f"\nMetrics (reference: {reference_name}):")
            print(f"{'Run':<30} {'LPIPS':>8} {'SSIM':>8}")
            print("-" * 48)
            metrics_all = {}
            for name, r in self.results.items():
                if name != reference_name:
                    metrics = compute_metrics(ref_img, r["image"])
                    metrics_all[name] = metrics
                    print(f"{name:<30} {metrics['lpips']:>8.4f} {metrics['ssim']:>8.4f}")

            # Save metrics
            with open(os.path.join(self.output_root, "metrics.json"), "w") as f:
                json.dump(metrics_all, f, indent=2)
