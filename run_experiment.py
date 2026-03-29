"""
Run PnP injection experiments from a YAML config.

Usage:
    python run_experiment.py --config experiment/configs/ablation_sweep.yaml
    python run_experiment.py --config experiment/configs/ablation_sweep.yaml --runs baseline_hard blended_linear_decay
    python run_experiment.py --config experiment/configs/ablation_sweep.yaml --compare-only
"""
import argparse
import yaml
from datetime import datetime

from experiment.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run PnP injection experiments")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment YAML config')
    parser.add_argument('--runs', nargs='*', default=None,
                        help='Run only these named runs (default: all)')
    parser.add_argument('--compare-only', action='store_true',
                        help='Skip execution, just compare existing outputs')
    parser.add_argument('--no-metrics', action='store_true',
                        help='Skip metric computation (faster, no lpips/torchmetrics needed)')
    parser.add_argument('--no-timestamp', action='store_true',
                        help='Write directly to output_root without a timestamp suffix')
    opt = parser.parse_args()

    with open(opt.config) as f:
        exp_config = yaml.safe_load(f)

    with open(exp_config["base_config"]) as f:
        base_config = yaml.safe_load(f)

    # Override seed if specified in experiment config
    if "seed" in exp_config:
        base_config["seed"] = exp_config["seed"]

    # Build output root — append timestamp unless --no-timestamp is set
    output_root = exp_config["output_root"]
    if not opt.no_timestamp:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_root = f"{output_root}_{ts}"
    print(f"Output directory: {output_root}")

    runner = ExperimentRunner(base_config, output_root)

    for run_cfg in exp_config["runs"]:
        if opt.runs is None or run_cfg["name"] in opt.runs:
            runner.add_run(run_cfg["name"], run_cfg)

    if not opt.compare_only:
        runner.execute()

    if not opt.no_metrics:
        # Use first run as reference for metrics
        ref_name = exp_config["runs"][0]["name"]
        runner.compare(reference_name=ref_name)


if __name__ == '__main__':
    main()
