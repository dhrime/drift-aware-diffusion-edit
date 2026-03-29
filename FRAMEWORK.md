# Controllable PnP Injection Framework

This document describes the controllable injection framework built on top of the
[Plug-and-Play Diffusion Features](https://arxiv.org/abs/2211.12572) (CVPR 2023) codebase.
It turns PnP from a static feature injection method into a system where injection
strength can vary over time, respond to feature deviations during generation, and be
swept across configurations in batch experiments.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Schedules](#schedules)
6. [Strategies](#strategies)
7. [Controller](#controller)
8. [Adaptive Controller](#adaptive-controller)
9. [Hooks (Blended Injection)](#hooks-blended-injection)
10. [PNPControlled](#pnpcontrolled)
11. [Experiment Runner](#experiment-runner)
12. [Metrics](#metrics)
13. [Experiment Config Format](#experiment-config-format)
14. [Writing Custom Schedules and Strategies](#writing-custom-schedules-and-strategies)
15. [Troubleshooting](#troubleshooting)

---

## Overview

The original PnP injects source image features into the generation process with a
**binary on/off** decision per timestep. This framework generalizes that to a
continuous **alpha blending** approach:

```
injected = alpha * source_features + (1 - alpha) * target_features
```

- `alpha = 1.0` &rarr; full replacement (original PnP behavior)
- `alpha = 0.0` &rarr; no injection
- `0 < alpha < 1` &rarr; partial blending

Alpha can vary over time (schedules), differ per layer type (strategies), and
adapt in real-time based on feature deviation (adaptive controller).

---

## Repository Structure

```
pnp-diffusers/
  pnp.py                        # Original PnP pipeline (unchanged)
  pnp_utils.py                  # Original injection hooks (unchanged)
  preprocess.py                  # DDIM inversion / latent extraction (unchanged)
  config_pnp.yaml               # Original config (unchanged)

  pnp_controlled.py              # PNPControlled - drop-in replacement using the framework
  run_experiment.py              # CLI entry point for batch experiments

  injection/                     # Core framework
    __init__.py                  # Package exports
    schedules.py                 # Time-varying alpha curves
    strategies.py                # Per-layer injection logic
    controller.py                # Central controller attached to UNet modules
    adaptive.py                  # PD feedback controller
    hooks.py                     # Monkey-patched forwards with alpha blending

  experiment/                    # Experiment infrastructure
    __init__.py
    runner.py                    # Batch runner + YAML config factory
    metrics.py                   # LPIPS, SSIM, comparison grids
    configs/
      ablation_sweep.yaml        # Example sweep with 6 configurations
```

All original files are **unchanged**. The original `pnp.py` still works as before.

---

## Quick Start

### Prerequisites

Install additional dependencies (on top of the original requirements):

```bash
pip install lpips torchmetrics
```

Or install everything:

```bash
pip install -r requirements.txt
```

### Step 1: Preprocess (extract latents)

Same as original PnP. Run this once per source image:

```bash
python preprocess.py --data_path data/horse.jpg --inversion_prompt "a photo of a horse"
```

This saves intermediate noisy latents to `latents_forward/horse/`.

### Step 2a: Run a single controlled generation

```bash
python pnp_controlled.py --config_path config_pnp.yaml
```

This uses the same config as the original `pnp.py` and produces identical output
(it defaults to `HardInjection` with the config thresholds).

### Step 2b: Run a full experiment sweep

```bash
python run_experiment.py --config experiment/configs/ablation_sweep.yaml
```

This runs 6 configurations (no injection, hard PnP, constant blend, linear decay,
cosine decay, adaptive PD), saves outputs to `experiments/sweep_001/`, and generates
a comparison grid with LPIPS/SSIM metrics.

### Step 2c: Run specific experiments only

```bash
python run_experiment.py --config experiment/configs/ablation_sweep.yaml \
    --runs baseline_hard blended_linear_decay adaptive_pd
```

### Step 2d: Compare existing outputs without re-running

```bash
python run_experiment.py --config experiment/configs/ablation_sweep.yaml --compare-only
```

### Step 2e: Skip metric computation (faster, no lpips/torchmetrics needed)

```bash
python run_experiment.py --config experiment/configs/ablation_sweep.yaml --no-metrics
```

---

## Core Concepts

The framework has four layers of abstraction:

```
Schedule          - maps progress [0,1] -> alpha [0,1]  (a single curve)
    |
Strategy          - picks which Schedule to use per layer type
    |
Controller        - maps raw timesteps to progress, queries strategy
    |
Hooks             - monkey-patched UNet forwards that query the controller and blend
```

**Progress** is a normalized value in `[0, 1]`:
- `0.0` = first denoising step (highest noise level)
- `1.0` = last denoising step (lowest noise level)

This decouples schedules from the specific scheduler timestep values.

---

## Schedules

**File:** `injection/schedules.py`

Schedules define how injection alpha varies over denoising progress.

### ConstantSchedule

```python
ConstantSchedule(alpha=1.0)
```

Returns a fixed alpha at every timestep. Useful as a baseline or as input to
the adaptive controller.

### StepSchedule

```python
StepSchedule(alpha=1.0, cutoff=0.5)
```

Returns `alpha` when `progress < cutoff`, `0.0` otherwise. This reproduces
the original PnP binary behavior when `alpha=1.0`.

### LinearDecaySchedule

```python
LinearDecaySchedule(start_alpha=1.0, end_alpha=0.0, cutoff=1.0)
```

Linearly interpolates from `start_alpha` to `end_alpha` over `[0, cutoff]`,
then returns `0.0`. Example: `LinearDecaySchedule(1.0, 0.0, 0.8)` decays from
1.0 to 0.0 over the first 80% of timesteps.

### CosineDecaySchedule

```python
CosineDecaySchedule(start_alpha=1.0, cutoff=1.0)
```

Cosine half-wave decay: `start_alpha * 0.5 * (1 + cos(pi * progress/cutoff))`.
Starts at `start_alpha`, smoothly decays to 0 at `cutoff`. Stays at 0 after.

---

## Strategies

**File:** `injection/strategies.py`

Strategies determine alpha for each **layer type** ("attention" or "conv").

### NoInjection

```python
NoInjection()
```

Always returns `0.0`. Pure text-guided generation with no source features.

### HardInjection

```python
HardInjection(attn_cutoff=0.5, conv_cutoff=0.8)
```

Returns `1.0` before the cutoff, `0.0` after. Separate cutoffs for attention
and conv layers. This exactly reproduces the original PnP paper behavior.

### BlendedInjection

```python
BlendedInjection(attn_schedule, conv_schedule)
```

Delegates to `Schedule` objects for each layer type. This is the most flexible
strategy -- you can combine any schedule for attention with any schedule for conv.

**Example:** Strong attention injection with gradual conv decay:

```python
BlendedInjection(
    attn_schedule=StepSchedule(alpha=1.0, cutoff=0.5),
    conv_schedule=LinearDecaySchedule(start_alpha=1.0, end_alpha=0.0, cutoff=0.8),
)
```

---

## Controller

**File:** `injection/controller.py`

```python
InjectionController(strategy, total_timesteps)
```

The controller is the bridge between the strategy and the UNet hooks:

- Converts raw timestep values (e.g., 981, 961, ...) to normalized progress
- Queries the strategy for alpha
- Gets attached to UNet modules via `setattr`
- Provides `log_features()` and `step_update()` hooks (no-ops in base class)

You don't usually create this directly -- the experiment runner builds it from
config. But for programmatic use:

```python
from injection import InjectionController, HardInjection

strategy = HardInjection(attn_cutoff=0.5, conv_cutoff=0.8)
controller = InjectionController(strategy, scheduler.timesteps)
```

---

## Adaptive Controller

**File:** `injection/adaptive.py`

```python
AdaptiveController(strategy, total_timesteps, kp=0.5, kd=0.1, target_deviation=0.1)
```

Wraps any base strategy with a proportional-derivative (PD) feedback controller.
At each timestep, it:

1. Measures feature deviation: `||target - source|| / ||source||`
2. Computes error: `deviation - target_deviation`
3. Adjusts alpha: `alpha = clip(base_alpha + kp * error + kd * d_error, 0, 1)`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kp` | 0.5 | Proportional gain. Higher = stronger reaction to current deviation |
| `kd` | 0.1 | Derivative gain. Higher = stronger reaction to change in deviation |
| `target_deviation` | 0.1 | Desired deviation level. Controller drives deviation toward this |

**Behavior:**
- If the generation is drifting far from the source (high deviation), the controller
  increases alpha to inject more structure.
- If the generation is very close to the source (low deviation), the controller
  decreases alpha to allow more editing freedom.

**Alpha history:** After running, call `controller.get_alpha_history()` to get a list
of dicts with per-timestep alpha values, adjustments, and base values. This is
automatically saved to `alpha_history.json` by the experiment runner.

---

## Hooks (Blended Injection)

**File:** `injection/hooks.py`

These functions monkey-patch UNet modules to query the controller and blend features.

### Attention Hooks

`register_attention_control(model, controller)` patches self-attention in:
- `up_blocks[1].attentions[1, 2]`
- `up_blocks[2].attentions[0, 1, 2]`
- `up_blocks[3].attentions[0, 1, 2]`

(Same blocks as the original PnP.)

The patched forward:
1. Computes Q, K from the input
2. Queries `controller.get_alpha(t, "attention")`
3. If alpha > 0: blends source Q,K into unconditional and conditional branches
4. Proceeds with standard attention computation

### Conv Hooks

`register_conv_control(model, controller)` patches:
- `up_blocks[1].resnets[1]`

The patched forward:
1. Runs the standard ResNet computation through conv2
2. Queries `controller.get_alpha(t, "conv")`
3. If alpha > 0: blends source hidden states into unconditional and conditional branches
4. Completes the residual connection

### Compatibility

The hooks read `.t` from the module (set by the original `register_time()` function).
The existing `register_time()` and `load_source_latents_t()` from `pnp_utils.py`
work unchanged. The `denoise_step()` method is inherited from the original `PNP` class.

---

## PNPControlled

**File:** `pnp_controlled.py`

```python
PNPControlled(config, controller=None)
```

A subclass of the original `PNP` that uses the controller-based hooks. Inherits
all model loading, text embedding, VAE decoding, and denoising logic.

**What it overrides:**
- `init_pnp()`: registers controller-based hooks instead of schedule-based ones
- `run_pnp()`: calls `init_pnp()` + `sample_loop()`
- `sample_loop()`: adds `controller.step_update(t)` after each denoising step

**Default behavior:** If no controller is passed, it creates a `HardInjection`
controller from the config's `pnp_attn_t` and `pnp_f_t` values, producing output
identical to the original `pnp.py`.

---

## Experiment Runner

**File:** `experiment/runner.py`

```python
runner = ExperimentRunner(base_config, output_root)
runner.add_run("name", run_config)
runner.execute()
runner.compare(reference_name="baseline_hard")
```

### What it does

1. **execute()**: For each queued run:
   - Seeds RNG for reproducibility
   - Builds controller from config
   - Creates `PNPControlled` and runs generation
   - Saves output image, metadata JSON, and alpha history (if adaptive)

2. **compare()**: After execution:
   - Creates a labeled comparison grid (`comparison_grid.png`)
   - Computes LPIPS and SSIM against a reference run
   - Prints a metrics table and saves to `metrics.json`

### Output structure

```
experiments/sweep_001/
  comparison_grid.png           # Side-by-side grid of all outputs
  metrics.json                  # LPIPS/SSIM for each run vs reference
  no_injection/
    output-<prompt>.png         # Generated image
    metadata.json               # Full config + strategy info
    config.yaml                 # Base config copy
  baseline_hard/
    output-<prompt>.png
    metadata.json
  adaptive_pd/
    output-<prompt>.png
    metadata.json
    alpha_history.json          # Per-timestep alpha values (adaptive only)
  ...
```

---

## Metrics

**File:** `experiment/metrics.py`

| Function | Description |
|----------|-------------|
| `compute_lpips(img1, img2, net='alex')` | LPIPS perceptual distance (lower = more similar) |
| `compute_ssim(img1, img2)` | Structural similarity (higher = more similar) |
| `compute_metrics(img1, img2)` | Returns `{"lpips": float, "ssim": float}` |
| `load_image_tensor(path, device)` | Load PNG/JPG as `[1, 3, H, W]` tensor in [0, 1] |
| `make_comparison_grid(images_dict, nrow)` | Creates labeled PIL grid from `{name: tensor}` dict |

All image tensors are expected in shape `[1, 3, H, W]` with values in `[0, 1]`.

---

## Experiment Config Format

Experiments are defined in YAML files. See `experiment/configs/ablation_sweep.yaml`
for the full example.

### Top-level fields

```yaml
base_config: config_pnp.yaml       # Path to base PnP config
output_root: experiments/sweep_001  # Output directory
seed: 1                             # Seed override (optional)
```

### Run definitions

Each run has a `name`, `strategy`, and strategy-specific parameters.

#### NoInjection

```yaml
- name: no_injection
  strategy: NoInjection
```

#### HardInjection

```yaml
- name: baseline_hard
  strategy: HardInjection
  params:
    attn_cutoff: 0.5
    conv_cutoff: 0.8
```

#### BlendedInjection

```yaml
- name: blended_linear
  strategy: BlendedInjection
  attn_schedule:
    type: LinearDecaySchedule
    params: {start_alpha: 1.0, end_alpha: 0.0, cutoff: 0.5}
  conv_schedule:
    type: CosineDecaySchedule
    params: {start_alpha: 1.0, cutoff: 0.8}
```

Available schedule types: `ConstantSchedule`, `StepSchedule`, `LinearDecaySchedule`,
`CosineDecaySchedule`.

#### Adaptive (add to any strategy)

```yaml
- name: adaptive_pd
  strategy: BlendedInjection
  attn_schedule:
    type: ConstantSchedule
    params: {alpha: 0.8}
  conv_schedule:
    type: ConstantSchedule
    params: {alpha: 0.8}
  adaptive_params:
    kp: 0.5
    kd: 0.1
    target_deviation: 0.1
```

---

## Writing Custom Schedules and Strategies

### Custom Schedule

Subclass `Schedule` and implement `get_value(progress)`:

```python
from injection.schedules import Schedule

class ExponentialDecaySchedule(Schedule):
    def __init__(self, start_alpha=1.0, decay_rate=3.0, cutoff=1.0):
        self.start_alpha = start_alpha
        self.decay_rate = decay_rate
        self.cutoff = cutoff

    def get_value(self, progress):
        if progress >= self.cutoff:
            return 0.0
        import math
        t = progress / self.cutoff
        return self.start_alpha * math.exp(-self.decay_rate * t)
```

To use it in YAML configs, add it to `SCHEDULE_MAP` in `experiment/runner.py`.

### Custom Strategy

Subclass `InjectionStrategy` and implement `get_alpha(progress, layer_type)`:

```python
from injection.strategies import InjectionStrategy

class LayerSelectiveInjection(InjectionStrategy):
    """Only inject attention, never conv."""
    def __init__(self, attn_alpha=1.0, cutoff=0.5):
        self.attn_alpha = attn_alpha
        self.cutoff = cutoff

    def get_alpha(self, progress, layer_type):
        if layer_type == "attention" and progress < self.cutoff:
            return self.attn_alpha
        return 0.0
```

### Programmatic Use (without YAML)

```python
from pnp_controlled import PNPControlled
from injection import InjectionController, BlendedInjection, LinearDecaySchedule, CosineDecaySchedule
from pnp_utils import seed_everything
import yaml

with open("config_pnp.yaml") as f:
    config = yaml.safe_load(f)

strategy = BlendedInjection(
    attn_schedule=LinearDecaySchedule(1.0, 0.0, 0.6),
    conv_schedule=CosineDecaySchedule(1.0, 0.8),
)
controller = InjectionController(strategy, total_timesteps=None)  # set during init

seed_everything(config["seed"])
pnp = PNPControlled(config, controller=controller)
output = pnp.run_pnp()
```

---

## Troubleshooting

### Output matches original PnP exactly

This is expected when using `HardInjection` with the same cutoffs as the original
config. It confirms the framework is correctly reproducing the baseline.

### "Missing latents at t ..." error

Run preprocessing first:

```bash
python preprocess.py --data_path <your_image> --inversion_prompt "<description>"
```

### Metrics fail to compute

Install the metric dependencies:

```bash
pip install lpips torchmetrics
```

Or use `--no-metrics` to skip metric computation.

### CUDA out of memory

The framework does not add significant memory overhead. If you're running out of
memory, try reducing `n_timesteps` in the config or using a smaller model version.

### Adaptive controller oscillates

Reduce `kp` and `kd` values. Start with `kp=0.1, kd=0.01` and increase gradually.
Check `alpha_history.json` to inspect the alpha trajectory over timesteps.
