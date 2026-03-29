import os
import torch
import torchvision.transforms as T
from tqdm import tqdm

from pnp import PNP
from pnp_utils import register_time, seed_everything
from injection.controller import InjectionController
from injection.strategies import HardInjection
from injection.hooks import register_attention_control, register_conv_control


class PNPControlled(PNP):
    """
    Extended PNP that uses InjectionController for flexible injection.
    Inherits model loading, text embedding, decoding from PNP.
    Overrides init_pnp and run_pnp to use the controller-based hooks.
    """

    def __init__(self, config, controller=None):
        """
        Args:
            config: dict from YAML config.
            controller: InjectionController instance. If None, builds a
                        HardInjection controller from config for backward compatibility.
        """
        super().__init__(config)
        self.controller = controller

    def init_pnp(self, conv_injection_t=None, qk_injection_t=None):
        """Uses controller-based hooks instead of schedule-based."""
        if self.controller is None:
            strategy = HardInjection(
                attn_cutoff=self.config.get("pnp_attn_t", 0.5),
                conv_cutoff=self.config.get("pnp_f_t", 0.8),
            )
            self.controller = InjectionController(strategy, self.scheduler.timesteps)

        # Set timesteps on controller if not yet set (e.g., when passed externally)
        if self.controller.total_timesteps is None:
            self.controller.total_timesteps = self.scheduler.timesteps
            self.controller.n_steps = len(self.scheduler.timesteps)
            self.controller._timestep_to_index = {
                int(t): i for i, t in enumerate(self.scheduler.timesteps)
            }

        register_attention_control(self, self.controller)
        register_conv_control(self, self.controller)

    def run_pnp(self):
        self.init_pnp()
        edited_img = self.sample_loop(self.eps)
        return edited_img

    def sample_loop(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t)
                self.controller.step_update(t)

            decoded_latent = self.decode_latent(x)
            T.ToPILImage()(decoded_latent[0]).save(
                f'{self.config["output_path"]}/output-{self.config["prompt"]}.png'
            )

        return decoded_latent


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config_pnp.yaml')
    opt = parser.parse_args()

    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["output_path"], exist_ok=True)
    with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)

    seed_everything(config["seed"])
    print(config)
    pnp = PNPControlled(config)
    pnp.run_pnp()
