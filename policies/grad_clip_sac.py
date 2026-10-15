"""SAC with gradient clipping and an optional entropy-coefficient floor.

The floor prevents the automatic entropy tuner from driving log_alpha to
extremely negative values. Without a floor, SAC can collapse to a
near-deterministic policy within the first few thousand gradient steps on
reward scales where the natural policy entropy falls below the target.

The log_alpha tensor is located via the entropy coefficient optimizer's
parameter list, which is the authoritative source in all Stable-Baselines3
versions that support automatic entropy tuning.
"""

import numpy as np
import torch
import torch.nn.utils as torch_utils
from stable_baselines3 import SAC


class GradClipSAC(SAC):
    """SAC with gradient-norm clipping and an optional alpha floor."""

    def __init__(
        self,
        *args,
        max_grad_norm: float = 1.0,
        alpha_floor: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_grad_norm = float(max_grad_norm)
        self.alpha_floor = float(alpha_floor)

        actor_opt = self.actor.optimizer
        critic_opt = self.critic.optimizer

        actor_orig_step = actor_opt.step
        critic_orig_step = critic_opt.step

        def _clipped_actor_step(closure=None):
            if closure is not None:
                raise RuntimeError("Closure not supported.")
            torch_utils.clip_grad_norm_(
                [p for g in actor_opt.param_groups for p in g["params"]],
                self.max_grad_norm,
            )
            return actor_orig_step()

        def _clipped_critic_step(closure=None):
            if closure is not None:
                raise RuntimeError("Closure not supported.")
            torch_utils.clip_grad_norm_(
                [p for g in critic_opt.param_groups for p in g["params"]],
                self.max_grad_norm,
            )
            return critic_orig_step()

        actor_opt.step = _clipped_actor_step
        critic_opt.step = _clipped_critic_step

    # ------------------------------------------------------------------
    # Log-alpha location
    # ------------------------------------------------------------------

    def _find_log_alpha_tensor(self):
        """Return the log_alpha parameter tensor, or None if not found.

        SB3 stores log_alpha inside the entropy coefficient optimizer's
        parameter list when ent_coef='auto_*'. This method inspects that
        optimizer first, then falls back to the model and actor attributes.
        """
        optimizer = getattr(self, "ent_coef_optimizer", None)
        if optimizer is not None:
            for group in optimizer.param_groups:
                for param in group.get("params", []):
                    if isinstance(param, torch.Tensor) and param.numel() == 1:
                        return param

        for candidate in (
            getattr(self, "log_alpha", None),
            getattr(getattr(self, "actor", None), "log_alpha", None),
        ):
            if isinstance(candidate, torch.Tensor):
                return candidate

        return None

    # ------------------------------------------------------------------
    # Train override
    # ------------------------------------------------------------------

    def train(self, gradient_steps, batch_size=None):
        super().train(gradient_steps, batch_size=batch_size)

        if self.alpha_floor <= 0.0:
            return

        log_alpha = self._find_log_alpha_tensor()
        if log_alpha is None:
            # ent_coef is not 'auto_*'; the floor is not applicable.
            return

        with torch.no_grad():
            min_log_alpha = float(np.log(self.alpha_floor))
            log_alpha.data.clamp_(min=min_log_alpha)