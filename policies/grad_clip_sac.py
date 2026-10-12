"""SAC with optional gradient clipping for diagnostics."""
from stable_baselines3 import SAC


class GradClipSAC(SAC):
    """SAC subclass that clips parameter gradients after each update."""

    def __init__(self, *args, max_grad_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_grad_norm = float(max_grad_norm)

    def train(self, gradient_steps, batch_size=None):
        super().train(gradient_steps, batch_size=batch_size)

        for module in (self.actor, self.critic):
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(
                        -self.max_grad_norm,
                        self.max_grad_norm,
                    )