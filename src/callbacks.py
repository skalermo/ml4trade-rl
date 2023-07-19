from stable_baselines3.common.callbacks import BaseCallback


class ResampleCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.model.policy.resample()
