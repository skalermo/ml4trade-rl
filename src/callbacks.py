from stable_baselines3.common.callbacks import BaseCallback


class ResampleCallback(BaseCallback):
    def _on_step(self) -> bool:
        self.model.policy.resample()
        return True

    def _on_rollout_start(self) -> None:
        self.model.policy.resample()
