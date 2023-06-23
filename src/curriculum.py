from stable_baselines3.common.callbacks import BaseCallback

from ml4trade.domain.battery import Battery


class CurriculumCallback(BaseCallback):
    def __init__(self, iw, eval_iw):
        self.iw = iw
        self.eval_iw = eval_iw
        self.iw.bat_eff_mean = 0.1
        self.eval_iw.bat_eff_mean = 0.1
        super().__init__()

    def _on_step(self) -> bool:
        cur_dist = 0.85 - self.iw.bat_eff_mean
        new_mean = 0.85 - 0.95 * cur_dist
        self.iw.bat_eff_mean = new_mean
        self.eval_iw.bat_eff_mean = new_mean
        self.logger.record("eval/bat_eff_mean", new_mean)
        return True


class ReportBateffCallback(BaseCallback):
    def __init__(self, battery: Battery):
        self.battery = battery
        super().__init__()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self):
        self.logger.record("rollout/bat_eff", self.battery.efficiency)
