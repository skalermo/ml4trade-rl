import gymnasium
import numpy as np


class DoNothingAgent:
    def __init__(self, env: gymnasium.Env, *args, **kwargs):
        self.env = env
        self.action_space = env.action_space

    def learn(self, total_timesteps: int, *args, **kwargs):
        self.env.reset()
        for _ in range(total_timesteps):
            self.env.step(self.action_space.sample())

    def predict(self, *args, **kwargs):
        length = len(self.action_space.sample())
        action = np.zeros(length)
        action[:] = -np.inf
        return action, None


class RuleBasedAgent:
    def __init__(self, env: gymnasium.Env, cfg, *args, **kwargs):
        self.env = env
        self.action_space = env.action_space
        self.cfg = cfg

    def learn(self, total_timesteps: int, *args, **kwargs):
        self.env.reset()
        for _ in range(total_timesteps):
            self.env.step(self.action_space.sample())

    def predict(self, obs, **kwargs):
        market_forecast = obs[:24]
        wind_forecast = obs[24:48]
        solar_forecast = obs[48:72]
        bat_forecast = obs[-1]

        amounts_to_buy = [0] * 24
        amounts_to_sell = [0] * 24

        energy_generated = [self._energy_generated_MWh(w, s) for w, s in zip(wind_forecast, solar_forecast)]
        # energy_consumed = [self._energy_consumed_MWh(h, households_number=self.cfg.env.households) * 1.15 for h in range(0, 24)]
        energy_consumed = [self._energy_consumed_MWh(h, households_number=self.cfg.env.households) * 1.10 for h in range(0, 24)]
        energy_surplus = [g - c for g, c in zip(energy_generated, energy_consumed)]

        for h in range(0, 24):
            amount = energy_surplus[h]
            # amounts_to_sell[h] = self.cfg.env.bat_cap * 0.001
            if amount > 0:
                amounts_to_sell[h] = amount
            else:
                amounts_to_buy[h] = -amount

        bottom_price_hour = np.argmin(market_forecast[:12])
        top_price_hour = bottom_price_hour + np.argmax(market_forecast[bottom_price_hour:])

        # refill battery
        amounts_to_buy[bottom_price_hour] += (self.cfg.env.bat_cap - bat_forecast) / 1.75 * 1.9
        # amounts_to_buy[bottom_price_hour] += self.cfg.env.bat_cap * 0.5

        # sell 80% of battery energy
        # amounts_to_sell[top_price_hour] += self.cfg.env.bat_cap * 0.75
        amounts_to_sell[top_price_hour] += self.cfg.env.bat_cap * 0.9
        # amounts_to_sell[top_price_hour] += self.cfg.env.bat_cap * 0.5

        actions = amounts_to_buy + amounts_to_sell + [1_000_000] * 24 + [0] * 24
        return np.array(actions), None

    def _energy_generated_MWh(self, wind_input: float, solar_input: int):
        wind_speed = wind_input * self.cfg.env.max_wind_speed
        if wind_speed > self.cfg.env.max_wind_speed or wind_speed < 0:
            wind_power = 0
        else:
            wind_power = wind_speed * self.cfg.env.max_wind_power / self.cfg.env.max_wind_speed

        cloudiness = solar_input * 8
        if cloudiness == 9:
            cloudiness = 8
        solar_power = self.cfg.env.max_solar_power * (1 - cloudiness / 8) * self.cfg.env.solar_efficiency

        return wind_power + solar_power

    def _energy_consumed_MWh(self, hour: int, households_number: int):
        energy_consumption_MWh = [
            0.000189, 0.000184, 0.000181, 0.000181, 0.000182, 0.000185,
            0.000194, 0.000222, 0.000238, 0.000246, 0.000246, 0.000246,
            0.000248, 0.000249, 0.000246, 0.000244, 0.000244, 0.000244,
            0.000243, 0.000246, 0.000246, 0.000242, 0.000225, 0.000208
        ]

        return energy_consumption_MWh[hour] * households_number
