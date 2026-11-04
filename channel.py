"""Модель каналу зв'язку за CHANNEL_SPEC_1.md (використовується до отримання офіційного файла)."""
import random

class Channel:
    def __init__(self, scenario: dict, seed: int = 42):
        self.loss = float(scenario.get("loss_probability", 0.0))
        burst_cfg = scenario.get("втрати_пачками", {})
        self.B = float(burst_cfg.get("середня_довжина_пачки", scenario.get("burst_mean", 1.0)))
        self.delay_cfg = scenario.get("delay_ms", 20)
        self.bandwidth_bps = float(scenario.get("bandwidth_bps", 64000))
        self.transmission_budget_s = float(scenario.get("transmission_budget_s", 60.0))

        if self.loss > 0.0:
            self.p_gb = self.loss / (self.loss + self.B * (1.0 - self.loss))
        else:
            self.p_gb = 0.0

        self.rng = random.Random(seed)
        self.clock = 0.0
        self.in_burst = 0
        self.sent = 0
        self.lost_count = 0
        self.over_budget = 0

    def _sample_delay_ms(self) -> float:
        if isinstance(self.delay_cfg, (list, tuple)) and len(self.delay_cfg) == 2:
            return self.rng.uniform(float(self.delay_cfg[0]), float(self.delay_cfg[1]))
        return float(self.delay_cfg)

    def send(self, packet: bytes) -> tuple[bool, float | None]:
        bits = 8 * len(packet)
        transmission_time = bits / self.bandwidth_bps

        if self.clock + transmission_time > self.transmission_budget_s:
            self.over_budget += 1
            return False, None

        self.clock += transmission_time
        self.sent += 1

        if self.in_burst > 0:
            self.in_burst -= 1
            lost = True
        elif self.loss > 0.0 and self.rng.random() < self.p_gb:
            length = 1
            if self.B > 1.0:
                while self.rng.random() > (1.0 / self.B):
                    length += 1
            self.in_burst = length - 1
            lost = True
        else:
            lost = False

        if lost:
            self.lost_count += 1
            return False, None

        arrival_time = self.clock + self._sample_delay_ms() / 1000.0
        return True, arrival_time

    def stats(self) -> dict:
        total_attempted = self.sent + self.over_budget
        return {
            "надіслано": self.sent,
            "втрачено": self.lost_count,
            "не_вміщено_в_бюджет": self.over_budget,
            "частка_втрат": (self.lost_count / self.sent) if self.sent > 0 else 0.0,
            "частка_недоставлених": (
                (self.lost_count + self.over_budget) / total_attempted
                if total_attempted > 0
                else 0.0
            ),
            "час_каналу_с": self.clock,
            "бюджет_с": self.transmission_budget_s,
        }
