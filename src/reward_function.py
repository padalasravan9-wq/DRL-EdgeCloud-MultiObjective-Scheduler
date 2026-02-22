"""
Adaptive Multi-Objective Reward Function
Matches the formulation in the Scientific Reports manuscript
"""

import numpy as np


class AdaptiveReward:
    def __init__(self, config):
        # Base weights
        rw = config["reward_weights"]
        self.base_latency = rw["latency"]
        self.base_energy = rw["energy"]
        self.base_sla = rw["sla"]
        self.base_overload = rw["overload"]

        # Adaptive rules
        ar = config["adaptive_rules"]
        self.peak_load_threshold = ar["peak_load_threshold"]
        self.offpeak_threshold = ar["offpeak_threshold"]
        self.sla_boost = ar["sla_boost"]
        self.overload_boost = ar["overload_boost"]
        self.energy_boost = ar["energy_boost"]
        self.priority_boost = ar["priority_boost"]

    # =====================================================
    # Dynamic weight adaptation
    # =====================================================
    def _adapt_weights(self, state):
        """
        Adjust weights based on system condition.
        state: dictionary from environment
        """
        latency_w = self.base_latency
        energy_w = self.base_energy
        sla_w = self.base_sla
        overload_w = self.base_overload

        cpu_util = state["cpu_util"]
        sla_violation = state["sla"]
        priority = state["priority"]

        # Peak load → emphasize SLA & overload
        if cpu_util >= self.peak_load_threshold:
            sla_w *= self.sla_boost
            overload_w *= self.overload_boost

        # Off-peak → emphasize energy saving
        elif cpu_util <= self.offpeak_threshold:
            energy_w *= self.energy_boost

        # High-priority tasks → boost latency weight
        if priority > 0.7:
            latency_w *= self.priority_boost

        # Normalize weights (important for stability)
        total = latency_w + energy_w + sla_w + overload_w
        latency_w /= total
        energy_w /= total
        sla_w /= total
        overload_w /= total

        return latency_w, energy_w, sla_w, overload_w

    # =====================================================
    # Compute reward
    # =====================================================
    def compute(self, state):
        """
        state keys expected:
        latency, energy, sla, overload, cpu_util, priority
        """

        latency = state["latency"]
        energy = state["energy"]
        sla = state["sla"]
        overload = state["overload"]

        # Adapt weights
        w_latency, w_energy, w_sla, w_overload = self._adapt_weights(state)

        # Multi-objective penalty (negative reward)
        reward = -(
            w_latency * latency
            + w_energy * energy
            + w_sla * sla
            + w_overload * overload
        )

        return float(reward)


# =========================================================
# Simple wrapper function (easy integration)
# =========================================================
def compute_reward(state, config):
    """
    Convenience function for quick use.
    """
    reward_obj = AdaptiveReward(config)
    return reward_obj.compute(state)
