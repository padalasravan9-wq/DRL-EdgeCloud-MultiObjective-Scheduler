"""
Edge–Cloud Scheduling Environment
Compatible with DQNAgent and hyperparameters.yaml
"""

import numpy as np
import random


class EdgeCloudEnv:
    def __init__(self, config, seed=42):
        self.config = config
        self.seed(seed)

        # Environment parameters
        env_cfg = config["environment"]

        self.num_edge = env_cfg["num_edge_devices"]
        self.num_cloud = env_cfg["num_cloud_servers"]
        self.max_queue = env_cfg["max_tasks_in_queue"]

        self.latency_range = env_cfg["latency_range_ms"]
        self.bandwidth_range = env_cfg["bandwidth_range_mbps"]

        # State dimension (matches paper conceptually)
        self.state_dim = 8
        self.action_dim = self.num_edge + self.num_cloud

        self.reset()

    # =====================================================
    # Seeding for reproducibility
    # =====================================================
    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    # =====================================================
    # Reset environment
    # =====================================================
    def reset(self):
        self.queue_length = np.random.randint(0, self.max_queue)
        self.cpu_util = np.random.uniform(0.2, 0.7)
        self.bandwidth = np.random.uniform(*self.bandwidth_range)
        self.latency = np.random.uniform(*self.latency_range)

        self.energy = np.random.uniform(3.5, 6.0)
        self.sla_violation = np.random.uniform(0.05, 0.15)
        self.overload = np.random.uniform(0.1, 0.5)
        self.priority = np.random.uniform(0, 1)

        return self._get_state()

    # =====================================================
    # Step function
    # =====================================================
    def step(self, action, reward_fn=None):
        """
        action: selected node index
        reward_fn: function(state_dict) -> reward
        """

        # --- simulate workload dynamics ---
        arrival = np.random.poisson(lam=5)
        self.queue_length = max(0, min(self.max_queue, self.queue_length + arrival - 1))

        # simulate resource fluctuation
        self.cpu_util = np.clip(
            self.cpu_util + np.random.normal(0, 0.05), 0, 1
        )
        self.bandwidth = np.clip(
            self.bandwidth + np.random.normal(0, 20),
            self.bandwidth_range[0],
            self.bandwidth_range[1],
        )
        self.latency = np.clip(
            self.latency + np.random.normal(0, 5),
            self.latency_range[0],
            self.latency_range[1],
        )

        # energy & SLA dynamics
        self.energy = np.clip(self.energy + np.random.normal(0, 0.2), 3.0, 7.0)
        self.sla_violation = np.clip(
            self.sla_violation + np.random.normal(0, 0.01), 0, 1
        )
        self.overload = np.clip(
            self.overload + np.random.normal(0, 0.03), 0, 1
        )

        # --- compute reward ---
        state_dict = {
            "latency": self.latency,
            "energy": self.energy,
            "sla": self.sla_violation,
            "overload": self.overload,
            "priority": self.priority,
            "cpu_util": self.cpu_util,
            "queue_length": self.queue_length,
        }

        if reward_fn is None:
            reward = -(
                0.35 * self.latency
                + 0.25 * self.energy
                + 0.30 * self.sla_violation
                + 0.10 * self.overload
            )
        else:
            reward = reward_fn(state_dict)

        # --- termination condition ---
        done = False
        if self.sla_violation > 0.9:
            done = True

        return self._get_state(), reward, done, {}

    # =====================================================
    # State vector
    # =====================================================
    def _get_state(self):
        state = np.array(
            [
                self.queue_length / self.max_queue,
                self.cpu_util,
                self.bandwidth / self.bandwidth_range[1],
                self.latency / self.latency_range[1],
                self.energy / 10.0,
                self.sla_violation,
                self.overload,
                self.priority,
            ],
            dtype=np.float32,
        )
        return state

    # =====================================================
    # Helper properties
    # =====================================================
    @property
    def observation_space(self):
        return self.state_dim

    @property
    def action_space(self):
        return self.action_dim
