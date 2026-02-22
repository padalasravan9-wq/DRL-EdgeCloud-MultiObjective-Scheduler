"""
DQN-based Edge–Cloud Scheduler
Connects environment, agent, and reward function
"""

import numpy as np


class EdgeCloudScheduler:
    def __init__(self, agent, env, reward_obj=None, config=None):
        self.agent = agent
        self.env = env
        self.reward_obj = reward_obj
        self.config = config

        # statistics tracking
        self.episode_reward = 0.0
        self.step_count = 0

    # =====================================================
    # Select action (DQN or fallback)
    # =====================================================
    def select_action(self, state, mode="dqn"):
        """
        mode:
            - "dqn" (default)
            - "greedy" (baseline)
            - "random"
        """

        if mode == "dqn":
            return self.agent.select_action(state)

        elif mode == "random":
            return np.random.randint(self.env.action_space)

        elif mode == "greedy":
            # simple heuristic: prefer cloud when queue high
            queue_level = state[0]  # normalized queue
            if queue_level > 0.7:
                return self.env.num_edge  # first cloud node
            else:
                return np.random.randint(self.env.num_edge)

        else:
            raise ValueError(f"Unknown scheduling mode: {mode}")

    # =====================================================
    # Run one environment step
    # =====================================================
    def step(self, state, mode="dqn"):
        action = self.select_action(state, mode=mode)

        # environment step
        if self.reward_obj is None:
            next_state, reward, done, info = self.env.step(action)
        else:
            next_state, reward, done, info = self.env.step(
                action, reward_fn=self.reward_obj.compute
            )

        # store transition for learning
        if mode == "dqn":
            self.agent.store_transition(state, action, reward, next_state, done)
            loss = self.agent.train_step()
        else:
            loss = None

        self.episode_reward += reward
        self.step_count += 1

        return next_state, reward, done, loss

    # =====================================================
    # Reset episode stats
    # =====================================================
    def reset_episode_stats(self):
        self.episode_reward = 0.0
        self.step_count = 0

    # =====================================================
    # Run full episode
    # =====================================================
    def run_episode(self, max_steps=200, mode="dqn"):
        state = self.env.reset()
        self.reset_episode_stats()

        losses = []

        for _ in range(max_steps):
            next_state, reward, done, loss = self.step(state, mode=mode)
            state = next_state

            if loss is not None:
                losses.append(loss)

            if done:
                break

        avg_loss = np.mean(losses) if losses else None

        return {
            "episode_reward": self.episode_reward,
            "steps": self.step_count,
            "avg_loss": avg_loss,
        }
