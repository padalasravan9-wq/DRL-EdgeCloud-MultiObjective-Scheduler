"""
Main training script for DRL Edge–Cloud Scheduler
Reviewer-friendly and reproducible
"""

import os
import yaml
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from dqn_agent import DQNAgent
from environment import EdgeCloudEnv
from reward_function import AdaptiveReward
from scheduler import EdgeCloudScheduler


# =========================================================
# Utilities
# =========================================================
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_seeds(seed_file):
    seeds = []
    if os.path.exists(seed_file):
        with open(seed_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    seeds.append(int(line))
    return seeds if seeds else [42]


# =========================================================
# Training Loop
# =========================================================
def train(config):

    # ----- reproducibility -----
    seed_file = config["reproducibility"]["seed_file"]
    seeds = load_seeds(seed_file)

    results = []

    for run_id, seed in enumerate(seeds):

        print(f"\n=== Run {run_id+1} | Seed {seed} ===")
        set_global_seed(seed)

        # ----- environment -----
        env = EdgeCloudEnv(config, seed=seed)

        state_dim = env.observation_space
        action_dim = env.action_space

        # ----- agent -----
        agent = DQNAgent(
            state_dim,
            action_dim,
            config,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # ----- reward -----
        reward_obj = AdaptiveReward(config)

        # ----- scheduler -----
        scheduler = EdgeCloudScheduler(agent, env, reward_obj, config)

        num_episodes = config["training"]["episodes"]
        max_steps = config["training"]["max_steps_per_episode"]

        episode_rewards = []

        # ================= TRAIN =================
        for ep in tqdm(range(num_episodes), desc=f"Training Run {run_id+1}"):

            stats = scheduler.run_episode(max_steps=max_steps, mode="dqn")
            episode_rewards.append(stats["episode_reward"])

        # ----- collect run statistics -----
        results.append({
            "seed": seed,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "final_reward": float(episode_rewards[-1]),
        })

    return results


# =========================================================
# Save Results
# =========================================================
def save_results(results, config):
    out_dir = config["logging"]["results_path"]
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(results)
    out_file = os.path.join(out_dir, "sample_output.csv")
    df.to_csv(out_file, index=False)

    print(f"\nResults saved to: {out_file}")


# =========================================================
# Main
# =========================================================
def main():
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "config",
        "hyperparameters.yaml",
    )

    config = load_config(config_path)

    print("Starting training...")
    results = train(config)
    save_results(results, config)
    print("Training complete.")


if __name__ == "__main__":
    main()
