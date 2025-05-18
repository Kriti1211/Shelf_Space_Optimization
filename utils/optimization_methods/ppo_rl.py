import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
import random
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

# Balance factor for lost‐revenue penalty
LAMBDA = 1.0

class ShelfSpaceEnv(gym.Env):
    def __init__(self, features: np.ndarray, total_space: int):
        """
        features: array of shape (n,3) columns = [
           Profit_Per_Unit, Sales_Last_30_Days, Balanced_Score
        ]
        """
        super().__init__()
        self.features    = features
        self.profits     = features[:, 0]
        self.sales       = features[:, 1]
        self.total_space = total_space
        self.n           = self.features.shape[0]

        # --- Observation space: flattened [profit, sales, balanced] for each SKU
        n_feat = self.features.shape[1]  # now 3
        self.observation_space = spaces.Box(
            low=0, high=np.inf,
            shape=(self.n * n_feat,), dtype=np.float32
        )
        # --- Action: one fraction per SKU
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n,), dtype=np.float32
        )

        # Pre‐compute ideal fractional allocation by sales only
        total_sales = self.sales.sum()
        if total_sales > 0:
            self.ideal_frac = self.sales / total_sales * self.total_space
        else:
            self.ideal_frac = np.zeros(self.n)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.features.flatten(), {}

    def step(self, action):
        # Normalize and allocate
        a = np.clip(action, 0, 1)
        w = a / (a.sum() if a.sum() > 0 else 1.0)
        alloc_frac = w * self.total_space
        alloc_base = np.floor(alloc_frac).astype(int)
        leftover   = self.total_space - alloc_base.sum()
        if leftover > 0:
            rems = alloc_frac - alloc_base
            for idx in np.argsort(-rems)[:leftover]:
                alloc_base[idx] += 1

        # Profit & lost‐revenue
        total_profit = float((alloc_base * self.profits).sum())
        lost_units   = np.maximum(self.ideal_frac - alloc_base, 0)
        total_lost   = float((lost_units * self.profits).sum())

        # Combined reward
        reward = total_profit - LAMBDA * total_lost

        obs = self.features.flatten()
        info = {
            "allocation": alloc_base,
            "total_profit": total_profit,
            "total_lost_revenue": total_lost
        }
        # single‐step episode
        return obs, reward, True, False, info

    def render(self, mode="human"):
        pass


def optimize_ppo_speedup(
    df: pd.DataFrame,
    total_space: int,
    timesteps: int = 5000,
    num_envs: int = 4,
    seed: int = 42
) -> pd.DataFrame:
    
    """
    Runs a single-step PPO to allocate shelf space.
    Returns a DataFrame with columns:
      Product_Name, Category, Sales_Last_30_Days, Profit_Per_Unit,
      Balanced_Score, Allocated_Space, Total_Profit, Total_Lost_Revenue, Net_Reward
    Deterministic given fixed `seed`.
    """
    # ─── 0) Seed everything ─────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

    # ─── 1) Aggregate & compute balanced score ─────────
    grouped = (
        df.groupby(["Product_Name","Category"], as_index=False)
          .agg({
              "Sales_Last_30_Days": "sum",
              "Profit_Per_Unit":    "mean"
          })
    )
    grouped["Balanced_Score"] = np.sqrt(
        grouped["Sales_Last_30_Days"] * grouped["Profit_Per_Unit"]
    )

    # ─── 2) Build feature matrix ───────────────────────
    features = grouped[[
        "Profit_Per_Unit",
        "Sales_Last_30_Days",
        "Balanced_Score"
    ]].to_numpy(np.float32)

    # ─── 3) Check a single env ─────────────────────────
    raw_env = ShelfSpaceEnv(features, total_space)
    check_env(raw_env, warn=True)

    # ─── 4) DummyVecEnv (no subprocesses) ─
    # Single‐process envs avoid any pickle/spawn issues on Windows
    vec_env = DummyVecEnv([lambda: ShelfSpaceEnv(features, total_space)])

    # ─── 5) Train PPO ───────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        seed=seed,
        device="auto",
        n_steps=256,
        batch_size=256,
        n_epochs=3,
        verbose=0
    )
    model.learn(total_timesteps=timesteps)

    # ─── 6) Single-step rollout ────────────────────────
    obs, _      = raw_env.reset(seed=seed)
    action, _   = model.predict(obs, deterministic=True)
    _, _, _, _, info = raw_env.step(action)
    alloc       = info["allocation"]

    # ─── 7) Attach results & return ────────────────────
    grouped["Allocated_Space"]    = alloc
    grouped["Total_Profit"]       = info["total_profit"]
    grouped["Total_Lost_Revenue"] = info["total_lost_revenue"]
    grouped["Net_Reward"]         = grouped["Total_Profit"] - LAMBDA * grouped["Total_Lost_Revenue"]

    return grouped[[
        "Product_Name","Category",
        "Sales_Last_30_Days","Profit_Per_Unit",
        "Balanced_Score","Allocated_Space",
        "Total_Profit","Total_Lost_Revenue","Net_Reward"
    ]]