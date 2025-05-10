# import os
# import torch
# import numpy as np
# import pandas as pd
# import gymnasium as gym
# from gymnasium import spaces
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.vec_env import SubprocVecEnv

# # ───── GPU CHECK ─────
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[ppo_speedup] using device: {device}")

# # ───── Fast Shelf‐Space Env ─────
# class ShelfSpaceEnv(gym.Env):
#     """
#     Single‐step env:
#     - obs: flattened [Profit, Sales] for each SKU
#     - act: floats in [0,1]^n → normalized → integer allocations summing to total_space
#     - reward: total profit
#     """
#     metadata = {"render_modes": []}

#     def __init__(self, features: np.ndarray, total_space: int):
#         super().__init__()
#         self.features    = features               # shape (n,2)
#         self.total_space = total_space
#         self.n           = features.shape[0]
#         n_features       = features.shape[1]

#         # state/action spaces
#         # now obs_space matches features.flatten() length
#         self.observation_space = spaces.Box(
#             low=0,
#             high=np.inf,
#             shape=(self.n * n_features,),
#             dtype=np.float32
#         )
#         # action is still 1 per SKU
#         self.action_space = spaces.Box(
#             low=0.0,
#             high=1.0,
#             shape=(self.n,),
#             dtype=np.float32
#         )

#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         return self.features.flatten(), {}

#     def step(self, action):
#         a = np.clip(action, 0, 1)
#         w = a / (a.sum() if a.sum() > 0 else 1.0)
#         alloc_frac = w * self.total_space
#         base = np.floor(alloc_frac).astype(int)
#         leftover = self.total_space - base.sum()
#         if leftover > 0:
#             rems = alloc_frac - base
#             for idx in np.argsort(-rems)[:leftover]:
#                 base[idx] += 1

#         reward = float((base * self.features[:,0]).sum())
#         return self.features.flatten(), reward, True, False, {"allocation": base}

#     def render(self, mode="human"):
#         pass


# # ───── Heuristic (Profit-Weighted) Warm-Start ─────
# def profit_weighted_allocation(df, total_space):
#     """
#     Fast baseline: floor(score/total_score*space) + remainder.
#     Returns an np.array of ints.
#     """
#     score = df["Profit_Per_Unit"] * df["Sales_Last_30_Days"]
#     total = score.sum()
#     if total <= 0:
#         base = np.zeros(len(df), dtype=int)
#     else:
#         frac = score / total * total_space
#         base = np.floor(frac).astype(int)
#         rem = total_space - base.sum()
#         if rem > 0:
#             r = frac - base
#             for idx in np.argsort(-r)[:rem]:
#                 base[idx] += 1
#     return base


# # ───── Combined Imitation + PPO Fine-Tune ─────
# def optimize_ppo_speedup(
#     df: pd.DataFrame,
#     total_space: int,
#     timesteps: int = 5000,
#     num_envs: int = 4,
#     use_warmup: bool = True
# ) -> pd.DataFrame:
#     """
#     1) Aggregate by Product/Category.
#     2) Compute features & initial heuristic allocation.
#     3) If use_warmup: collect (state, action) pairs and do a quick supervised warm-start.
#     4) Create SubprocVecEnv with num_envs workers.
#     5) Train PPO with GPU, small net, tuned hyperparams.
#     6) Roll out to get final allocation.
#     """
#     # 1) group
#     grouped = (
#         df.groupby(["Product_Name","Category"], as_index=False)
#           .agg({"Sales_Last_30_Days":"sum","Profit_Per_Unit":"mean"})
#     )
#     grouped["Score"] = grouped["Profit_Per_Unit"] * grouped["Sales_Last_30_Days"]
#     features = grouped[["Score"]].to_numpy(np.float32)

#     # 2) initial heuristic
#     init_alloc = profit_weighted_allocation(grouped, total_space)

#     # 3) (Optional) supervised warm-start
#     if use_warmup:
#         import torch.nn as nn
#         from torch.utils.data import Dataset, DataLoader

#         class ShelfDataset(Dataset):
#             def __init__(self, feats, alloc):
#                 self.feats = torch.from_numpy(feats)
#                 alloc_arr = alloc.values if isinstance(alloc, pd.Series) else alloc
#                 self.alloc = torch.from_numpy(np.asarray(alloc_arr)).float()
#             def __len__(self): return len(self.feats)
#             def __getitem__(self, i): return self.feats[i], self.alloc[i]

#         ds = ShelfDataset(features, init_alloc)
#         loader = DataLoader(ds, batch_size=256, shuffle=True)

#         # NOTE: input_dim = features.shape[1] = 1, but we want to keep it flexible
#         # for future use cases (e.g. more features).
#         input_dim = features.shape[1]
#         net = nn.Sequential(
#             nn.Linear(input_dim, 64),  # now matches whatever features.shape[1] is
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         ).to(device)

#         opt = torch.optim.Adam(net.parameters(), lr=1e-3)
#         for epoch in range(3):
#             for x, y in loader:
#                 x, y = x.to(device), y.to(device)/total_space
#                 pred = net(x).squeeze()  # in [0,1]
#                 loss = nn.MSELoss()(pred, y)
#                 opt.zero_grad(); loss.backward(); opt.step()
#         # override policy extractor via SB3 policy_kwargs
#         policy_kwargs = dict(
#             net_arch=[dict(pi=[64,64], vf=[64,64])]
#         )

#     else:
#         policy_kwargs = dict(net_arch=[dict(pi=[64,64], vf=[64,64])])

#     # 4) Parallel envs
#     def make_env():
#         return ShelfSpaceEnv(features, total_space)
#     env = SubprocVecEnv([make_env for _ in range(num_envs)])

#     # 5) PPO on GPU
#     check_env(make_env(), warn=True)
#     model = PPO(
#         "MlpPolicy", env,
#         device=device,
#         policy_kwargs=policy_kwargs,
#         n_steps=256, batch_size=256, n_epochs=3,
#         verbose=1
#     )
#     model.learn(total_timesteps=timesteps)

#     # 6) final rollout
#     raw_env = ShelfSpaceEnv(features, total_space)
#     obs, _ = raw_env.reset()
#     action, _ = model.predict(obs, deterministic=True)
#     _, _, _, _, info = raw_env.step(action)
#     final_alloc = info["allocation"]

#     # merge and return
#     grouped["Allocated_Space"] = final_alloc.astype(int)
#     return grouped[[
#         "Product_Name","Category",
#         "Sales_Last_30_Days","Profit_Per_Unit",
#         "Score","Allocated_Space"
#     ]]











# utils/optimization_methods/ppo_speedup.py

# import numpy as np
# import pandas as pd
# import torch
# import gymnasium as gym
# from gymnasium import spaces
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.env_checker import check_env

# # Balance factor: how strongly to penalize lost revenue
# LAMBDA = 1.0  # 1.0 = equal weight on profit and lost revenue

# class ShelfSpaceEnv(gym.Env):
#     def __init__(self, features: np.ndarray, total_space: int):
#         """
#         features: array of shape (n,2) columns = [Profit_Per_Unit, Sales_Last_30_Days]
#         """
#         super().__init__()
#         self.features    = features
#         self.profits     = features[:, 0]
#         self.sales       = features[:, 1]
#         self.total_space = total_space
#         self.n           = self.features.shape[0]

#         # Observation = flattened feature vector
#         n_features = self.features.shape[1]
#         self.observation_space = spaces.Box(
#             low=0, high=np.inf,
#             shape=(self.n * n_features,),
#             dtype=np.float32
#         )
#         # Action = fraction [0,1] per SKU
#         self.action_space = spaces.Box(
#             low=0.0, high=1.0,
#             shape=(self.n,), dtype=np.float32
#         )

#         # Precompute the "ideal" fractional allocation based purely on sales
#         total_sales = self.sales.sum()
#         if total_sales > 0:
#             self.ideal_frac = self.sales / total_sales * self.total_space
#         else:
#             self.ideal_frac = np.zeros(self.n)

#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         return self.features.flatten(), {}

#     def step(self, action):
#         # 1) Normalize action to fractions
#         a = np.clip(action, 0, 1)
#         if a.sum() > 0:
#             w = a / a.sum()
#         else:
#             w = np.ones(self.n) / self.n

#         # 2) Compute integer allocation
#         alloc_frac = w * self.total_space
#         alloc_base = np.floor(alloc_frac).astype(int)
#         leftover   = self.total_space - alloc_base.sum()
#         if leftover > 0:
#             remainders = alloc_frac - alloc_base
#             for idx in np.argsort(-remainders)[:leftover]:
#                 alloc_base[idx] += 1

#         # 3) Compute total profit
#         total_profit = float((alloc_base * self.profits).sum())

#         # 4) Compute lost units & lost revenue
#         lost_units = np.maximum(self.ideal_frac - alloc_base, 0)
#         total_lost = float((lost_units * self.profits).sum())

#         # 5) Combined reward
#         reward = total_profit - LAMBDA * total_lost

#         # Episode ends immediately
#         terminated = True
#         truncated  = False

#         obs = self.features.flatten()
#         info = {
#             "allocation": alloc_base,
#             "total_profit": total_profit,
#             "total_lost_revenue": total_lost
#         }
#         return obs, reward, terminated, truncated, info

#     def render(self, mode="human"):
#         pass

# def optimize_ppo_speedup(
#     df: pd.DataFrame,
#     total_space: int,
#     timesteps: int = 5000,
#     num_envs: int = 4
# ) -> pd.DataFrame:
#     # 1) Aggregate
#     grouped = (
#         df.groupby(["Product_Name","Category"], as_index=False)
#           .agg({"Sales_Last_30_Days":"sum","Profit_Per_Unit":"mean"})
#     )
#     # 2) Build feature array
#     features = grouped[["Profit_Per_Unit","Sales_Last_30_Days"]].to_numpy(np.float32)

#     # 3) Create & check env
#     raw_env = ShelfSpaceEnv(features, total_space)
#     check_env(raw_env, warn=True)

#     # 4) Parallelize
#     def make_env(): return ShelfSpaceEnv(features, total_space)
#     vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])

#     # 5) Train PPO
#     model = PPO(
#         "MlpPolicy", vec_env,
#         device="auto",
#         n_steps=256, batch_size=256, n_epochs=3,
#         verbose=1
#     )
#     model.learn(total_timesteps=timesteps)

#     # 6) Get final allocation
#     obs, _      = raw_env.reset()
#     action, _   = model.predict(obs, deterministic=True)
#     _, _, _, _, info = raw_env.step(action)
#     alloc = info["allocation"]

#     # 7) Merge back
#     grouped["Allocated_Space"]     = alloc
#     grouped["Total_Profit"]        = info["total_profit"]
#     grouped["Total_Lost_Revenue"]  = info["total_lost_revenue"]
#     grouped["Net_Reward"]          = grouped["Total_Profit"] - LAMBDA*grouped["Total_Lost_Revenue"]

#     return grouped[[
#         "Product_Name","Category",
#         "Sales_Last_30_Days","Profit_Per_Unit",
#         "Allocated_Space","Total_Profit",
#         "Total_Lost_Revenue","Net_Reward"
#     ]]


# utils/optimization_methods/ppo_speedup.py

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

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
    num_envs: int = 4
) -> pd.DataFrame:
    # 1) Aggregate
    grouped = (
        df.groupby(["Product_Name","Category"], as_index=False)
          .agg({"Sales_Last_30_Days":"sum","Profit_Per_Unit":"mean"})
    )
    # 2) Compute balanced score
    grouped["Balanced_Score"] = np.sqrt(
        grouped["Profit_Per_Unit"] * grouped["Sales_Last_30_Days"]
    )

    # 3) Build features array: now 3 cols per SKU
    features = grouped[["Profit_Per_Unit",
                        "Sales_Last_30_Days",
                        "Balanced_Score"]].to_numpy(np.float32)

    # 4) Create & validate raw env
    raw_env = ShelfSpaceEnv(features, total_space)
    check_env(raw_env, warn=True)

    # 5) Parallelize
    def make_env(): return ShelfSpaceEnv(features, total_space)
    vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # 6) Train PPO
    model = PPO(
        "MlpPolicy", vec_env,
        device="auto",
        n_steps=256, batch_size=256, n_epochs=3,
        verbose=1
    )
    model.learn(total_timesteps=timesteps)

    # 7) Roll out one step for final allocation
    obs, _        = raw_env.reset()
    action, _     = model.predict(obs, deterministic=True)
    _, _, _, _, info = raw_env.step(action)
    alloc = info["allocation"]

    # 8) Merge & return
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