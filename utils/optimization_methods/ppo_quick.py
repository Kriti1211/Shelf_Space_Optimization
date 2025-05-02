# import hashlib
# import pandas as pd
# import streamlit as st
# from utils.optimization_methods.reinforcement_learning import optimize_ppo
# from utils.optimization_methods.linear_programming import optimize_lp

# # Constants for PPO optimization
# MAX_SKUS_PPO = 200   # Maximum number of SKUs considered by PPO
# BASE_STEPS = 12000   # Base number of training steps for PPO
# MIN_STEPS = 4000     # Minimum number of training steps for PPO

# def _hash_df(df: pd.DataFrame) -> str:
#     """
#     Generate a SHA-256 hash for the given DataFrame.
#     Used to uniquely identify the DataFrame state for caching purposes.
#     """
#     m = hashlib.sha256()
#     # Hash all values including the index to ensure uniqueness
#     m.update(pd.util.hash_pandas_object(df, index=True).values)
#     return m.hexdigest()

# @st.cache_resource(show_spinner=False)
# def _train_cached(df_hash: str, df_subset: pd.DataFrame, steps: int):
#     """
#     Train or retrieve a cached PPO optimization model on the given subset of data.

#     Parameters:
#     - df_hash: Hash of the DataFrame slice used for caching key.
#     - df_subset: Subset of the DataFrame containing top SKUs.
#     - steps: Number of training steps for PPO.

#     Returns:
#     - Allocation results from optimize_ppo.
#     """
#     return optimize_ppo(df_subset, train_steps=steps)


# def quick_optimize(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Quickly optimize shelf space allocation for products.
#     Uses linear programming for small SKU sets; otherwise uses PPO.

#     Parameters:
#     - df: DataFrame with columns 'Sales_Last_30_Days', 'Product_Name',
#           'Category', and 'Profit_Per_Unit'.

#     Returns:
#     - DataFrame with ['Product_Name', 'Category', 'Allocated_Space', 'Profit_Per_Unit'].
#     """
#     n = len(df)

#     # If we have few products, solve via LP for speed
#     if n <= 30:
#         return optimize_lp(df)

#     # Determine number of SKUs to feed into PPO (capped by MAX_SKUS_PPO)
#     k = min(n, MAX_SKUS_PPO)

#     # Scale training steps with the number of SKUs, enforce minimum
#     steps = max(int(BASE_STEPS * k / MAX_SKUS_PPO), MIN_STEPS)

#     # Select top-k SKUs by recent sales
#     df_top = df.nlargest(k, 'Sales_Last_30_Days').reset_index(drop=True)

#     # Compute a hash of the subset for caching
#     df_hash = _hash_df(df_top)

#     # Run or fetch the cached PPO training session
#     alloc = _train_cached(df_hash, df_top, steps)

#     # Merge the allocation back into the full DataFrame
#     out = df.merge(
#         alloc,
#         on=['Product_Name', 'Category', 'Profit_Per_Unit'],
#         how='left'
#     )

#     # Default allocation for any missing entries
#     out['Allocated_Space'].fillna(5, inplace=True)

#     # Cast allocations to integers
#     out['Allocated_Space'] = out['Allocated_Space'].astype(int)

#     # Return only the relevant columns
#     return out[['Product_Name', 'Category', 'Allocated_Space', 'Profit_Per_Unit']]

# utils/optimization_methods/ppo_quick.py

import hashlib
import pandas as pd
import streamlit as st

from utils.optimization_methods.reinforcement_learning import optimize_ppo
from utils.optimization_methods.linear_programming import optimize_lp

# Constants for PPO optimization
MAX_SKUS_PPO = 200   # Maximum number of SKUs considered by PPO
BASE_STEPS   = 12000 # Base number of training steps for PPO
MIN_STEPS    = 4000  # Minimum number of training steps for PPO

def _hash_df(df: pd.DataFrame) -> str:
    """
    Generate a SHA-256 hash for the given DataFrame slice.
    Used as cache key for PPO training.
    """
    m = hashlib.sha256()
    m.update(pd.util.hash_pandas_object(df, index=True).values)
    return m.hexdigest()

@st.cache_resource(show_spinner=False)
def _train_cached(
    df_hash: str,
    df_subset: pd.DataFrame,
    steps: int,
    total_space: int
) -> pd.DataFrame:
    """
    Train or retrieve a cached PPO optimization model.
    """
    return optimize_ppo(df_subset, train_steps=steps, total_space=total_space)

def quick_optimize(
    df: pd.DataFrame,
    total_space: int
) -> pd.DataFrame:
    """
    Quickly optimize shelf space:
      - If SKU count ≤30: run LP with capacity.
      - Otherwise: train/fetch PPO on top-k SKUs then merge back.
    """
    n = len(df)
    # 1) Small set → LP directly
    if n <= 30:
        return optimize_lp(df)

    # 2) PPO path: pick top-k SKUs by sales
    k = min(n, MAX_SKUS_PPO)
    steps = max(int(BASE_STEPS * k / MAX_SKUS_PPO), MIN_STEPS)

    df_top = df.nlargest(k, 'Sales_Last_30_Days').reset_index(drop=True)
    df_hash = _hash_df(df_top)

    # Train or fetch cached PPO result (respects total_space)
    alloc = _train_cached(df_hash, df_top, steps, total_space)

    # 3) Merge allocations back into full set
    out = df.merge(
        alloc,
        on=['Product_Name', 'Category', 'Profit_Per_Unit'],
        how='left'
    )
    out['Allocated_Space'].fillna(5, inplace=True)
    out['Allocated_Space'] = out['Allocated_Space'].astype(int)

    return out[['Product_Name', 'Category', 'Allocated_Space', 'Profit_Per_Unit']]