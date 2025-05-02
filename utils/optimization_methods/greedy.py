import numpy as np
import pandas as pd

# Constants for greedy optimization
TOTAL_SHELF_SPACE = 1000   # Total shelf capacity
LP_MAX_BUFFER     = 1.2    # Proportional buffer multiplier for sales-based allocation
LP_MIN_STOCK_MIN  = 5      # Minimum units allocated per product in base allocation


def optimize_greedy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick greedy allocation strategy:
      1. Give each SKU a base allocation of LP_MIN_STOCK_MIN units.
      2. Compute remaining space (TOTAL_SHELF_SPACE minus base allocations).
      3. Determine each SKU's "desired extra" units up to floor(sales * LP_MAX_BUFFER).
      4. If total desired extra <= remaining space, allocate full desired extras.
      5. Otherwise, distribute remaining space proportionally to desired extras,
         then allocate any leftover units one-by-one to SKUs with highest profit.

    Returns:
        DataFrame with columns ['Product_Name', 'Category', 'Allocated_Space', 'Profit_Per_Unit'].
    """
    # Number of SKUs
    n = len(df)

    # 1. Base allocation: assign minimum stock to each SKU
    base_alloc = np.full(n, LP_MIN_STOCK_MIN, dtype=int)

    # 2. Remaining shelf space
    remaining = TOTAL_SHELF_SPACE - base_alloc.sum()
    if remaining <= 0:
        # Not enough space for extras: return base allocations
        df_out = df.copy()
        df_out['Allocated_Space'] = base_alloc
        return df_out[['Product_Name', 'Category', 'Allocated_Space', 'Profit_Per_Unit']]

    # 3. Compute desired extra per SKU: up to floor(sales * LP_MAX_BUFFER) minus base
    max_allowed = np.floor(df['Sales_Last_30_Days'].to_numpy() * LP_MAX_BUFFER).astype(int)
    desired_extra = np.clip(max_allowed - LP_MIN_STOCK_MIN, 0, None)

    total_desired = desired_extra.sum()
    if total_desired <= remaining:
        # 4. Enough space: allocate all desired extras
        final_alloc = base_alloc + desired_extra
    else:
        # 5a. Proportional distribution of remaining space
        proportion = remaining / total_desired
        final_alloc = base_alloc + np.floor(desired_extra * proportion).astype(int)

        # 5b. Allocate any leftover units by descending profit
        leftover = TOTAL_SHELF_SPACE - final_alloc.sum()
        if leftover > 0:
            profit_indices = np.argsort(-df['Profit_Per_Unit'].to_numpy())
            for idx in profit_indices:
                if leftover == 0:
                    break
                final_alloc[idx] += 1
                leftover -= 1

    # Build result DataFrame with allocations
    df_out = df.copy()
    df_out['Allocated_Space'] = final_alloc
    return df_out[['Product_Name', 'Category', 'Allocated_Space', 'Profit_Per_Unit']]