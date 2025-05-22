import pulp
import pandas as pd
import numpy as np
import streamlit as st

# Updated Constants for improved allocation
LP_MAX_BUFFER = 2.0   # allow up to 200% of recent sales
LP_MIN_BUFFER = 0.2   # min proportion of recent sales
# (Removed the absolute min constraint for more flexibility)
LAMBDA = 0.2            # reduced penalty weight to encourage allocations to approach ideal


def optimize_lp(df: pd.DataFrame, total_space: int) -> pd.DataFrame:
    """
    LP that maximizes:  ∑(Profit_i * x_i)  –  LAMBDA * ∑(Profit_i * short_i)
    subject to:
      • ∑ x_i ≤ total_space
      • 0 ≤ x_i ≤ sales_i * LP_MAX_BUFFER
      • x_i ≥ sales_i * LP_MIN_BUFFER
      • short_i ≥ ideal_i − x_i
      • short_i ≥ 0

    where 
      ideal_i = sales_i / Σ sales_j * total_space,
      and short_i is the positive shortfall relative to that ideal.

    These updated parameters allow higher allocation (and thus profit) while reducing the penalty on shortfall.
    """

    # 1) Aggregate per SKU
    grouped = (
        df.groupby(['Product_Name', 'Category'], as_index=False)
          .agg({
              'Sales_Last_30_Days': 'sum',
              'Profit_Per_Unit': 'mean'
          })
    )
    n = len(grouped)
    sales = grouped['Sales_Last_30_Days'].to_numpy()
    profits = grouped['Profit_Per_Unit'].to_numpy()

    # Precompute ideal allocation based on sales share
    total_sales = sales.sum()
    if total_sales > 0:
        ideal_alloc = sales / total_sales * total_space
    else:
        ideal_alloc = np.zeros(n)

    # 2) Build LP problem
    prob = pulp.LpProblem("BalancedShelfLP", pulp.LpMaximize)

    # Decision variables: x_i = allocation, short_i = shortfall relative to ideal allocation
    x_vars = [pulp.LpVariable(
        f"x_{i}", lowBound=0, cat='Integer') for i in range(n)]
    short_vars = [pulp.LpVariable(
        f"s_{i}", lowBound=0, cat='Continuous') for i in range(n)]

    # Objective: maximize profit minus penalty on shortfall
    prob += (
        pulp.lpSum(x_vars[i] * profits[i] for i in range(n))
        - LAMBDA * pulp.lpSum(short_vars[i] * profits[i] for i in range(n))
    )

    # Total space constraint
    prob += pulp.lpSum(x_vars) <= total_space

    # SKU-level constraints
    for i in range(n):
        # Do not allocate more than LP_MAX_BUFFER times recent sales
        prob += x_vars[i] <= sales[i] * LP_MAX_BUFFER
        # Ensure a minimum allocation (as a fraction of recent sales)
        prob += x_vars[i] >= sales[i] * LP_MIN_BUFFER
        # Define shortfall: short_vars[i] covers any gap from the ideal allocation
        prob += short_vars[i] >= ideal_alloc[i] - x_vars[i]

    # 3) Solve the LP
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != 'Optimal':
        st.warning(f"LP status: {pulp.LpStatus[status]}")

    # 4) Extract results
    alloc = [int(x_vars[i].value() or 0) for i in range(n)]
    short = [float(short_vars[i].value() or 0) for i in range(n)]
    lost_rev = [short[i] * profits[i] for i in range(n)]
    total_profit = [alloc[i] * profits[i] for i in range(n)]

    # 5) Build output DataFrame
    out = grouped.copy()
    out['Allocated_Units'] = alloc
    out['Shortfall_Units'] = np.ceil(short).astype(int)
    out['Lost_Revenue'] = lost_rev
    out['Achieved_Profit'] = total_profit
    out['Objective_Value'] = out['Achieved_Profit'] - \
        LAMBDA * out['Lost_Revenue']

    return out[[
        'Product_Name', 'Category',
        'Sales_Last_30_Days', 'Profit_Per_Unit',
        'Allocated_Units', 'Shortfall_Units', 'Lost_Revenue', 'Achieved_Profit', 'Objective_Value'
    ]]
