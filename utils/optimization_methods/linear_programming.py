import pulp
import pandas as pd
import numpy as np
import streamlit as st

# Constants
LP_MAX_BUFFER    = 1.2   # max proportion of recent sales
LP_MIN_BUFFER    = 0.2   # min proportion of recent sales
LP_MIN_STOCK_MIN = 5     # absolute min per SKU

# Penalty weight: how much we penalize lost revenue per ₹1 lost
LAMBDA = 1.0

def optimize_lp(df: pd.DataFrame, total_space: int) -> pd.DataFrame:
    """
    LP that maximizes:  ∑(Profit_i * x_i)  –  LAMBDA * ∑(Profit_i * short_i)
    subject to:
      • ∑ x_i ≤ total_space
      • 0 ≤ x_i ≤ sales_i * LP_MAX_BUFFER
      • x_i ≥ min(LP_MIN_STOCK_MIN, sales_i * LP_MIN_BUFFER)
      • short_i ≥ ideal_i – x_i
      • short_i ≥ 0
    
    where 
      ideal_i = sales_i / Σ sales_j * total_space,
      short_i is the positive shortfall relative to that ideal.
    """

    # 1) Aggregate
    grouped = (
        df.groupby(['Product_Name','Category'], as_index=False)
          .agg({
              'Sales_Last_30_Days':'sum',
              'Profit_Per_Unit':'mean'
          })
    )
    n = len(grouped)
    sales   = grouped['Sales_Last_30_Days'].to_numpy()
    profits = grouped['Profit_Per_Unit'].to_numpy()

    # Precompute ideal fractional allocation by sales
    total_sales = sales.sum()
    if total_sales > 0:
        ideal_frac = sales / total_sales * total_space
    else:
        ideal_frac = np.zeros(n)

    # 2) Build LP
    prob = pulp.LpProblem("BalancedShelfLP", pulp.LpMaximize)

    # Decision vars: x_i = allocation, short_i = shortfall
    x_vars     = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Integer') for i in range(n)]
    short_vars = [pulp.LpVariable(f"s_{i}", lowBound=0, cat='Continuous') for i in range(n)]

    # Objective: profit minus penalty on lost revenue
    prob += (
        pulp.lpSum(x_vars[i] * profits[i] for i in range(n))
        - LAMBDA * pulp.lpSum(short_vars[i] * profits[i] for i in range(n))
    )

    # Total space constraint
    prob += pulp.lpSum(x_vars) <= total_space

    # Per-product bounds & shortfall definitions
    for i in range(n):
        prob += x_vars[i] <= sales[i] * LP_MAX_BUFFER
        prob += x_vars[i] >= min(LP_MIN_STOCK_MIN, sales[i] * LP_MIN_BUFFER)

        # short_i ≥ ideal_frac_i − x_i
        prob += short_vars[i] >= ideal_frac[i] - x_vars[i]

    # 3) Solve
    status = prob.solve()
    if pulp.LpStatus[status] != 'Optimal':
        return None

    # 4) Extract results
    alloc   = [int(x_vars[i].value() or 0) for i in range(n)]
    short   = [float(short_vars[i].value() or 0) for i in range(n)]
    lost_rev = [short[i] * profits[i] for i in range(n)]
    total_profit = [alloc[i] * profits[i] for i in range(n)]

    # 5) Build output DataFrame
    out = grouped.copy()
    out['Allocated_Space']  = alloc
    out['Shortfall_Units']  = np.ceil(short).astype(int)
    out['Lost_Revenue']      = lost_rev
    out['Achieved_Profit']   = total_profit
    out['Objective_Value']   = out['Achieved_Profit'] - LAMBDA * out['Lost_Revenue']

    return out[[
        'Product_Name','Category',
        'Sales_Last_30_Days','Profit_Per_Unit',
        'Allocated_Space','Shortfall_Units','Lost_Revenue','Achieved_Profit','Objective_Value'
    ]]