import pulp
import pandas as pd
import streamlit as st

# Constants for LP optimization
TOTAL_SHELF_SPACE = 1000   # Total available shelf space units
LP_MAX_BUFFER = 1.2        # Max proportion of recent sales for upper allocation bound
LP_MIN_BUFFER = 0.2        # Min proportion of recent sales for lower allocation bound
LP_MIN_STOCK_MIN = 5       # Absolute minimum allocation per product

def optimize_lp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize shelf space allocation using linear programming (LP).
    This function maximizes total profit under shelf space and per-product constraints.

    Parameters:
    - df: DataFrame containing at least the columns:
        'Product_Name', 'Category', 'Profit_Per_Unit', 'Sales_Last_30_Days'

    Returns:
    - DataFrame with columns:
        ['Product_Name', 'Category', 'Allocated_Space', 'Profit_Per_Unit']
      where 'Allocated_Space' is integer allocation per product.
    """
    # Check for required columns
    required = ['Product_Name', 'Category', 'Profit_Per_Unit', 'Sales_Last_30_Days']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}'")

    # Initialize LP problem: maximize total profit across all SKUs
    prob = pulp.LpProblem("ShelfSpaceLP", pulp.LpMaximize)

    # Decision variables: integer allocation v[i] for each product index
    vars_ = [pulp.LpVariable(f"v{i}", lowBound=0, cat='Integer') for i in df.index]

    # Profit and sales data arrays for vectorized usage
    profits = df['Profit_Per_Unit'].to_numpy()
    sales = df['Sales_Last_30_Days'].to_numpy()

    # Objective: maximize sum(profit_per_unit * allocated_space)
    prob += pulp.lpSum(vars_[i] * profits[i] for i in df.index)

    # Constraint: total allocated space cannot exceed overall shelf space
    prob += pulp.lpSum(vars_) <= TOTAL_SHELF_SPACE

    # Per-product lower and upper bound constraints
    for i in df.index:
        # Upper bound: up to LP_MAX_BUFFER * recent sales
        prob += vars_[i] <= sales[i] * LP_MAX_BUFFER
        # Lower bound: at least LP_MIN_STOCK_MIN or LP_MIN_BUFFER * sales
        prob += vars_[i] >= min(LP_MIN_STOCK_MIN, sales[i] * LP_MIN_BUFFER)

    # Solve the LP problem
    status = prob.solve()
    # Notify if non-optimal solution is returned
    if pulp.LpStatus[status] != 'Optimal':
        st.warning(f"LP status: {pulp.LpStatus[status]}")

    # Extract allocations from decision variables
    allocations = [int(v.value() or 0) for v in vars_]

    # Merge allocations with original DataFrame
    result = df.copy()
    result['Allocated_Space'] = allocations

    # Return only the essential columns
    return result[['Product_Name', 'Category', 'Allocated_Space', 'Profit_Per_Unit']]