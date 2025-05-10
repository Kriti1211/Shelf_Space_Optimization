# import numpy as np
# import pandas as pd

# # Constants for Genetic Algorithm
# LP_MAX_BUFFER    = 1.2       # Multiplier for max allocation based on recent sales
# LP_MIN_BUFFER    = 0.2       # Multiplier for min allocation based on recent sales
# LP_MIN_STOCK_MIN = 5         # Minimum units per SKU

# GA_POPULATION_SIZE = 50      # Number of individuals in each generation
# GA_GENERATIONS     = 100     # Total generations to evolve
# GA_MUTATION_RATE   = 0.1     # Probability of mutation per offspring

# def optimize_ga(df: pd.DataFrame, total_space: int) -> pd.DataFrame:
#     """
#     Optimize shelf space allocation using a simple genetic algorithm (GA),
#     after grouping by Product_Name & Category:
    
#     1) Group raw df by ['Product_Name','Category']:
#          - Sales_Last_30_Days := sum
#          - Profit_Per_Unit   := mean
#     2) Compute Total_Profit (the score) = Profit_Per_Unit * Sales_Last_30_Days.
#     3) Initialize a GA population of random allocations summing to total_space.
#     4) Evolve over GA_GENERATIONS via tournament selection, crossover, mutation,
#        with fitness = total_profit - penalties for constraint violations.
#     5) Return the best allocation with columns:
#        ['Product_Name','Category','Allocated_Space',
#         'Profit_Per_Unit','Sales_Last_30_Days','Total_Profit'].
#     """
#     np.random.seed(12345)

#     # 1) Aggregate to product/category level
#     grouped = (
#         df
#         .groupby(['Product_Name', 'Category'], as_index=False)
#         .agg({
#             'Sales_Last_30_Days': 'sum',
#             'Profit_Per_Unit': 'mean'
#         })
#     )
#     # 2) Compute score
#     grouped['Total_Profit'] = (
#         grouped['Profit_Per_Unit'] * grouped['Sales_Last_30_Days']
#     )

#     # Convert to numpy arrays
#     profits = grouped['Profit_Per_Unit'].to_numpy()
#     sales   = grouped['Sales_Last_30_Days'].to_numpy()
#     n       = len(grouped)

#     def create_individual() -> np.ndarray:
#         """Random allocation vector summing to total_space."""
#         weights    = np.random.rand(n)
#         alloc_frac = weights / weights.sum() * total_space
#         return np.floor(alloc_frac).astype(int)

#     def fitness(ind: np.ndarray) -> float:
#         """Fitness = total profit minus penalties for violations."""
#         total_profit = ind.dot(profits)
#         penalty = 0.0
#         total_alloc = ind.sum()
#         # Over-capacity penalty
#         if total_alloc > total_space:
#             penalty += (total_alloc - total_space) * 100
#         # Bounds penalties
#         over  = ind > sales * LP_MAX_BUFFER
#         under = ind < np.minimum(LP_MIN_STOCK_MIN, sales * LP_MIN_BUFFER)
#         penalty += (over.sum() + under.sum()) * 100
#         return total_profit - penalty

#     # 3) Initialize population
#     population = [create_individual() for _ in range(GA_POPULATION_SIZE)]

#     # 4) Evolve
#     for _ in range(GA_GENERATIONS):
#         # Evaluate fitness
#         scores = np.array([fitness(ind) for ind in population])

#         # Tournament selection
#         parents = []
#         for _ in range(GA_POPULATION_SIZE):
#             contenders = np.random.choice(GA_POPULATION_SIZE, 3, replace=False)
#             winner     = contenders[np.argmax(scores[contenders])]
#             parents.append(population[winner])

#         # Crossover & mutation
#         next_pop = []
#         for i in range(0, GA_POPULATION_SIZE, 2):
#             p1, p2 = parents[i], parents[i+1]
#             cp = np.random.randint(n)
#             # Single‐point crossover
#             c1 = np.concatenate([p1[:cp], p2[cp:]])
#             c2 = np.concatenate([p2[:cp], p1[cp:]])
#             # Mutations
#             if np.random.rand() < GA_MUTATION_RATE:
#                 c1[np.random.randint(n)] = np.random.randint(total_space)
#             if np.random.rand() < GA_MUTATION_RATE:
#                 c2[np.random.randint(n)] = np.random.randint(total_space)
#             next_pop.extend([c1, c2])
#         population = next_pop

#     # 5) Pick best individual
#     best_alloc = max(population, key=fitness).astype(int)

#     # Build result DataFrame
#     result = grouped.copy()
#     result['Allocated_Space'] = best_alloc

#     return result[[
#         'Product_Name',
#         'Category',
#         'Allocated_Space',
#         'Profit_Per_Unit',
#         'Sales_Last_30_Days',
#         'Total_Profit'
#     ]]

# utils/optimization_methods/genetic_algorithm.py

import numpy as np
import pandas as pd

# GA hyperparameters
GA_POPULATION_SIZE = 50
GA_GENERATIONS     = 100
GA_MUTATION_RATE   = 0.1

# Penalty / tradeoff
LAMBDA = 1.0  # weight on lost revenue in fitness


def optimize_ga(df: pd.DataFrame, total_space: int) -> pd.DataFrame:
    # 1) Group and compute metrics
    grouped = (
        df.groupby(['Product_Name','Category'], as_index=False)
          .agg({
              'Sales_Last_30_Days':'sum',
              'Profit_Per_Unit':'mean'
          })
    )
    # Total profit per SKU
    grouped['Total_Profit'] = (
        grouped['Profit_Per_Unit'] * grouped['Sales_Last_30_Days']
    )
    # Balanced score to avoid extremes
    grouped['Balanced_Score'] = np.sqrt(grouped['Total_Profit'])

    # Precompute ideal fractional allocation by sales
    sales = grouped['Sales_Last_30_Days'].to_numpy()
    total_sales = sales.sum()
    if total_sales > 0:
        ideal_frac = sales / total_sales * total_space
    else:
        ideal_frac = np.zeros_like(sales)

    profits = grouped['Profit_Per_Unit'].to_numpy()
    n = len(grouped)

    # 2) GA individual creation
    def create_individual() -> np.ndarray:
        # initialize biased by balanced score
        weights = grouped['Balanced_Score'].to_numpy()
        # add a bit of noise so population varies
        weights = weights + np.random.rand(n) * weights.mean() * 0.1
        alloc_frac = weights / weights.sum() * total_space
        return np.floor(alloc_frac).astype(int)

    # 3) Fitness with lost-revenue penalty + constraint penalties
    def fitness(ind: np.ndarray) -> float:
        total_profit = ind.dot(profits)
        # lost revenue
        lost_units = np.maximum(ideal_frac - ind, 0)
        total_lost = float((lost_units * profits).sum())
        # capacity/bounds penalties
        penalty = 0.0
        total_alloc = ind.sum()
        if total_alloc > total_space:
            penalty += (total_alloc - total_space) * 100
        # simple lower/upper bounds (you can tune these)
        over  = ind > sales * 1.2
        under = ind < np.minimum(5, sales * 0.2)
        penalty += (over.sum() + under.sum()) * 100
        # combined objective
        return total_profit - LAMBDA * total_lost - penalty

    # 4) Initialize population
    population = [create_individual() for _ in range(GA_POPULATION_SIZE)]

    # 5) Evolve
    for _ in range(GA_GENERATIONS):
        # evaluate fitness
        scores = np.array([fitness(ind) for ind in population])

        # tournament selection
        parents = []
        for _ in range(GA_POPULATION_SIZE):
            c = np.random.choice(GA_POPULATION_SIZE, 3, replace=False)
            parents.append(population[c[np.argmax(scores[c])]])

        # crossover & mutation
        new_pop = []
        for i in range(0, GA_POPULATION_SIZE, 2):
            p1, p2 = parents[i], parents[i+1]
            cp = np.random.randint(n)
            c1 = np.concatenate([p1[:cp], p2[cp:]])
            c2 = np.concatenate([p2[:cp], p1[cp:]])
            # mutation
            if np.random.rand() < GA_MUTATION_RATE:
                idx = np.random.randint(n)
                c1[idx] = np.random.randint(total_space)
            if np.random.rand() < GA_MUTATION_RATE:
                idx = np.random.randint(n)
                c2[idx] = np.random.randint(total_space)
            new_pop.extend([c1, c2])
        population = new_pop

    # 6) Select best
    best = max(population, key=fitness)

    # 7) Build result DataFrame
    result = grouped.copy()
    result['Allocated_Space']    = best
    # recompute for report
    result['Lost_Units']         = np.maximum(ideal_frac - best, 0).astype(int)
    result['Lost_Revenue']       = result['Lost_Units'] * result['Profit_Per_Unit']
    result['Net_Objective']      = (
        result['Allocated_Space'] * result['Profit_Per_Unit']
        - LAMBDA * result['Lost_Revenue']
    )

    return result[[
        'Product_Name','Category',
        'Sales_Last_30_Days','Profit_Per_Unit',
        'Balanced_Score','Allocated_Space',
        'Lost_Units','Lost_Revenue','Net_Objective'
    ]]