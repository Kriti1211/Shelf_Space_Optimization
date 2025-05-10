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